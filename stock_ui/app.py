"""Local stock selector and future-session forecasting app for Kronos."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Callable
import urllib.parse
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from market_data import DataError, FEATURES, SHANGHAI, TradingCalendar, fetch_latest, normalize_symbol
from model_loader import MODEL_CONTEXT_LIMITS, ModelLoadError, configure_network, load_pair

APP_VERSION = "2.2.0"
APP_ID = "kronos-stock-forecast-v2"
LOOKBACK_OPTIONS = (128, 256, 400, 512, 1024, 1536, 2048)
HISTORY_OPTIONS = (500, 1000, 2000, 5000)


class ModelRunner:
    def __init__(self):
        self.models = {}

    def predict(self, frame: pd.DataFrame, future: list, model_key: str,
                progress: Callable[[str], None]) -> pd.DataFrame:
        if model_key not in MODEL_CONTEXT_LIMITS or len(frame) > MODEL_CONTEXT_LIMITS[model_key]:
            raise DataError("模型输入超过官方上下文上限，请缩短预测输入窗口。")
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer
        if model_key not in self.models:
            # Keep one resident predictor; model switching reuses weights on disk.
            self.models.clear()
            progress("正在校验官方模型配置与本机权重…")
            try:
                tokenizer, model = load_pair(model_key, KronosTokenizer, Kronos, progress)
                self.models[model_key] = KronosPredictor(model, tokenizer, device="cpu", max_context=MODEL_CONTEXT_LIMITS[model_key])
            except ModelLoadError:
                raise
            except Exception as error:
                raise DataError("模型推理器初始化失败。详情：" + type(error).__name__ + ": " + str(error)[:260]) from error
        progress(f"正在用最新{len(frame)}根日线预测后续{len(future)}个交易日…")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        torch.manual_seed(42)
        with torch.no_grad():
            return self.models[model_key].predict(
                df=frame[FEATURES].copy(), x_timestamp=frame["timestamps"].reset_index(drop=True),
                y_timestamp=pd.Series(pd.to_datetime(future)), pred_len=len(future),
                T=1.0, top_p=0.9, sample_count=3, verbose=False,
            )


def validate_options(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise DataError("请求格式不正确。")
    symbol, _ = normalize_symbol(payload.get("symbol", ""))
    horizon = payload.get("horizon", 5)
    lookback = payload.get("lookback", 256)
    model_key = payload.get("model", "small")
    history_count = payload.get("history_count", 5000)
    if type(horizon) is not int or horizon not in (1, 3, 5, 10, 20):
        raise DataError("预测长度请选择1、3、5、10或20个交易日。")
    if type(lookback) is not int or lookback not in LOOKBACK_OPTIONS:
        raise DataError("预测输入窗口请选择128、256、400、512、1024、1536或2048根。")
    if model_key not in MODEL_CONTEXT_LIMITS:
        raise DataError("请选择Kronos-mini、Kronos-small或Kronos-base。")
    if lookback > MODEL_CONTEXT_LIMITS[model_key]:
        raise DataError(f"Kronos-{model_key}最多输入{MODEL_CONTEXT_LIMITS[model_key]}根日线；"
                        "5000根历史可下载、浏览和导出，不会全部送入本次预测。")
    if type(history_count) is not int or history_count not in HISTORY_OPTIONS or history_count < lookback:
        raise DataError("历史行情请选择500、1000、2000或5000根，且不能少于预测输入根数。")
    return {"symbol": symbol, "horizon": horizon, "lookback": lookback, "model": model_key,
            "history_count": history_count}


def make_forecast(options: dict, progress: Callable[[str], None],
                  loader=fetch_latest, runner=None, now: datetime | None = None) -> dict:
    calendar = TradingCalendar()
    history_frame, metadata = loader(options["symbol"], options["lookback"], now=now, progress=progress,
                                     calendar=calendar, history_count=options.get("history_count", 5000))
    frame = history_frame.tail(options["lookback"]).reset_index(drop=True).copy()
    if len(frame) != options["lookback"] or len(frame) > MODEL_CONTEXT_LIMITS[options["model"]]:
        raise DataError("实际预测输入根数不足或超过模型上限，未执行推理。")
    # Validate metadata against the requested symbol, not a previous UI selection.
    if metadata["symbol"] != options["symbol"]:
        raise DataError("股票身份校验失败，已取消该预测。")
    last = frame["timestamps"].iloc[-1].date()
    future = calendar.following(last, options["horizon"])
    prediction = (runner or ModelRunner()).predict(frame, future, options["model"], progress)
    if len(prediction) != len(future) or "close" not in prediction:
        raise DataError("模型输出长度或字段异常。")
    closes = prediction["close"].to_numpy(dtype=float)
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise DataError("模型输出包含无效价格，本次预测不能展示。")
    last_close = float(frame["close"].iloc[-1])
    forecasts = [{"date": day.isoformat(), "close": float(value), "change_pct": float((value / last_close - 1) * 100)}
                 for day, value in zip(future, closes)]
    history = [{"date": row.timestamps.date().isoformat(), "open": float(row.open), "high": float(row.high),
                "low": float(row.low), "close": float(row.close), "volume": float(row.volume), "amount": float(row.amount)}
               for row in history_frame.itertuples(index=False)]
    return {**metadata, "history": history, "forecast": forecasts, "last_close": last_close,
            "horizon": options["horizon"], "lookback": options["lookback"], "model": "Kronos-" + options["model"],
            "input_rows": len(frame), "input_start": frame["timestamps"].iloc[0].date().isoformat(),
            "input_end": last.isoformat(), "model_context_limit": MODEL_CONTEXT_LIMITS[options["model"]],
            "sample_count": 3, "generated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "calendar_through": calendar.last.isoformat()}


class JobStore:
    def __init__(self, loader=fetch_latest, runner=None, now=None, results_dir: Path | None = None):
        self.loader = loader
        self.runner = runner or ModelRunner()
        self.now = now
        self.results_dir = results_dir or HERE.parent / "results"
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None

    def submit(self, options: dict) -> str:
        with self.lock:
            if self.active is not None:
                raise DataError("已有预测正在运行，请等待当前任务完成。")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"id": job_id, "status": "running", "message": "正在准备行情…", "symbol": options["symbol"]}
            self.active = job_id
            while len(self.jobs) > 20:
                self.jobs.pop(next(iter(self.jobs)))
        threading.Thread(target=self._work, args=(job_id, options), daemon=True).start()
        return job_id

    def _work(self, job_id: str, options: dict) -> None:
        def progress(message: str):
            with self.lock:
                self.jobs[job_id]["message"] = message
        try:
            result = make_forecast(options, progress, loader=self.loader, runner=self.runner, now=self.now)
            self.results_dir.mkdir(exist_ok=True)
            (self.results_dir / (job_id + ".json")).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            with self.lock:
                self.jobs[job_id].update(status="done", message="预测完成", result=result)
        except Exception as error:
            logging.exception("Forecast failed for %s", options["symbol"])
            with self.lock:
                self.jobs[job_id].update(status="error", message=str(error)[:900])
        finally:
            with self.lock:
                self.active = None

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            item = self.jobs.get(job_id)
            return dict(item) if item else None


def make_handler(store: JobStore):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self) -> bool:
            port = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def send(self, status: int, data: bytes, content_type: str = "application/json; charset=utf-8", extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def json(self, status: int, obj: dict):
            self.send(status, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def do_GET(self):
            if not self.valid_host():
                return self.json(403, {"error": "仅允许本机访问。"})
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/":
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parsed.path == "/api/health":
                return self.json(200, {"app": APP_ID, "version": APP_VERSION, "ready": True,
                                       "calendar_through": TradingCalendar().last.isoformat(),
                                       "download_source": configure_network()["label"]})
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})(/download)?", parsed.path)
            if match:
                item = store.get(match[1])
                if not item:
                    return self.json(404, {"error": "找不到该任务。"})
                if match[2]:
                    if item["status"] != "done":
                        return self.json(409, {"error": "预测尚未成功完成。"})
                    result = item["result"]
                    kind = urllib.parse.parse_qs(parsed.query).get("kind", ["forecast"])[0]
                    if kind not in ("history", "input", "forecast"):
                        return self.json(400, {"error": "不支持的导出类型。"})
                    rows = (result["history"] if kind == "history" else
                            result["history"][-result["input_rows"]:] if kind == "input" else result["forecast"])
                    buf = io.StringIO(newline="")
                    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                    name = result["symbol"] + "_" + kind + "_" + result["as_of"] + ".csv"
                    return self.send(200, buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                                     {"Content-Disposition": f'attachment; filename="{name}"'})
                return self.json(200, item)
            return self.json(404, {"error": "地址不存在。"})

        def do_POST(self):
            if not self.valid_host():
                return self.json(403, {"error": "仅允许本机访问。"})
            port = self.server.server_address[1]
            if self.headers.get("Origin") not in {None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                return self.json(403, {"error": "来源不允许。"})
            if self.path != "/api/jobs":
                return self.json(404, {"error": "地址不存在。"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise DataError("请求长度不正确。")
                payload = json.loads(self.rfile.read(length))
                options = validate_options(payload)
                job_id = store.submit(options)
                return self.json(202, {"id": job_id})
            except (ValueError, TypeError) as error:
                return self.json(400, {"error": str(error)})

        def log_message(self, fmt, *args):
            logging.info("HTTP " + fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7070)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    TradingCalendar().latest_complete()  # Fail clearly if the calendar needs an update.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(JobStore()))
    logging.info("Kronos stock forecast v%s: http://127.0.0.1:%s", APP_VERSION, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
