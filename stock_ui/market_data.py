"""Fresh A-share daily bars, validated before use by Kronos.

Public endpoint conventions are documented by AKShare. This adapter uses only
the standard library and pandas, so existing Kronos installations need no new
market-data package. It never substitutes demonstration or cached prices.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
import logging
from pathlib import Path
import re
from statistics import median
from typing import Callable
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

SHANGHAI = timezone(timedelta(hours=8))
HERE = Path(__file__).resolve().parent
PRICE_COLUMNS = ["open", "high", "low", "close"]
FEATURES = PRICE_COLUMNS + ["volume", "amount"]
KNOWN_NAMES = {"601868": "中国能建", "002185": "华天科技", "000157": "中联重科",
               "600089": "特变电工", "601012": "隆基绿能", "300750": "宁德时代"}


class DataError(ValueError):
    pass


def normalize_symbol(value: str) -> tuple[str, str]:
    value = str(value).strip().lower()
    value = KNOWN_NAMES_REVERSE.get(value, value)
    match = re.fullmatch(r"(?:(sh|sz))?([036]\d{5})(?:\.(sh|sz))?", value)
    if not match:
        raise DataError("请输入6位沪深A股代码，例如601868、002185或600089。")
    before, code, after = match.groups()
    market = "sh" if code.startswith("6") else "sz"
    if (before and before != market) or (after and after != market):
        raise DataError("股票代码与交易所前缀不一致，请直接输入6位代码。")
    return code, market


KNOWN_NAMES_REVERSE = {value: key for key, value in KNOWN_NAMES.items()}


class TradingCalendar:
    def __init__(self, path: Path | None = None):
        data = json.loads((path or HERE / "trading_calendar.json").read_text(encoding="utf-8"))
        self.first = date.fromisoformat(data["valid_from"])
        self.last = date.fromisoformat(data["valid_through"])
        self.holidays = {date.fromisoformat(value) for value in data["holidays"]}

    def is_session(self, value: date) -> bool:
        if not self.first <= value <= self.last:
            raise DataError("交易日历只核实到%s，请更新启动包后再预测该日期。" % self.last)
        return value.weekday() < 5 and value not in self.holidays

    def latest_complete(self, now: datetime | None = None) -> date:
        current = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
        day = current.date()
        # A short publication buffer prevents incomplete end-of-day bars.
        if current.time().replace(tzinfo=None) < time(15, 15):
            day -= timedelta(days=1)
        for _ in range(40):
            if self.is_session(day):
                return day
            day -= timedelta(days=1)
        raise DataError("无法确定最近已收盘交易日。")

    def following(self, after: date, count: int) -> list[date]:
        result = []
        day = after
        while len(result) < count:
            day += timedelta(days=1)
            if self.is_session(day):
                result.append(day)
        return result


def get_json(url: str, params: dict, timeout: int = 12) -> dict:
    request = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise DataError("行情响应异常大，请稍后重试。")
    text = raw.decode("utf-8-sig")
    # Tencent can wrap otherwise standard JSON in a variable assignment.
    begin = text.find("{")
    if begin < 0:
        raise DataError("行情接口没有返回可解析的数据。")
    return json.loads(text[begin:].strip().rstrip(";"))


def eastmoney(code: str, market: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    payload = get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {
        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989", "klt": "101", "fqt": "1",
        "secid": ("1." if market == "sh" else "0.") + code,
        "beg": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    })
    data = payload.get("data") or {}
    if data.get("code") and str(data["code"]) != code:
        raise DataError("行情返回的股票代码与请求不符。")
    rows = []
    for line in data.get("klines") or []:
        fields = line.split(",")
        if len(fields) < 7:
            raise DataError("东方财富行情字段不完整。")
        rows.append(dict(zip(["timestamps", "open", "close", "high", "low", "volume", "amount"], fields[:7])))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100.0  # hands -> shares
    return frame, str(data.get("name") or KNOWN_NAMES.get(code) or code)


def tencent(code: str, market: str, start: date, end: date, progress=None) -> tuple[pd.DataFrame, str]:
    symbol = market + code
    rows = []
    name = KNOWN_NAMES.get(code, code)
    for year in range(start.year, end.year + 1, 2):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(end, date(year + 1, 12, 31))
        if progress:
            progress(f"正在从腾讯财经分段下载{code}：{chunk_start}至{chunk_end}…")
        payload = get_json("https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get", {
            "param": f"{symbol},day,{chunk_start.isoformat()},{chunk_end.isoformat()},640,qfq",
        })
        data = (payload.get("data") or {}).get(symbol) or {}
        quote = (data.get("qt") or {}).get(symbol)
        if isinstance(quote, list) and len(quote) > 1:
            name = str(quote[1])
        # Never mix unadjusted old sections into a series labelled qfq.
        bars = data.get("qfqday") or []
        if not bars and data.get("day"):
            raise DataError("腾讯接口只返回未复权日线，不能与前复权历史拼接。")
        for values in bars:
            if len(values) < 9:
                raise DataError("腾讯行情缺少真实成交额字段，不能作为模型输入。")
            if not chunk_start.isoformat() <= str(values[0])[:10] <= chunk_end.isoformat():
                continue
            rows.append({"timestamps": values[0], "open": values[1], "close": values[2],
                         "high": values[3], "low": values[4], "volume": values[5],
                         "amount": float(values[8]) * 10000.0})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # Vendor volume conventions differ by board. Confirm units against real
        # turnover on recent bars instead of assuming a stock prefix is an index.
        latest = frame.sort_values("timestamps").drop_duplicates("timestamps").tail(20)
        ratios = (latest["amount"] / (latest["volume"] * latest["close"])).replace([np.inf, -np.inf], np.nan).dropna()
        ratios = ratios[ratios > 0]
        if ratios.empty:
            raise DataError("腾讯成交量单位无法确认。")
        scale = median(ratios.tolist())
        if 30 < scale < 300:
            frame["volume"] *= 100.0
        elif not 0.3 < scale < 3:
            raise DataError("腾讯成交量与成交额单位不匹配，已停止使用该响应。")
    return frame, name


def validate_bars(frame: pd.DataFrame, target: date, lookback: int,
                  history_count: int | None = None) -> pd.DataFrame:
    columns = ["timestamps"] + FEATURES
    if frame.empty or not set(columns).issubset(frame.columns):
        raise DataError("没有取得该股票的完整日线数据，请检查代码及网络。")
    frame = frame[columns].copy()
    frame["timestamps"] = pd.to_datetime(frame["timestamps"], errors="coerce").dt.normalize()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamps").drop_duplicates("timestamps", keep="last")
    frame = frame[frame["timestamps"] <= pd.Timestamp(target)]
    if frame.empty:
        raise DataError("没有可用的已收盘日线。")
    last_date = frame["timestamps"].iloc[-1].date()
    if last_date != target:
        raise DataError(f"数据仅更新到{last_date}，最近已收盘交易日为{target}；可能停牌、退市或接口未更新。")
    if not np.isfinite(frame[FEATURES].tail(lookback).to_numpy(dtype=float)).all():
        raise DataError("最近行情含有缺失值或无效数值，不能生成预测。")
    if float(frame["volume"].iloc[-1]) <= 0 or float(frame["amount"].iloc[-1]) <= 0:
        raise DataError("最新交易日没有有效成交量或成交额，可能停牌或行情不完整，暂不生成预测。")
    frame = frame[(frame["volume"] > 0) & (frame["amount"] > 0)].tail(history_count or lookback).reset_index(drop=True)
    if len(frame) < lookback:
        raise DataError(f"只有{len(frame)}根有效日线，当前需要{lookback}根；可调低历史窗口后重试。")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise DataError("输入行情仍含有无效数值，不能生成预测。")
    prices = frame[PRICE_COLUMNS]
    if (frame["high"] + 1e-6 < prices.max(axis=1)).any() or (frame["low"] - 1e-6 > prices.min(axis=1)).any():
        raise DataError("历史行情价格关系异常，已停止预测。")
    if (prices.tail(lookback) <= 0).any().any():
        raise DataError("预测输入窗口中包含非正的前复权价格，请缩短输入窗口后重试。")
    return frame


def fetch_latest(symbol: str, lookback: int, now: datetime | None = None,
                 progress: Callable[[str], None] | None = None,
                 providers=None, calendar: TradingCalendar | None = None,
                 history_count: int | None = None) -> tuple[pd.DataFrame, dict]:
    code, market = normalize_symbol(symbol)
    calendar = calendar or TradingCalendar()
    target = calendar.latest_complete(now)
    requested = history_count if history_count is not None else lookback
    if type(requested) is not int or not lookback <= requested <= 5000:
        raise DataError("历史行情根数需不少于预测输入根数，且最多5000根。")
    start = max(date(1990, 1, 1), target - timedelta(days=requested * 2 + 120))
    providers = providers or [("东方财富", eastmoney),
                              ("腾讯财经", lambda *args: tencent(*args, progress=progress))]
    errors = []
    best = None
    for label, fetch in providers:
        if progress:
            progress(f"正在从{label}下载{code}最多{requested}根历史日线，要求更新到{target}…")
        try:
            raw, name = fetch(code, market, start, target)
            frame = validate_bars(raw, target, lookback, requested)
            metadata = {"symbol": code, "market": market, "name": name[:40], "source": label,
                           "as_of": target.isoformat(), "adjustment": "前复权", "rows": len(frame),
                           "history_requested": requested, "history_available": len(frame),
                           "history_start": frame["timestamps"].iloc[0].date().isoformat(),
                           "history_end": target.isoformat(),
                           "fetched_at": datetime.now(SHANGHAI).isoformat(timespec="seconds")}
            if best is None or len(frame) > len(best[0]):
                best = (frame, metadata)
            if len(frame) >= requested:
                return frame, metadata
            if progress:
                progress(f"{label}本次可用{len(frame)}根，继续检查另一数据源的历史覆盖…")
        except Exception as error:
            logging.warning("%s %s: %s", label, code, error)
            errors.append(label + "：" + str(error)[:220])
    if best is not None:
        frame, metadata = best
        metadata["history_note"] = (f"请求{requested}根，本次数据源可用{len(frame)}根；"
                                    "上市时间或接口历史覆盖可能不足，未补造数据。")
        return frame, metadata
    raise DataError("未取得最新有效行情。" + "；".join(errors))
