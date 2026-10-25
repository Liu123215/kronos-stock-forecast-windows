"""Base and long-history checks; all fixtures stay outside the shipped ZIP."""
from contextlib import nullcontext
from datetime import date
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_ui"))
import app
import market_data as md
import model_loader as ml
from test_loader import fake_class
from test_market import NOW, CAL, TestRunner


def long_bars(count=6200, code="600089"):
    dates = pd.bdate_range(end="2026-10-23", periods=count)
    price = np.linspace(3.0, 12.0, count) + (5 if code == "002185" else 0)
    return pd.DataFrame({"timestamps": dates, "open": price - .02, "close": price,
                         "high": price + .08, "low": price - .08, "volume": 100000.,
                         "amount": price * 100000})


def long_loader(symbol, lookback, now=None, progress=None, calendar=None, history_count=None):
    if symbol == "600099":
        raise md.DataError("测试：行情源不可用，未生成预测。")
    provider = lambda *args: (long_bars(code=symbol), "QA长历史")
    return md.fetch_latest(symbol, lookback, now=NOW, progress=progress, calendar=calendar,
                           history_count=history_count, providers=[("QA完整来源", provider)])


class Tests(unittest.TestCase):
    def test_base_configuration_and_pinned_weight(self):
        config, spec = ml.read_spec("Kronos-base", fake_class("Kronos"))
        self.assertEqual((config["d_model"], config["n_layers"], config["n_heads"]), (832, 12, 16))
        self.assertEqual(spec["size_bytes"], 409264008)
        self.assertEqual(spec["sha256"], "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83")
        self.assertEqual(spec["revision"], "31afc02ed0a3cfaa1a97238c8b948d663e45b597")
        with patch.object(ml, "load_component", side_effect=["tokenizer", "base"]) as load:
            self.assertEqual(ml.load_pair("base", object, object, lambda _: None), ("tokenizer", "base"))
        self.assertEqual([c.args[0] for c in load.call_args_list], ["Kronos-Tokenizer-base", "Kronos-base"])

    def test_input_limits_no_silent_truncation(self):
        for key in ("base", "small"):
            self.assertEqual(app.validate_options({"symbol": "600089", "model": key, "lookback": 512})["lookback"], 512)
            with self.assertRaisesRegex(md.DataError, "最多输入512"):
                app.validate_options({"symbol": "600089", "model": key, "lookback": 1024})
        self.assertEqual(app.validate_options({"symbol": "600089", "model": "mini", "lookback": 2048})["lookback"], 2048)
        with self.assertRaises(md.DataError):
            app.validate_options({"symbol": "600089", "model": "mini", "lookback": 5000})
        with self.assertRaises(md.DataError):
            app.validate_options({"symbol": "600089", "model": "mini", "lookback": 2048, "history_count": 1000})

    def test_5000_archive_and_512_base_input(self):
        runner = TestRunner()
        result = app.make_forecast(app.validate_options({"symbol": "600089", "model": "base", "lookback": 512,
                                                        "history_count": 5000, "horizon": 5}), lambda _: None,
                                   loader=long_loader, runner=runner, now=NOW)
        self.assertEqual(len(result["history"]), 5000)
        self.assertEqual(len(runner.calls[0][0]), 512)
        self.assertEqual(result["model"], "Kronos-base")
        self.assertEqual(result["input_rows"], 512)
        self.assertEqual(result["input_start"], runner.calls[0][0]["timestamps"].iloc[0].date().isoformat())
        self.assertEqual(result["history_start"], long_bars().tail(5000)["timestamps"].iloc[0].date().isoformat())
        self.assertEqual(result["history_end"], result["input_end"])
        self.assertEqual(result["forecast"][0]["date"], "2026-10-26")

    def test_2048_mini_input_stays_daily(self):
        runner = TestRunner()
        result = app.make_forecast(app.validate_options({"symbol": "600089", "model": "mini", "lookback": 2048}),
                                   lambda _: None, loader=long_loader, runner=runner, now=NOW)
        self.assertEqual(len(result["history"]), 5000)
        self.assertEqual(len(runner.calls[0][0]), 2048)
        self.assertEqual(result["input_rows"], 2048)
        self.assertEqual(runner.calls[0][0]["timestamps"].iloc[-1], pd.Timestamp("2026-10-23"))

    def test_history_shortage_visible_but_not_padded(self):
        provider = lambda *args: (long_bars(600), "short")
        frame, meta = md.fetch_latest("600089", 512, now=NOW, history_count=5000, providers=[("short", provider)])
        self.assertEqual(len(frame), 600)
        self.assertIn("本次数据源可用600根", meta["history_note"])
        with self.assertRaises(md.DataError):
            md.fetch_latest("600089", 1024, now=NOW, history_count=5000, providers=[("short", provider)])

    def test_latest_zero_turnover_cannot_silently_drop_latest_day(self):
        frame = long_bars()
        frame.loc[frame.index[-1], "amount"] = 0
        with self.assertRaisesRegex(md.DataError, "最新交易日"):
            md.validate_bars(frame, date(2026, 10, 23), 512, 5000)

    def test_provider_with_longer_fresh_history_wins(self):
        short = lambda *args: (long_bars(600), "short")
        full = lambda *args: (long_bars(), "full")
        frame, meta = md.fetch_latest("600089", 256, now=NOW, history_count=5000,
                                     providers=[("short", short), ("full", full)])
        self.assertEqual(len(frame), 5000)
        self.assertEqual(meta["source"], "full")
        stale = lambda *args: (long_bars().iloc[:-1], "stale")
        frame, meta = md.fetch_latest("600089", 256, now=NOW, history_count=5000,
                                     providers=[("short", short), ("stale", stale)])
        self.assertEqual(len(frame), 600)
        self.assertEqual(meta["source"], "short")

    def test_old_nonpositive_qfq_is_not_used_as_model_input(self):
        frame = long_bars()
        frame.loc[:1500, ["open", "high", "low", "close"]] -= 10
        selected = md.validate_bars(frame, date(2026, 10, 23), 512, 5000)
        self.assertEqual(len(selected), 5000)
        self.assertLess(selected["close"].iloc[0], 0)
        self.assertGreater(selected["close"].tail(512).min(), 0)

    def test_tencent_long_history_segments_are_bounded_and_deduplicated(self):
        calls = []
        def payload(url, params, **kwargs):
            calls.append(params["param"])
            symbol, _, begin, end, limit, adjust = params["param"].split(",")
            dates = pd.bdate_range(begin, end)
            self.assertLess(len(dates), int(limit))
            bars = [[day.date().isoformat(), "10", "10", "11", "9", "1000", None, "1", "100"] for day in dates]
            return {"data": {symbol: {"qfqday": bars}}}
        with patch.object(md, "get_json", side_effect=payload):
            frame, _ = md.tencent("600089", "sh", date(2000, 1, 1), date(2026, 10, 23))
        checked = md.validate_bars(frame, date(2026, 10, 23), 512, 5000)
        self.assertEqual(len(checked), 5000)
        self.assertEqual(len(calls), 14)
        self.assertFalse(checked["timestamps"].duplicated().any())
        self.assertEqual(float(checked["volume"].iloc[-1]), 100000.)

    def test_tencent_rejects_unadjusted_section(self):
        response = {"data": {"sh600089": {"day": [["2026-10-23", "10", "10", "11", "9", "1000", None, "1", "100"]]}}}
        with patch.object(md, "get_json", return_value=response):
            with self.assertRaisesRegex(md.DataError, "未复权"):
                md.tencent("600089", "sh", date(2026, 1, 1), date(2026, 10, 23))

    def test_base_runner_uses_512_context_and_keeps_one_resident_model(self):
        captured = []
        class Predictor:
            def __init__(self, model, tokenizer, **kwargs):
                captured.append(kwargs)
            def predict(self, **kwargs):
                return pd.DataFrame({"close": [float(kwargs["df"]["close"].iloc[-1])]})
        torch = types.SimpleNamespace(set_num_threads=lambda _: None, manual_seed=lambda _: None, no_grad=nullcontext)
        model = types.SimpleNamespace(Kronos=object, KronosTokenizer=object, KronosPredictor=Predictor)
        runner = app.ModelRunner()
        with patch.dict(sys.modules, {"torch": torch, "model": model}), patch.object(app, "load_pair", return_value=(object(), object())):
            runner.predict(long_bars(512), [date(2026, 10, 26)], "base", lambda _: None)
            self.assertEqual(list(runner.models), ["base"])
            self.assertEqual(captured[-1]["max_context"], 512)
            runner.predict(long_bars(256), [date(2026, 10, 26)], "mini", lambda _: None)
            self.assertEqual(list(runner.models), ["mini"])
            self.assertEqual(captured[-1]["max_context"], 2048)
        with self.assertRaises(md.DataError):
            runner.predict(long_bars(513), [], "base", lambda _: None)


if __name__ == "__main__":
    unittest.main()
