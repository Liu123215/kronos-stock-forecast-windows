"""Development verification only. This file is not included in the user package."""
import importlib.util
from datetime import date, datetime, timedelta
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_ui"))
import app
import market_data as md

NOW = datetime(2026, 10, 24, 18, tzinfo=md.SHANGHAI)
CAL = md.TradingCalendar()


def bars(code="601868"):
    end = date(2026, 10, 23)
    day = date(2024, 5, 1)
    dates = []
    while day <= end:
        if CAL.is_session(day):
            dates.append(day)
        day += timedelta(days=1)
    offset = 2.0 if code == "601868" else 15.0
    closes = np.linspace(offset, offset + 0.8, len(dates))
    return pd.DataFrame({"timestamps": pd.to_datetime(dates), "open": closes-.02, "high": closes+.08,
                         "low": closes-.06, "close": closes, "volume": 10000., "amount": closes*10000})


def test_loader(symbol, lookback, now=None, progress=None, calendar=None, history_count=None):
    if symbol == "600099":
        raise md.DataError("测试：行情源不可用，未生成预测。")
    provider = lambda code, market, start, end: (bars(code), "QA测试·" + ("中国能建" if code=="601868" else "华天科技"))
    return md.fetch_latest(symbol, lookback, now=NOW, progress=progress,
                           calendar=calendar, providers=[("QA 测试数据", provider)], history_count=history_count)


class TestRunner:
    def __init__(self):
        self.calls = []
    def predict(self, frame, future, model, progress):
        self.calls.append((frame.copy(), list(future)))
        last = float(frame["close"].iloc[-1])
        return pd.DataFrame({"close": [last*(1+.004*(i+1)) for i in range(len(future))]})


class Tests(unittest.TestCase):
    def test_calendar_weekend_and_holiday(self):
        self.assertEqual(CAL.latest_complete(NOW), date(2026,10,23))
        self.assertEqual(CAL.following(date(2026,10,23),5), [date(2026,10,i) for i in range(26,31)])
        self.assertEqual(CAL.following(date(2026,9,30),3), [date(2026,10,8),date(2026,10,9),date(2026,10,12)])
        self.assertEqual(CAL.latest_complete(datetime(2026,10,26,14,tzinfo=md.SHANGHAI)),date(2026,10,23))
        with self.assertRaises(md.DataError):
            CAL.following(date(2026,12,31),1)

    def test_latest_tail_and_stock_switch(self):
        runner = TestRunner()
        one = app.make_forecast(app.validate_options({"symbol":"601868","lookback":128,"horizon":5}),lambda _:None,
                                loader=test_loader,runner=runner,now=NOW)
        two = app.make_forecast(app.validate_options({"symbol":"002185","lookback":128,"horizon":3}),lambda _:None,
                                loader=test_loader,runner=runner,now=NOW)
        self.assertEqual(one["symbol"],"601868")
        self.assertEqual(two["symbol"],"002185")
        self.assertNotEqual(one["last_close"],two["last_close"])
        self.assertEqual(runner.calls[0][0]["timestamps"].iloc[0], bars().tail(128)["timestamps"].iloc[0])
        self.assertEqual(one["forecast"][0]["date"],"2026-10-26")
        self.assertTrue(all(day>date(2026,10,23) for day in runner.calls[0][1]))

    def test_stale_and_suspended_rejected(self):
        stale = bars().iloc[:-1]
        with self.assertRaisesRegex(md.DataError,"数据仅更新到"):
            md.validate_bars(stale,date(2026,10,23),128)
        frame=bars();frame.loc[frame.index[-1],"volume"]=0
        with self.assertRaisesRegex(md.DataError,"停牌"):
            md.validate_bars(frame,date(2026,10,23),128)

    def test_fallback_and_no_demo_fallback(self):
        old = lambda *args:(bars().iloc[:-1],"old")
        new = lambda *args:(bars(),"new")
        frame,meta=md.fetch_latest("601868",128,now=NOW,providers=[("old",old),("new",new)])
        self.assertEqual(meta["source"],"new")
        with self.assertRaisesRegex(md.DataError,"未取得最新有效行情"):
            md.fetch_latest("601868",128,now=NOW,providers=[("old",old)])

    def test_input_identity_and_model_output(self):
        for value in ["123", "sh002185", "../../secret", "900001", "510300"]:
            with self.assertRaises(md.DataError):md.normalize_symbol(value)
        self.assertEqual(md.normalize_symbol("002185.SZ"),("002185","sz"))
        self.assertEqual(md.normalize_symbol("中国能建"),("601868","sh"))
        class BadRunner:
            def predict(self,*args):return pd.DataFrame({"close":[float('nan')]*5})
        with self.assertRaisesRegex(md.DataError,"无效价格"):
            app.make_forecast(app.validate_options({"symbol":"601868","lookback":128}),lambda _:None,
                              loader=test_loader,runner=BadRunner(),now=NOW)

    def test_eastmoney_payload_units_and_identity(self):
        original=md.get_json
        try:
            md.get_json=lambda *args,**kwargs:{"data":{"code":"601868","name":"test","klines":["2026-10-23,2,2.1,2.2,1.9,500,105000,0,0,0,0"]}}
            frame,name=md.eastmoney("601868","sh",date(2026,1,1),date(2026,10,23))
            self.assertEqual(float(frame["volume"].iloc[0]),50000)
            md.get_json=lambda *args,**kwargs:{"data":{"code":"002185","klines":[]}}
            with self.assertRaises(md.DataError):md.eastmoney("601868","sh",date(2026,1,1),date(2026,10,23))
        finally:md.get_json=original

    def test_tencent_units(self):
        original=md.get_json
        try:
            md.get_json=lambda *args,**kwargs:{"data":{"sz002185":{"qfqday":[["2026-10-23","10","10","11","9","1000",None,"1","100"]],"qt":{"sz002185":["51","test"]}}}}
            frame,_=md.tencent("002185","sz",date(2026,1,1),date(2026,10,23))
            self.assertEqual(float(frame["volume"].iloc[0]),100000)
            self.assertEqual(float(frame["amount"].iloc[0]),1000000)
        finally:md.get_json=original


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    if "--serve" in sys.argv:
        temp = tempfile.TemporaryDirectory(prefix="kronos-ui-qa-")
        store=app.JobStore(loader=test_loader,runner=TestRunner(),now=NOW,results_dir=Path(temp.name))
        server=app.ThreadingHTTPServer(("127.0.0.1",8781),app.make_handler(store))
        print("QA fixture server: http://127.0.0.1:8781",flush=True)
        server.serve_forever()
    else:
        unittest.main()
