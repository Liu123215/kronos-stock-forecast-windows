"""Run Node UI checks against an isolated fixture server; no real weights used."""
from pathlib import Path
import logging
import os
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_base_history import app, long_loader, NOW, TestRunner

logging.basicConfig(level=logging.CRITICAL)
with tempfile.TemporaryDirectory(prefix="kronos-ui-check-") as folder:
    store = app.JobStore(loader=long_loader, runner=TestRunner(), now=NOW, results_dir=Path(folder))
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment["KRONOS_QA_URL"] = "http://127.0.0.1:" + str(server.server_port)
    try:
        result = subprocess.run(["node", "tests/check_ui.js"], cwd=ROOT, env=environment, timeout=60)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    raise SystemExit(result.returncode)
