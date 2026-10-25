"""Install the Kronos stock selector and live daily-bar forecasting app."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
import zipfile

COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
PAYLOAD_SHA256 = "412fda0245b2dbad83ee680c8ec37f1a6a34a8dc13aa02197d2f43a29116bc81"
APP_VERSION = "2.2.0"
HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "official_source.zip"
SUPPORTED = {(3, 10), (3, 11), (3, 12)}
REQUIRED = (
    "requirements.txt", "model/__init__.py", "model/kronos.py", "model/module.py",
    "stock_ui/app.py", "stock_ui/market_data.py", "stock_ui/index.html",
    "stock_ui/trading_calendar.json", "stock_ui/requirements.txt", "stock_ui/VERSION", "LICENSE",
    "stock_ui/model_loader.py", "stock_ui/model_configs/manifest.json",
    "stock_ui/model_configs/Kronos-Tokenizer-base.json", "stock_ui/model_configs/Kronos-Tokenizer-2k.json",
    "stock_ui/model_configs/Kronos-small.json", "stock_ui/model_configs/Kronos-mini.json",
    "stock_ui/model_configs/Kronos-base.json",
)


def verify_payload() -> None:
    digest = hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()
    if digest != PAYLOAD_SHA256:
        raise RuntimeError("The bundled source archive is incomplete or changed. Extract the original ZIP again.")
    with zipfile.ZipFile(PAYLOAD) as archive:
        if not set(REQUIRED).issubset(archive.namelist()):
            raise RuntimeError("The source archive is missing required files.")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or chr(92) in item.filename or ":" in item.filename:
                raise RuntimeError("Unsafe path in source archive.")
            if item.filename.endswith(".py"):
                ast.parse(archive.read(item).decode("utf-8"), filename=item.filename)
        if archive.read("stock_ui/VERSION").decode().strip() != APP_VERSION:
            raise RuntimeError("Application version does not match the launcher.")
        calendar = json.loads(archive.read("stock_ui/trading_calendar.json"))
        if not calendar.get("valid_through") or not calendar.get("holidays"):
            raise RuntimeError("The trading calendar is incomplete.")


def is_project(path: Path) -> bool:
    if not all((path / item).is_file() for item in REQUIRED):
        return False
    return (path / "stock_ui/VERSION").read_text(encoding="utf-8").strip() == APP_VERSION


def find_project(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    skip = {".git", ".venv", ".kronos-runtime", "venv", "env", "__pycache__", "node_modules", "site-packages", "logs"}
    for current, dirs, _ in os.walk(base):
        path = Path(current)
        if is_project(path):
            return path
        depth = len(path.relative_to(base).parts)
        dirs[:] = sorted(d for d in dirs if d.lower() not in skip) if depth < 4 else []
    return None


def prepare_project(base: Path) -> Path:
    existing = find_project(base)
    if existing:
        logging.info("Found existing project: %s", existing)
        return existing
    root = base / ("stock-forecast-v2-" + COMMIT[:12])
    if root.exists():
        root = base / ("stock-forecast-v2-" + COMMIT[:12] + "-" + str(time.time_ns()))
    root.mkdir(parents=True)
    with zipfile.ZipFile(PAYLOAD) as archive:
        archive.extractall(root)
    if not is_project(root):
        raise RuntimeError("Could not prepare the official source files.")
    logging.info("Prepared official source: %s", root)
    return root


def choose_base() -> Path:
    primary = Path(r"C:\AI\Kronos")
    candidates = [primary]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Kronos")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / (".kronos-write-check-" + str(time.time_ns()))
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable installation folder was found.")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    logging.info("Running: %s", subprocess.list2cmdline(command))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace", env=env) as process:
        assert process.stdout is not None
        for line in process.stdout:
            logging.info("%s", line.rstrip())
        if process.wait() != 0:
            raise RuntimeError("The command above failed. The full error is saved in the log.")


IMPORT_CHECK = (
    "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
    "import torch, pandas, numpy, einops, huggingface_hub, safetensors; "
    "from model import Kronos, KronosTokenizer, KronosPredictor; "
    "sys.path.insert(0, str(Path(sys.argv[1]) / 'stock_ui')); import app; import model_loader; "
    "assert app.APP_VERSION == '2.2.0'; "
    "print('Kronos stock forecast v2 import OK. Torch:', torch.__version__, 'Device: CPU')"
)


def prepare_python(base: Path, root: Path) -> Path:
    environment = base / ".kronos-runtime" / ("python" + str(sys.version_info.major) + str(sys.version_info.minor))
    python = environment / "Scripts" / "python.exe"
    if not python.exists():
        logging.info("Creating isolated environment: %s", environment)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirement_digest = hashlib.sha256(
        (root / "requirements.txt").read_bytes() + (root / "stock_ui/requirements.txt").read_bytes() + b"numpy-1.26.4-cpu-stock-v2"
    ).hexdigest()
    marker = environment / ("kronos-ready-" + requirement_digest + ".json")
    ready = False
    if python.is_file():
        result = subprocess.run([str(python), "-I", "-c", IMPORT_CHECK, str(root)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace")
        ready = result.returncode == 0
    if not ready:
        pip = [str(python), "-I", "-m", "pip", "--disable-pip-version-check"]
        run_command(pip + ["install", "--upgrade", "pip", "--progress-bar", "off"])
        logging.info("Installing CPU PyTorch from the official PyTorch package index. This download can be large.")
        run_command(pip + ["install", "--index-url", "https://download.pytorch.org/whl/cpu",
                           "torch>=2.2.0", "--progress-bar", "off"])
        run_command(pip + ["install", "-r", str(root / "requirements.txt"),
                           "-r", str(root / "stock_ui/requirements.txt"), "numpy==1.26.4", "--progress-bar", "off"])
        run_command(pip + ["check"])
        run_command([str(python), "-I", "-c", IMPORT_CHECK, str(root)])
        marker.write_text(json.dumps({"source": str(root), "commit_of_bundled_source": COMMIT,
                                      "python": sys.version, "requirements_sha256": requirement_digest}, indent=2), encoding="utf-8")
    else:
        logging.info("The isolated Python environment is ready; no installation is needed.")
    return python


def prepare_data(root: Path) -> None:
    configurations = root / "stock_ui/model_configs"
    models = root.parent / "model_files"
    for name in json.loads((configurations / "manifest.json").read_text(encoding="utf-8")):
        target = models / name
        target.mkdir(parents=True, exist_ok=True)
        config = target / "config.json"
        if not config.exists():
            shutil.copy2(configurations / (name + ".json"), config)
    logging.info("Model files folder: %s", models)
    logging.info("The stock selector will fetch fresh daily bars for each requested symbol.")


def choose_port() -> int:
    for port in range(7070, 7091):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Ports 7070 through 7090 are occupied. Close an old launcher and try again.")


def start_server(python: Path, root: Path, log_path: Path) -> int:
    port = choose_port()
    url = "http://127.0.0.1:" + str(port)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    command = [str(python), "-I", str(root / "stock_ui/app.py"), "--port", str(port)]
    process = subprocess.Popen(command, cwd=root / "stock_ui", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", env=env)
    def collect_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            logging.info("[server] %s", line.rstrip())
    reader = threading.Thread(target=collect_output, daemon=True)
    reader.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("The Web UI stopped during startup. Please share the log.")
            try:
                with opener.open(url + "/api/health", timeout=2) as response:
                    status = json.load(response)
                if status.get("app") != "kronos-stock-forecast-v2" or not status.get("ready"):
                    raise RuntimeError("The stock forecast application did not become ready.")
                break
            except (OSError, ValueError):
                time.sleep(0.5)
        else:
            raise RuntimeError("The Web UI did not become ready. Please share the log.")
        logging.info("READY: %s", url)
        logging.info("Enter a six-digit A-share code, then click the refresh-and-predict button.")
        logging.info("The app automatically downloads current daily bars and loads the model.")
        logging.info("The first model load downloads weights from Hugging Face and needs Internet access.")
        logging.info("Forecast dates follow the latest completed trading day, excluding weekends and known market holidays.")
        logging.info("Keep this window open. Press Ctrl+C to stop. Log: %s", log_path)
        try:
            webbrowser.open(url)
        except Exception:
            logging.info("Open the READY address manually in your browser.")
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-package", action="store_true")
    args = parser.parse_args()
    if args.verify_package:
        verify_payload()
        print("Archive hash, safe paths, Python syntax, app version, and trading calendar: OK")
        return 0
    if os.name != "nt" or sys.version_info[:2] not in SUPPORTED:
        print("This launcher needs Windows and Python 3.10, 3.11, or 3.12.")
        return 1
    log_path = None
    try:
        base = choose_base()
        logs = base / "logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / time.strftime("kronos-launcher-%Y%m%d-%H%M%S.log")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")])
        logging.info("Kronos Windows setup and launcher. Installation folder: %s", base)
        logging.info("[1/5] Verifying the bundled official source files.")
        verify_payload()
        logging.info("[2/5] Locating or preparing the project files.")
        root = prepare_project(base)
        logging.info("[3/5] Preparing the isolated Python environment.")
        python = prepare_python(base, root)
        logging.info("[4/5] Preparing the stock selector.")
        prepare_data(root)
        logging.info("[5/5] Starting the local Web UI.")
        return start_server(python, root, log_path)
    except KeyboardInterrupt:
        print("\nKronos stopped.")
        return 0
    except Exception as error:
        logging.error("ERROR: %s", error)
        if log_path:
            logging.error("Please share the error above or this log: %s", log_path)
        else:
            print("ERROR:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
