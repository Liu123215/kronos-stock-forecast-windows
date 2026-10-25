"""Validated, local-first model loading with bundled official configurations."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import uuid

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "model_configs"
OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"
MODEL_CONTEXT_LIMITS = {"mini": 2048, "small": 512, "base": 512}


def configure_network() -> dict:
    def timeout_value(key: str, minimum: int, maximum: int) -> int:
        try:
            value = int(os.environ.get(key, str(minimum)))
        except ValueError:
            value = minimum
        value = min(maximum, max(minimum, value))
        os.environ[key] = str(value)
        return value
    etag = timeout_value("HF_HUB_ETAG_TIMEOUT", 60, 300)
    body = timeout_value("HF_HUB_DOWNLOAD_TIMEOUT", 300, 1800)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    endpoint = os.environ.get("KRONOS_DOWNLOAD_ENDPOINT") or os.environ.get("HF_ENDPOINT") or OFFICIAL_ENDPOINT
    endpoint = endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("模型下载地址必须是有效的HTTPS站点，不能包含账号或令牌。")
    os.environ["HF_ENDPOINT"] = endpoint
    label = "Hugging Face 官方" if endpoint == OFFICIAL_ENDPOINT else "HF-Mirror（第三方镜像）" if endpoint == MIRROR_ENDPOINT else "自定义下载站点 " + parsed.hostname
    return {"endpoint": endpoint, "label": label, "etag_timeout": etag, "download_timeout": body}


# Set timeout environment variables before the model core imports HF constants.
configure_network()


class ModelLoadError(ValueError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


def read_spec(name: str, model_class, config_dir: Path | None = None) -> tuple[dict, dict]:
    directory = config_dir or CONFIG_DIR
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        spec = manifest[name]
        config = json.loads((directory / (name + ".json")).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as error:
        raise ModelLoadError("configuration", f"{name}的随包官方配置缺失或损坏，请重新解压修复包。详情：{error}") from error
    if not isinstance(config, dict):
        raise ModelLoadError("configuration", f"{name}配置必须是JSON对象。")
    parameters = inspect.signature(model_class.__init__).parameters
    required = [key for key, param in parameters.items()
                if key != "self" and param.default is inspect.Parameter.empty
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    missing = [key for key in required if key not in config]
    if missing:
        raise ModelLoadError("configuration", f"{name}配置缺少参数：{', '.join(missing)}；尚未尝试创建模型。")
    unexpected = set(config) - set(parameters)
    if unexpected:
        raise ModelLoadError("configuration", f"{name}配置与模型代码不匹配，未知字段：{', '.join(sorted(unexpected))}。")
    return config, spec


def file_matches(path: Path, spec: dict) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != spec["size_bytes"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == spec["sha256"]
    except OSError:
        return False


def file_summary(path: Path) -> str:
    """Return read-back evidence only when verification has failed."""
    try:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        return f"{size}字节，SHA256={digest.hexdigest()}"
    except OSError as error:
        return f"无法读取（{type(error).__name__}: {error}）"


def save_verified_weights(name: str, source: Path, weights: Path, spec: dict, progress) -> None:
    """Copy actual cache bytes into an independent file, then atomically install."""
    temporary = weights.parent / ("model.safetensors.tmp-" + uuid.uuid4().hex)
    try:
        # HF snapshots can be relative symlinks. Linking that directory entry
        # into another folder can create a dangling link on Windows and Linux.
        # Resolve it, then copy bytes so the saved file also survives cache edits.
        actual_source = source.resolve(strict=True)
        progress(f"{name}源文件已校验，正在复制真实权重并验证本机保存结果…")
        with actual_source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if not file_matches(temporary, spec):
            raise ModelLoadError("integrity", f"{name}权重保存校验失败。目标：{weights}；"
                                 f"应为{spec['size_bytes']}字节，SHA256={spec['sha256']}；"
                                 f"保存结果：{file_summary(temporary)}。原有文件未替换。")
        if weights.exists() or weights.is_symlink():
            backup = weights.parent / ("model.safetensors.invalid-" + str(time.time_ns()))
            weights.replace(backup)
        temporary.replace(weights)
    except (OSError, RuntimeError) as error:
        raise ModelLoadError("storage", f"{name}权重在本机保存失败。目标：{weights}；"
                             f"详情：{type(error).__name__}: {error}") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.warning("Could not remove temporary weight file: %s", temporary)


def cache_candidates(name: str, spec: dict, cache_root: Path | None = None):
    if cache_root is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_root = Path(HF_HUB_CACHE)
        except ImportError:
            return []
    repository = cache_root / ("models--NeoQuasar--" + name)
    # An existing main snapshot can reference the same LFS blob even when the
    # pinned revision has never been resolved on this machine.
    return [repository / "blobs" / spec["sha256"],
            repository / "snapshots" / spec["revision"] / "model.safetensors"]


def transient_download_error(error: Exception) -> bool:
    seen = set()
    current = error
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status in (408, 429) or (isinstance(status, int) and 500 <= status <= 599):
            return True
        if isinstance(status, int) and 400 <= status <= 499:
            return False
        description = (type(current).__name__ + " " + str(current)).lower()
        if isinstance(current, (TimeoutError, ConnectionError)) or any(word in description for word in
            ("readtimeout", "connecttimeout", "timed out", "connectionerror", "connection reset", "connection aborted", "remotedisconnected")):
            return True
        current = current.__cause__ or current.__context__
    return False


def download_weights(name: str, spec: dict, progress, downloader=None, *, sleep_fn=None) -> Path:
    settings = configure_network()
    sleep_fn = sleep_fn or time.sleep
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        except ImportError as error:
            raise ModelLoadError("dependency", "缺少huggingface_hub依赖，请用修复包的启动文件重新启动。") from error
    progress(f"官方配置已校验；从{settings['label']}下载{name}权重（约{spec['size_bytes']/1024/1024:.1f} MB）…")
    for force in (False, True):
        for attempt in range(1, 4):
            progress(f"{settings['label']}：下载{name}，第{attempt}/3次尝试；元数据等待{settings['etag_timeout']}秒，文件读取超时{settings['download_timeout']}秒…")
            try:
                path = Path(downloader(repo_id="NeoQuasar/" + name, filename="model.safetensors",
                                       revision=spec["revision"], token=False, force_download=force,
                                       etag_timeout=settings["etag_timeout"], endpoint=settings["endpoint"]))
                break
            except Exception as error:
                if attempt < 3 and transient_download_error(error):
                    delay = 2 ** attempt
                    cache_note = "保留下载缓存" if not force else "重新获取未通过校验的权重"
                    progress(f"{settings['label']}连接超时或临时中断，{delay}秒后重试；{cache_note}…")
                    logging.warning("%s attempt %s failed: %s", name, attempt, error)
                    sleep_fn(delay)
                    continue
                hint = "可关闭窗口，改用START_KRONOS_MIRROR.cmd（第三方镜像），或按说明手动放置权重。" if settings["endpoint"] == OFFICIAL_ENDPOINT else "可改用START_KRONOS.cmd官方入口，或按说明手动放置权重。"
                raise ModelLoadError("download", f"下载{name}/model.safetensors失败，来源：{settings['label']}，已尝试{attempt}次；配置已完整。{hint}详情：{type(error).__name__}: {str(error)[:220]}") from error
        if file_matches(path, spec):
            return path
        logging.warning("%s weight integrity check failed; force_download=%s", name, force)
        if not force:
            progress(f"{name}缓存权重不完整，正在重新下载该文件…")
    raise ModelLoadError("integrity", f"{name}权重的大小或SHA256与官方版本不一致，已停止加载。")


def load_component(name: str, model_class, progress, *, models_dir: Path | None = None,
                   config_dir: Path | None = None, cache_root: Path | None = None, downloader=None):
    config, spec = read_spec(name, model_class, config_dir)
    models_dir = models_dir or HERE.parent.parent / "model_files"
    local = models_dir / name
    try:
        local.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ModelLoadError("storage", f"无法创建{name}的模型目录：{local}；详情：{error}") from error
    weights = local / "model.safetensors"
    if not file_matches(weights, spec):
        source = next((candidate for candidate in cache_candidates(name, spec, cache_root)
                       if file_matches(candidate, spec)), None)
        if source is None:
            source = download_weights(name, spec, progress, downloader)
        save_verified_weights(name, source, weights, spec, progress)
    progress(f"{name}权重完整，正在从本机配置和权重加载…")
    temporary_config = local / ("config.json.tmp-" + uuid.uuid4().hex)
    temporary_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary_config.replace(local / "config.json")
    try:
        # Explicit constructor kwargs are essential: a remote missing config must
        # never be silently converted into cls() with all required args missing.
        instance = model_class.from_pretrained(str(local), local_files_only=True, strict=True, **config)
        instance.eval()
        return instance
    except Exception as error:
        raise ModelLoadError("initialization", f"{name}配置和权重均已校验，但模型初始化失败。详情：{type(error).__name__}: {str(error)[:260]}") from error


def load_pair(model_key: str, tokenizer_class, model_class, progress):
    if model_key not in MODEL_CONTEXT_LIMITS:
        raise ModelLoadError("configuration", "请选择Kronos-mini、Kronos-small或Kronos-base。")
    tokenizer_name = "Kronos-Tokenizer-2k" if model_key == "mini" else "Kronos-Tokenizer-base"
    tokenizer = load_component(tokenizer_name, tokenizer_class, progress)
    model = load_component("Kronos-" + model_key, model_class, progress)
    return tokenizer, model
