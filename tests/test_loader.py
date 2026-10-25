"""Regression checks for the user's exact missing-16-arguments failure.

Constructor signatures come from the real bundled Kronos code. Test checkpoint
bytes and fake classes are only used here, never shipped in the application.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_ui"))
import model_loader as ml


def fake_class(name):
    tree = ast.parse((ROOT / "model/kronos.py").read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    init = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    args = ast.unparse(init.args)
    text = f'''class ExactSignature:
    def __init__({args}):
        self.arguments = {{key:value for key,value in locals().items() if key != 'self'}}
        self.evaluated = False
    @classmethod
    def from_pretrained(cls, path, local_files_only=True, strict=True, **kwargs):
        assert local_files_only and strict
        assert Path(path, 'config.json').is_file()
        assert Path(path, 'model.safetensors').is_file()
        return cls(**kwargs)
    def eval(self):
        self.evaluated = True
'''
    scope = {"Path": Path}
    exec(text, scope)
    return scope["ExactSignature"]


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kronos-loader-qa-")
        self.base = Path(self.temp.name)
        self.configs = self.base / "configs"
        shutil.copytree(ml.CONFIG_DIR, self.configs)
        self.weights = self.base / "downloaded.safetensors"
        self.weights.write_bytes(b"test-only-checkpoint-bytes")
        self.specs = json.loads((self.configs / "manifest.json").read_text())
        for spec in self.specs.values():
            spec.update(size_bytes=self.weights.stat().st_size, sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest())
        (self.configs / "manifest.json").write_text(json.dumps(self.specs))
        self.calls = []
        self.Tokenizer = fake_class("KronosTokenizer")
        self.Model = fake_class("Kronos")
    def tearDown(self):
        self.temp.cleanup()
    def download(self, **kwargs):
        self.calls.append(kwargs)
        self.assertEqual(kwargs["filename"], "model.safetensors")
        self.assertFalse(kwargs["token"])
        return str(self.weights)
    def load(self, name="Kronos-Tokenizer-base", cls=None, downloader=None, cache_root=None):
        return ml.load_component(name, cls or self.Tokenizer, lambda _: None,
                                 models_dir=self.base / "models", config_dir=self.configs,
                                 cache_root=cache_root or self.base / "cache", downloader=downloader or self.download)

    def test_original_failure_reproduced_and_fixed(self):
        with self.assertRaisesRegex(TypeError, "16 required positional arguments"):
            self.Tokenizer()
        obj = self.load()
        self.assertEqual(len(obj.arguments), 16)
        self.assertEqual(obj.arguments["d_in"], 6)
        self.assertEqual(obj.arguments["group_size"], 4)
        self.assertTrue(obj.evaluated)
        self.assertEqual(len(self.calls), 1)

    def test_all_official_configs_match_real_signatures(self):
        for name, cls in [("Kronos-Tokenizer-base", self.Tokenizer), ("Kronos-Tokenizer-2k", self.Tokenizer),
                          ("Kronos-small", self.Model), ("Kronos-mini", self.Model)]:
            config, spec = ml.read_spec(name, cls)
            self.assertTrue(config)
            self.assertEqual(len(spec["sha256"]), 64)
            self.assertEqual(len(spec["revision"]), 40)
        obj = self.load("Kronos-Tokenizer-2k")
        self.assertEqual(obj.arguments["group_size"], 5)
        obj = self.load("Kronos-small", self.Model)
        self.assertEqual(obj.arguments["d_model"], 512)

    def test_missing_config_stops_before_constructor_or_network(self):
        path = self.configs / "Kronos-Tokenizer-base.json"
        data = json.loads(path.read_text()); del data["d_in"]
        path.write_text(json.dumps(data))
        with self.assertRaises(ml.ModelLoadError) as raised:
            self.load()
        self.assertEqual(raised.exception.stage, "configuration")
        self.assertIn("d_in", str(raised.exception))
        self.assertEqual(self.calls, [])

    def test_download_error_is_not_reported_as_constructor_failure(self):
        def failed(**kwargs):raise ConnectionError("test offline")
        with self.assertRaises(ml.ModelLoadError) as raised:self.load(downloader=failed)
        self.assertEqual(raised.exception.stage, "download")
        self.assertIn("配置已完整", str(raised.exception))
        self.assertNotIn("missing 16", str(raised.exception))

    def test_cached_weights_require_no_download(self):
        spec = self.specs["Kronos-Tokenizer-base"]
        cached = self.base / "cache/models--NeoQuasar--Kronos-Tokenizer-base/blobs" / spec["sha256"]
        cached.parent.mkdir(parents=True);shutil.copyfile(self.weights, cached)
        obj = self.load()
        self.assertTrue(obj.evaluated)
        self.assertEqual(self.calls, [])
        obj = self.load()
        self.assertTrue(obj.evaluated)
        self.assertEqual(self.calls, [])

    def test_corrupt_weight_is_redownloaded_before_initialization(self):
        bad = self.base / "bad.safetensors";bad.write_bytes(b"bad")
        flags = []
        def fetch(**kwargs):
            flags.append(kwargs["force_download"])
            return str(self.weights if kwargs["force_download"] else bad)
        obj=self.load(downloader=fetch)
        self.assertTrue(obj.evaluated)
        self.assertEqual(flags,[False,True])

    def test_weight_mismatch_and_initialization_are_distinct_errors(self):
        bad=self.base/'bad.safetensors';bad.write_bytes(b'bad')
        with self.assertRaises(ml.ModelLoadError) as raised:self.load(downloader=lambda **_:str(bad))
        self.assertEqual(raised.exception.stage,'integrity')
        class Broken(self.Tokenizer):
            @classmethod
            def from_pretrained(cls,*args,**kwargs):raise RuntimeError('test state mismatch')
        with self.assertRaises(ml.ModelLoadError) as raised:self.load(cls=Broken)
        self.assertEqual(raised.exception.stage,'initialization')


if __name__ == "__main__":unittest.main()
