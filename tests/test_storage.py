"""Storage regressions; fixture bytes and model doubles are never shipped."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_ui"))
import model_loader as ml
from test_loader import fake_class


class StorageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kronos-save-qa-")
        self.base = Path(self.temp.name)
        self.configs = self.base / "configs"
        shutil.copytree(ml.CONFIG_DIR, self.configs)
        self.data = b"fixture-only-weight-bytes-not-a-real-checkpoint"
        self.source = self.base / "downloaded.safetensors"
        self.source.write_bytes(self.data)
        self.name = "Kronos-Tokenizer-2k"
        self.models = self.base / "Chinese path 模型文件"
        self.target = self.models / self.name / "model.safetensors"
        self.specs = json.loads((self.configs / "manifest.json").read_text())
        self.spec = self.specs[self.name]
        self.spec.update(size_bytes=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())
        (self.configs / "manifest.json").write_text(json.dumps(self.specs))
        self.Tokenizer = fake_class("KronosTokenizer")
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def download(self, **kwargs):
        self.calls.append(kwargs)
        return str(self.source)

    def load(self, downloader=None):
        return ml.load_component(self.name, self.Tokenizer, lambda _: None,
                                 models_dir=self.models, config_dir=self.configs,
                                 cache_root=self.base / "hub", downloader=downloader or self.download)

    def snapshot(self):
        repository = self.base / "hub" / ("models--NeoQuasar--" + self.name)
        blob = repository / "blobs" / self.spec["sha256"]
        blob.parent.mkdir(parents=True)
        blob.write_bytes(self.data)
        snapshot = repository / "snapshots" / self.spec["revision"] / "model.safetensors"
        snapshot.parent.mkdir(parents=True)
        snapshot.symlink_to("../../blobs/" + self.spec["sha256"])
        return blob, snapshot

    def test_downloaded_relative_snapshot_link_saves_real_bytes(self):
        blob, snapshot = self.snapshot()
        # Simulate a newly returned HF snapshot, not the pre-download cache path.
        with patch.object(ml, "cache_candidates", return_value=[]):
            obj = self.load(downloader=lambda **_: str(snapshot))
        self.assertTrue(obj.evaluated)
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.target.samefile(blob))

    def test_cached_snapshot_and_blob_can_be_removed_after_save(self):
        blob, snapshot = self.snapshot()
        obj = self.load()
        self.assertTrue(obj.evaluated)
        self.assertEqual(self.calls, [])
        snapshot.unlink()
        blob.unlink()
        self.assertTrue(self.load().evaluated)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(self.calls, [])

    def test_snapshot_without_blob_candidate_is_reused_offline(self):
        blob, snapshot = self.snapshot()
        renamed = blob.with_name("actual-weight-file")
        blob.rename(renamed)
        snapshot.unlink()
        snapshot.symlink_to("../../blobs/actual-weight-file")
        self.assertTrue(self.load().evaluated)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.target.is_symlink())

    def test_ordinary_file_is_an_independent_copy(self):
        self.assertTrue(self.load().evaluated)
        self.source.write_bytes(b"cache changed later")
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertTrue(ml.file_matches(self.target, self.spec))

    def test_broken_existing_link_is_preserved_and_repaired(self):
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to("old-missing-cache-target")
        self.assertTrue(self.load().evaluated)
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(self.target.read_bytes(), self.data)
        backups = list(self.target.parent.glob("model.safetensors.invalid-*"))
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0].is_symlink())

    def test_bad_saved_bytes_do_not_replace_previous_file(self):
        self.target.parent.mkdir(parents=True)
        previous = b"preserve previous local file"
        self.target.write_bytes(previous)
        def corrupt(reader, writer, **kwargs):
            writer.write(b"short")
        with patch.object(ml.shutil, "copyfileobj", side_effect=corrupt):
            with self.assertRaises(ml.ModelLoadError) as raised:
                self.load()
        self.assertEqual(raised.exception.stage, "integrity")
        self.assertIn(self.spec["sha256"], str(raised.exception))
        self.assertIn("5字节", str(raised.exception))
        self.assertIn(str(self.target), str(raised.exception))
        self.assertEqual(self.target.read_bytes(), previous)
        self.assertEqual(list(self.target.parent.glob("*.tmp-*")), [])
        self.assertEqual(list(self.target.parent.glob("*.invalid-*")), [])

    def test_disk_write_error_is_identified_and_keeps_previous_file(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"previous")
        with patch.object(ml.shutil, "copyfileobj", side_effect=PermissionError("test file locked")):
            with self.assertRaises(ml.ModelLoadError) as raised:
                self.load()
        self.assertEqual(raised.exception.stage, "storage")
        self.assertIn("PermissionError", str(raised.exception))
        self.assertIn(str(self.target), str(raised.exception))
        self.assertEqual(self.target.read_bytes(), b"previous")
        self.assertEqual(list(self.target.parent.glob("*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
