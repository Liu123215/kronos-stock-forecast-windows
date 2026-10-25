"""Development tests for timeout handling and explicit mirror selection."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stock_ui'))
import model_loader as ml


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='kronos-timeout-qa-')
        self.weights=Path(self.temp.name)/'model.safetensors'
        self.weights.write_bytes(b'test-only-weight-file')
        self.spec={'revision':'1'*40,'sha256':hashlib.sha256(self.weights.read_bytes()).hexdigest(),
                   'size_bytes':self.weights.stat().st_size}
    def tearDown(self):self.temp.cleanup()

    def test_timeouts_raise_10s_to_longer_values(self):
        with patch.dict(os.environ,{'HF_HUB_ETAG_TIMEOUT':'10','HF_HUB_DOWNLOAD_TIMEOUT':'10',
                                   'KRONOS_DOWNLOAD_ENDPOINT':ml.OFFICIAL_ENDPOINT}):
            settings=ml.configure_network()
            self.assertEqual(settings['etag_timeout'],60)
            self.assertEqual(settings['download_timeout'],300)
            self.assertEqual(os.environ['HF_HUB_DOWNLOAD_TIMEOUT'],'300')

    def test_timeout_retries_reuse_partial_cache_and_explicit_timeout(self):
        calls=[];delays=[]
        def download(**kwargs):
            calls.append(kwargs)
            if len(calls)<3:raise TimeoutError('Read timed out. read timeout=10')
            return str(self.weights)
        with patch.dict(os.environ,{'KRONOS_DOWNLOAD_ENDPOINT':ml.OFFICIAL_ENDPOINT}):
            path=ml.download_weights('Kronos-mini',self.spec,lambda _:None,downloader=download,sleep_fn=delays.append)
        self.assertEqual(path,self.weights)
        self.assertEqual(delays,[2,4])
        self.assertEqual(len(calls),3)
        self.assertTrue(all(not call['force_download'] for call in calls))
        self.assertTrue(all(call['etag_timeout']>=60 for call in calls))
        self.assertTrue(all(call['token'] is False for call in calls))

    def test_retry_limit_has_usable_error(self):
        calls=[]
        def download(**kwargs):calls.append(kwargs);raise ConnectionError('test disconnected')
        with patch.dict(os.environ,{'KRONOS_DOWNLOAD_ENDPOINT':ml.OFFICIAL_ENDPOINT}):
            with self.assertRaises(ml.ModelLoadError) as raised:
                ml.download_weights('Kronos-mini',self.spec,lambda _:None,downloader=download,sleep_fn=lambda _:None)
        self.assertEqual(len(calls),3)
        self.assertIn('START_KRONOS_MIRROR.cmd',str(raised.exception))
        self.assertEqual(raised.exception.stage,'download')

    def test_mirror_is_explicit_and_unauthenticated(self):
        calls=[]
        def download(**kwargs):calls.append(kwargs);return str(self.weights)
        with patch.dict(os.environ,{'KRONOS_DOWNLOAD_ENDPOINT':ml.MIRROR_ENDPOINT}):
            settings=ml.configure_network()
            ml.download_weights('Kronos-mini',self.spec,lambda _:None,downloader=download,sleep_fn=lambda _:None)
        self.assertIn('第三方',settings['label'])
        self.assertEqual(calls[0]['endpoint'],ml.MIRROR_ENDPOINT)
        self.assertFalse(calls[0]['token'])
        self.assertEqual(calls[0]['revision'],self.spec['revision'])

    def test_mirror_payload_still_requires_official_hash(self):
        bad=Path(self.temp.name)/'bad.safetensors';bad.write_bytes(b'wrong')
        with patch.dict(os.environ,{'KRONOS_DOWNLOAD_ENDPOINT':ml.MIRROR_ENDPOINT}):
            with self.assertRaises(ml.ModelLoadError) as raised:
                ml.download_weights('Kronos-mini',self.spec,lambda _:None,downloader=lambda **_:str(bad),sleep_fn=lambda _:None)
        self.assertEqual(raised.exception.stage,'integrity')

    def test_missing_file_is_not_retried(self):
        class MissingError(Exception):
            response=type('Response',(),{'status_code':404})()
        calls=[]
        def download(**kwargs):calls.append(kwargs);raise MissingError('not found')
        with self.assertRaises(ml.ModelLoadError):
            ml.download_weights('Kronos-mini',self.spec,lambda _:None,downloader=download,sleep_fn=lambda _:None)
        self.assertEqual(len(calls),1)

    def test_wrapped_timeout_is_detected(self):
        try:
            try:raise TimeoutError('timeout')
            except TimeoutError as error:raise OSError('cannot locate file') from error
        except OSError as error:self.assertTrue(ml.transient_download_error(error))


if __name__=='__main__':unittest.main()
