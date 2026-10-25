@echo off
setlocal
title Kronos - Third-party Public Mirror
set "KRONOS_DOWNLOAD_ENDPOINT=https://hf-mirror.com"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_ETAG_TIMEOUT=60"
set "HF_HUB_DOWNLOAD_TIMEOUT=300"
set "HF_HUB_DISABLE_XET=1"
echo Download source: hf-mirror.com - third-party public mirror.
echo Official SHA256 verification is required. No API token is sent.
call "%~dp0START_KRONOS.cmd"
exit /b %ERRORLEVEL%
