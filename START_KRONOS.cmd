@echo off
setlocal
title Kronos - Setup and Start
if not defined KRONOS_DOWNLOAD_ENDPOINT set "KRONOS_DOWNLOAD_ENDPOINT=https://huggingface.co"
set "HF_ENDPOINT=%KRONOS_DOWNLOAD_ENDPOINT%"
set "HF_HUB_ETAG_TIMEOUT=60"
set "HF_HUB_DOWNLOAD_TIMEOUT=300"
set "HF_HUB_DISABLE_XET=1"
set "KRONOS_SETUP=%~dp0setup_kronos.py"
set "KRONOS_EXIT=1"
set "KRONOS_RAN="
if not exist "%KRONOS_SETUP%" goto MISSING_FILES
if not exist "%~dp0official_source.zip" goto MISSING_FILES
where py >nul 2>nul
if errorlevel 1 goto TRY_DEFAULT
for %%V in (3.10 3.11 3.12) do (
    call :TRY_VERSION %%V
    if defined KRONOS_RAN goto DONE
)
:TRY_DEFAULT
python -I -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)" >nul 2>nul
if errorlevel 1 goto NO_PYTHON
python -I "%KRONOS_SETUP%"
set "KRONOS_EXIT=%ERRORLEVEL%"
goto DONE
:TRY_VERSION
py -%1 -I -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "KRONOS_RAN=1"
py -%1 -I "%KRONOS_SETUP%"
set "KRONOS_EXIT=%ERRORLEVEL%"
exit /b 0
:MISSING_FILES
echo Please extract the complete ZIP first, then double-click START_KRONOS.cmd.
goto DONE
:NO_PYTHON
echo Python 3.10, 3.11, or 3.12 was not found.
echo Please share this message so we can locate your Python installation.
:DONE
echo.
pause
exit /b %KRONOS_EXIT%
