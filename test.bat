@echo off
setlocal
cd /d "%~dp0"
call "%~dp0environment.bat"
if errorlevel 1 goto fail
if not exist "%CA_PY%" (
  echo Run setup.bat first.
  goto fail
)
"%CA_PY%" -m pip install -r requirements-dev.txt
if errorlevel 1 goto fail
set "QT_QPA_PLATFORM=offscreen"
"%CA_PY%" -m pytest -q tests
if errorlevel 1 goto fail
pause
exit /b 0
:fail
pause
exit /b 1
