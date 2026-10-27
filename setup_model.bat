@echo off
setlocal
cd /d "%~dp0"
call "%~dp0environment.bat"
if errorlevel 1 goto fail
if not exist "%CA_PY%" (
  echo Run setup.bat first.
  goto fail
)
set "TORCH_FLAVOR=cu128"
if /i "%~1"=="cpu" set "TORCH_FLAVOR=cpu"
"%CA_PY%" tools\install_torch.py --flavor %TORCH_FLAVOR%
if errorlevel 1 goto fail
"%CA_PY%" -m pip install -r requirements-model.txt
if errorlevel 1 goto fail
"%CA_PY%" tools\install_mobilesam.py
if errorlevel 1 goto fail
"%CA_PY%" tools\diagnose.py
if errorlevel 1 goto fail
echo MobileSAM setup complete.
pause
exit /b 0
:fail
echo Model setup failed. Check the error above and README.md.
pause
exit /b 1
