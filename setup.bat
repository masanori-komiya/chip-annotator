@echo off
setlocal
cd /d "%~dp0"
call "%~dp0environment.bat"
if errorlevel 1 goto fail
py -3.12 -c "import sys,struct; sys.exit(not (sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8))" >nul 2>&1
if errorlevel 1 (
  python -c "import sys,struct; sys.exit(not (sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8))" >nul 2>&1
  if errorlevel 1 goto python_missing
  set "CA_BOOT=python"
) else (
  set "CA_BOOT=py -3.12"
)
%CA_BOOT% --version
if not exist "%CA_HOME%" mkdir "%CA_HOME%"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%CHIPANNOTATOR_CACHE%" mkdir "%CHIPANNOTATOR_CACHE%"
if not exist "%CA_PY%" (
  %CA_BOOT% -m venv "%CA_ENV%"
  if errorlevel 1 goto env_failed
)
"%CA_PY%" -m pip install --upgrade pip
if errorlevel 1 goto dependency_failed
"%CA_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto dependency_failed
echo Desktop setup complete. Run launch.bat. For SAM run setup_model.bat.
echo Environment: "%CA_ENV%"
pause
exit /b 0
:python_missing
echo Python 3.12 x64 was not found through py or python. Install it and retry.
goto fail
:env_failed
echo Could not create the Python environment. Check permissions, free space and the error above.
goto fail
:dependency_failed
echo Dependency installation failed. Check the error above, network, free space and path length.
:fail
pause
exit /b 1
