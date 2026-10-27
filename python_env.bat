@echo off
setlocal
call "%~dp0environment.bat"
if errorlevel 1 exit /b 1
if not exist "%CA_PY%" (
  echo Run setup.bat first.
  exit /b 1
)
"%CA_PY%" %*
exit /b %errorlevel%
