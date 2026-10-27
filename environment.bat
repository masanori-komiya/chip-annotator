@echo off
rem Shared short environment, independent of the ZIP extraction path.
if defined CHIPANNOTATOR_HOME (
  set "CA_HOME=%CHIPANNOTATOR_HOME%"
) else if exist "%~dp0runtime-home.txt" (
  set /p "CA_HOME="<"%~dp0runtime-home.txt"
) else (
  set "CA_HOME=%LOCALAPPDATA%\CA"
)
set "CA_ENV=%CA_HOME%\env312"
set "CA_PY=%CA_ENV%\Scripts\python.exe"
set "CHIPANNOTATOR_CACHE=%CA_HOME%\cache"
set "TEMP=%CA_HOME%\tmp"
set "TMP=%CA_HOME%\tmp"
if not "%CA_ENV:~75%"=="" (
  echo Environment path is too long. Set CHIPANNOTATOR_HOME to a shorter writable folder.
  exit /b 1
)
exit /b 0
