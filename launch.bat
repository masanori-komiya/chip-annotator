@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_env.bat" app.py %*
if errorlevel 1 pause
