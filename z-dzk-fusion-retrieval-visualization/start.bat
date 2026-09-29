@echo off
setlocal
cd /d "%~dp0"
py -3 "%~dp0serve_ui.py" %*
if errorlevel 1 pause
