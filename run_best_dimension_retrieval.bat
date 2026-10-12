@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_best_dimension_retrieval.ps1"
pause
