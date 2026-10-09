@echo off
cd /d "%~dp0..\..\.."
start "" /b py -3 -m http.server 8000 --bind 127.0.0.1
start "" "http://127.0.0.1:8000/rag_core/retrieval_fusion/visualization/"
