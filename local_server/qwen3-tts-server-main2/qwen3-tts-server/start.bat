@echo off
set PYTHONIOENCODING=utf-8
set PATH=%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin;%~dp0.venv\Lib\site-packages\nvidia\cublas\bin;%~dp0.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin;%PATH%
cd /d %~dp0
.venv\python.exe -m server.main
