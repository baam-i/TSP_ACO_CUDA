@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set CUDA_HOME=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\nvvm\bin;%CUDA_PATH%\nvvm\bin\x64;%PATH%
"%~dp0venv\Scripts\python.exe" "%~dp0tsp_aco.py" --cuda --file "%~dp0cities.txt" %*
