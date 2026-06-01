@echo off
"%~dp0venv\Scripts\python.exe" "%~dp0tsp_aco.py" --cpu --file "%~dp0cities.txt" %*
