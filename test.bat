@echo off
REM Runs the automated pytest test suite using the project's Python 3.11 virtual environment.
"%~dp0.venv\Scripts\python.exe" -m pytest %*

