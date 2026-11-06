@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)
".venv\Scripts\python.exe" app.py %*
exit /b %errorlevel%
:fail
echo Setup failed.
pause
exit /b 1
