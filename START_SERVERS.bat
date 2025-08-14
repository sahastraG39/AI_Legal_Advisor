@echo off
echo ========================================
echo AI Legal Document Explainer - Startup Script
echo ========================================
echo.

echo Starting Backend Server (Port 8000)...
start "Backend API" cmd /k "cd /d %~dp0 && uvicorn src.api:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Waiting 10 seconds for backend to start...
timeout /t 10 /nobreak > nul

echo.
echo Starting Frontend Server (Port 3000)...
start "Frontend React" cmd /k "cd /d %~dp0\frontend && npm start"

echo.
echo ========================================
echo Both servers are starting up!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend Website: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul
