@echo off
set USE_MOCK=true
echo Starting AI Exam Proctoring System (MOCK MODE)...
echo.

start cmd /k "cd backend && npm start"
timeout /t 5 /nobreak >nul

start cmd /k "cd frontend && npm start"

echo.
echo Services starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo ========================================
echo LOGIN CREDENTIALS:
echo Student: student1@exam.com / password
echo Supervisor: supervisor@exam.com / password
echo Admin: admin@exam.com / password
echo ========================================
echo.
echo Press any key to stop all services...
pause >nul

taskkill /F /FI "WINDOWTITLE eq *npm start*"
