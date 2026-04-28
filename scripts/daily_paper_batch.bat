@echo off
REM ============================================================
REM Daily paper trading batch — Windows Task Scheduler wrapper
REM
REM Schedules:
REM   schtasks /create /tn "llm-quant paper batch" /tr "E:\llm-quant\scripts\daily_paper_batch.bat" /sc daily /st 22:00 /f
REM
REM Remove:
REM   schtasks /delete /tn "llm-quant paper batch" /f
REM
REM Manual test:
REM   E:\llm-quant\scripts\daily_paper_batch.bat
REM ============================================================

setlocal

set "PROJECT_ROOT=E:\llm-quant"
set "PYTHONPATH=%PROJECT_ROOT%\src"
set "LOG_DIR=%PROJECT_ROOT%\logs\paper_cron"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ISO date for log file (YYYY-MM-DD), independent of locale
for /f "tokens=*" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%I"
set "LOG_FILE=%LOG_DIR%\%TODAY%.log"

cd /d "%PROJECT_ROOT%"

echo ==== Run started %DATE% %TIME% ==== >> "%LOG_FILE%"
python "%PROJECT_ROOT%\scripts\daily_paper_cron.py" >> "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
echo ==== Run finished %DATE% %TIME% rc=%RC% ==== >> "%LOG_FILE%"

endlocal & exit /b %RC%
