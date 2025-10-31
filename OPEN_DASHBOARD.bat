@echo off
REM Quick launcher for RAG-CLI Dashboard
echo Opening RAG-CLI Dashboard...
echo Dashboard URL: http://127.0.0.1:5000
echo.

REM Try Chrome with disabled security (allows localhost)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome with security disabled for localhost...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --disable-web-security --user-data-dir=%TEMP%\chrome-rag-cli http://127.0.0.1:5000
    goto :end
)

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome with security disabled for localhost...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --user-data-dir=%TEMP%\chrome-rag-cli http://127.0.0.1:5000
    goto :end
)

REM Fallback to default browser
echo Chrome not found, using default browser...
start http://127.0.0.1:5000

:end
echo.
echo Dashboard opened. If you see connection errors, try:
echo   - Hard refresh: Ctrl+Shift+R
echo   - Different browser: Edge or Firefox
echo   - Check firewall settings
pause
