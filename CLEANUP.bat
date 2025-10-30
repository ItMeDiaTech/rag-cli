@echo off
REM RAG-CLI Cleanup Script - Run after closing Claude Code

echo Cleaning up locked files from old installation...

REM Remove old manual installation directory (may have locked log files)
rmdir /S /Q "C:\Users\DiaTech\.claude\plugins\rag-cli" 2>nul

REM Remove any remaining backup directories
for /d %%d in ("C:\Users\DiaTech\.claude\plugins\rag-cli_backup_*") do (
    rmdir /S /Q "%%d" 2>nul
)

echo.
echo Cleanup complete!
echo You can now reinstall the plugin via:
echo   /plugin marketplace add ItMeDiaTech/rag-cli
echo   /plugin install rag-cli
echo.
pause
