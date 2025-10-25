@echo off
REM RAG-CLI Plugin Update Helper Script
REM This script makes it easy to sync plugin files to Claude Code

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Parse arguments
set DRY_RUN=
set VERBOSE=
set FORCE=
set NO_BACKUP=
set NO_SYMLINK=

:parse_args
if "%1"=="" goto run_sync
if "%1"=="--dry-run" (
    set DRY_RUN=--dry-run
    shift
    goto parse_args
)
if "%1"=="--verbose" (
    set VERBOSE=--verbose
    shift
    goto parse_args
)
if "%1"=="-v" (
    set VERBOSE=--verbose
    shift
    goto parse_args
)
if "%1"=="--force" (
    set FORCE=--force
    shift
    goto parse_args
)
if "%1"=="-f" (
    set FORCE=--force
    shift
    goto parse_args
)
if "%1"=="--no-backup" (
    set NO_BACKUP=--no-backup
    shift
    goto parse_args
)
if "%1"=="--no-symlink" (
    set NO_SYMLINK=--no-symlink
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_help
)
if "%1"=="/?" (
    goto show_help
)
echo Unknown argument: %1
goto show_help

:run_sync
cls
echo.
echo ========================================
echo   RAG-CLI Plugin Update
echo ========================================
echo.

python sync_plugin.py %DRY_RUN% %VERBOSE% %FORCE% %NO_BACKUP% %NO_SYMLINK%

if errorlevel 1 (
    echo.
    echo ERROR: Sync failed with exit code %errorlevel%
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Plugin synced successfully
    if "%DRY_RUN%"=="" (
        echo.
        echo Tip: Run with --dry-run to preview changes before syncing
    )
    pause
    exit /b 0
)

:show_help
cls
echo.
echo RAG-CLI Plugin Update Helper
echo.
echo Usage: update_plugin.bat [options]
echo.
echo Options:
echo   --dry-run         Preview changes without making them
echo   --verbose, -v     Verbose output
echo   --force, -f       Force sync (ignore timestamps)
echo   --no-backup       Skip backup creation
echo   --no-symlink      Use copy instead of symlinks
echo   --help, /?        Show this help message
echo.
echo Examples:
echo   update_plugin.bat              - Normal sync
echo   update_plugin.bat --dry-run    - Preview changes
echo   update_plugin.bat --verbose    - Detailed output
echo.
pause
exit /b 0
