<#
.SYNOPSIS
    Startup script for RAG-CLI monitoring services

.DESCRIPTION
    This script auto-starts RAG-CLI monitoring services when Claude Code launches.
    Run this script once to register it for auto-startup, or call it manually.

.PARAMETER InstallAsTask
    Install this script to run at system startup via Windows Task Scheduler

.PARAMETER NoWait
    Don't wait for services to be ready before returning

.EXAMPLE
    .\startup.ps1                          # Start services now
    .\startup.ps1 -InstallAsTask           # Install as Windows Task
    .\startup.ps1 -NoWait                  # Start without waiting
#>

param(
    [switch]$InstallAsTask,
    [switch]$NoWait
)

# Get project root
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source

if (-not $pythonExe) {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "RAG-CLI Service Startup" -ForegroundColor Cyan
Write-Host "=" * 50

# Function to check if port is open
function Test-PortOpen {
    param([int]$Port)
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("127.0.0.1", $Port)
        $tcpClient.Close()
        return $true
    } catch {
        return $false
    }
}

# Function to start services
function Start-Services {
    Write-Host "`nStarting RAG-CLI services..." -ForegroundColor Green

    # Start TCP server if not running
    if (-not (Test-PortOpen -Port 9999)) {
        Write-Host "Starting TCP Monitoring Server on port 9999..." -ForegroundColor Yellow
        $scriptBlock = {
            Set-Location $using:projectRoot
            & $using:pythonExe -m src.monitoring.tcp_server
        }
        $null = Start-Job -ScriptBlock $scriptBlock -Name "RAG-TCP-Server" -ErrorAction SilentlyContinue

        if (-not $NoWait) {
            # Wait for server to start
            $timeout = 30
            for ($i = 0; $i -lt $timeout; $i++) {
                if (Test-PortOpen -Port 9999) {
                    Write-Host "TCP server started successfully" -ForegroundColor Green
                    break
                }
                Start-Sleep -Milliseconds 100
            }
        }
    } else {
        Write-Host "TCP server already running on port 9999" -ForegroundColor Cyan
    }

    # Start web dashboard if not running
    if (-not (Test-PortOpen -Port 5000)) {
        Write-Host "Starting Web Dashboard on port 5000..." -ForegroundColor Yellow
        $scriptBlock = {
            Set-Location $using:projectRoot
            & $using:pythonExe -m src.monitoring.web_dashboard 5000
        }
        $null = Start-Job -ScriptBlock $scriptBlock -Name "RAG-Dashboard" -ErrorAction SilentlyContinue

        if (-not $NoWait) {
            # Wait for dashboard to start
            $timeout = 30
            for ($i = 0; $i -lt $timeout; $i++) {
                if (Test-PortOpen -Port 5000) {
                    Write-Host "Web dashboard started successfully" -ForegroundColor Green
                    Write-Host "Dashboard available at: http://localhost:5000" -ForegroundColor Cyan
                    break
                }
                Start-Sleep -Milliseconds 100
            }
        }
    } else {
        Write-Host "Web dashboard already running on port 5000" -ForegroundColor Cyan
    }

    Write-Host "`nAll services started successfully!" -ForegroundColor Green
}

# Function to install as Windows Task
function Install-AsWindowsTask {
    Write-Host "`nInstalling RAG-CLI startup task..." -ForegroundColor Green

    $taskName = "RAG-CLI-Startup"
    $taskDescription = "Auto-start RAG-CLI monitoring services"
    $scriptPath = $PSScriptRoot + "\startup.ps1"

    # Remove existing task if present
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Host "Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }

    # Create task trigger (at system startup)
    $trigger = New-ScheduledTaskTrigger -AtStartup

    # Create task action
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -NoExit -Command `"& '$scriptPath' -NoWait`""

    # Register the task
    try {
        Register-ScheduledTask `
            -TaskName $taskName `
            -Trigger $trigger `
            -Action $action `
            -Description $taskDescription `
            -RunLevel Highest `
            -Force | Out-Null

        Write-Host "Task registered successfully!" -ForegroundColor Green
        Write-Host "Task name: $taskName" -ForegroundColor Cyan
        Write-Host "Trigger: At system startup" -ForegroundColor Cyan
        Write-Host "`nServices will now auto-start when Windows boots" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to register task" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

# Main execution
if ($InstallAsTask) {
    # Check for admin privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "ERROR: Admin privileges required for task registration" -ForegroundColor Red
        Write-Host "Please run this script as Administrator" -ForegroundColor Yellow
        exit 1
    }

    Install-AsWindowsTask
    Start-Services
} else {
    Start-Services
}

Write-Host "`nStartup complete!" -ForegroundColor Green
