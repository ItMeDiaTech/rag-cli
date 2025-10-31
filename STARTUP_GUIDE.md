# RAG-CLI Auto-Startup Guide

This guide explains how to configure RAG-CLI monitoring services to start automatically when Claude Code launches.

## Overview

RAG-CLI provides three ways to auto-start monitoring services:

1. **Automatic Hook-Based (Recommended)** - Services auto-start when you first use the plugin
2. **System Task Scheduler** - Services auto-start on Windows boot
3. **Claude Code MCP** - Services auto-start when Claude Code connects

## Method 1: Automatic Hook-Based (Default)

This is the easiest method and requires no additional configuration.

**How it works:**
- When you first use any RAG command or query, the plugin automatically checks if services are running
- If services aren't running, they're started automatically
- This happens transparently in the background

**No setup required** - this is enabled by default!

**Services Started:**
- TCP Monitoring Server (port 9999)
- Web Dashboard (port 5000)

---

## Method 2: Windows Task Scheduler

Install RAG-CLI services to auto-start on Windows boot.

### Prerequisites
- Administrator access (required for task registration)
- PowerShell 5.0+
- Python installed and in PATH

### Setup Instructions

1. Open PowerShell as Administrator:
   - Press `Win + X`, select "Windows PowerShell (Admin)"
   - Or search for "PowerShell" and "Run as administrator"

2. Navigate to the RAG-CLI scripts directory:
   ```powershell
   cd "C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI\scripts"
   ```

3. Run the startup script with task installation:
   ```powershell
   .\startup.ps1 -InstallAsTask
   ```

4. A Windows Task Scheduler task named "RAG-CLI-Startup" will be created

### Verification

Check that the task was created:
```powershell
Get-ScheduledTask -TaskName "RAG-CLI-Startup"
```

### What Happens at Boot
- Windows automatically runs the startup task
- RAG-CLI monitoring services start in the background
- Web dashboard available at `http://localhost:5000`

### Manual Startup (Any Time)
If you just want to start services now without scheduling:
```powershell
.\startup.ps1
```

---

## Method 3: Claude Code MCP Server

Configure Claude Code to auto-start the RAG MCP server.

### Prerequisites
- Claude Code installed
- MCP server configuration file created

### Setup Instructions

1. **Copy MCP Configuration**:
   The file `mcp-server.json` in the RAG-CLI root directory contains the MCP server configuration.

2. **Configure Claude Code**:

   Add the following to your Claude Code `settings.json` (usually at `~/.claude/settings.json`):

   ```json
   {
     "mcp": {
       "servers": {
         "rag-cli": {
           "command": "python",
           "args": ["-m", "src.monitoring.service_manager"],
           "env": {
             "PYTHONUNBUFFERED": "1",
             "RAG_AUTO_START": "true"
           },
           "cwd": "C:\\Users\\DiaTech\\Pictures\\DiaTech\\Programs\\DocHub\\development\\RAG-CLI",
           "autoStart": true
           "maxRetries": 3,
           "retryDelayMs": 1000
         }
       }
     }
   }
   ```

3. **Restart Claude Code**:
   - Close and reopen Claude Code
   - The MCP server will auto-connect and start monitoring services

---

## Verification

To check if services are running:

### Method 1: Check Ports
```powershell
Test-NetConnection -ComputerName localhost -Port 9999  # TCP server
Test-NetConnection -ComputerName localhost -Port 5000  # Web dashboard
```

### Method 2: Access Web Dashboard
Open your browser and visit: `http://localhost:5000`

### Method 3: PowerShell Monitor
```powershell
cd "C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI"
.\scripts\monitor.ps1 WATCH
```

---

## Troubleshooting

### Services Not Starting at Boot

1. **Check Task Scheduler**:
   ```powershell
   Get-ScheduledTask -TaskName "RAG-CLI-Startup" | Select-Object State, LastRunTime
   ```

2. **View Task History**:
   - Open Task Scheduler (search for "Task Scheduler")
   - Navigate to: Task Scheduler Library -> RAG-CLI-Startup
   - Check "History" tab for errors

3. **Manual Start**:
   ```powershell
   Start-ScheduledTask -TaskName "RAG-CLI-Startup"
   ```

### Port Already in Use

If you get an error about ports 9999 or 5000 being in use:

1. **Find process using the port**:
   ```powershell
   # For port 9999
   Get-Process | Where-Object { $_.Handles | Select-String 9999 }

   # Or check all listening ports
   netstat -ano | findstr :9999
   ```

2. **Kill the process** (if not needed):
   ```powershell
   Stop-Process -Id <PID> -Force
   ```

3. **Or change the port** in `src/core/config.py`

### Services Running But Dashboard Not Accessible

1. Check if services are actually running:
   ```powershell
   curl http://localhost:5000
   ```

2. Check logs:
   ```powershell
   cat "C:\path\to\RAG-CLI\logs\*.log"
   ```

3. Restart services:
   ```powershell
   .\startup.ps1 -NoWait
   ```

### Admin Rights Required

The task installation requires administrator access. If you get an "Access Denied" error:

1. Right-click PowerShell
2. Select "Run as administrator"
3. Navigate to the scripts folder and try again

---

## Uninstall/Disable Auto-Start

### Remove Windows Task

```powershell
Unregister-ScheduledTask -TaskName "RAG-CLI-Startup" -Confirm:$false
```

### Disable Hook-Based Auto-Start

Edit `config/rag_settings.json`:
```json
{
  "auto_start_services": false
}
```

### Remove MCP Configuration

Remove the `rag-cli` server entry from Claude Code's `settings.json`

---

## Service Details

### TCP Monitoring Server (Port 9999)
- **Purpose**: Provides monitoring data via TCP protocol
- **Used by**: PowerShell monitoring dashboard
- **Started by**: All auto-start methods

### Web Dashboard (Port 5000)
- **Purpose**: Beautiful browser-based monitoring interface
- **Access**: `http://localhost:5000`
- **Features**: Real-time metrics, logs, health status
- **Started by**: All auto-start methods

### MCP Server
- **Purpose**: Provides RAG functionality to Claude Code
- **Protocol**: Model Context Protocol (stdio)
- **Auto-starts**: Monitoring services on initialization

---

## Best Practices

1. **Use Hook-Based Auto-Start** (default):
   - Simplest approach
   - No additional setup required
   - Services start when needed

2. **Use Windows Task** for Always-On:
   - If you want services running even when Claude Code is closed
   - For background monitoring and metrics collection
   - Useful for production deployments

3. **Check Status Regularly**:
   - Monitor the web dashboard for system health
   - Watch logs for errors or warnings
   - Use `.\startup.ps1` to restart if needed

---

## Quick Command Reference

```powershell
# Start services manually
.\scripts\startup.ps1

# Install auto-start task (admin required)
.\scripts\startup.ps1 -InstallAsTask

# View web dashboard
Start-Process chrome http://localhost:5000

# Watch system monitoring
.\scripts\monitor.ps1 WATCH

# Check service status
Get-ScheduledTask -TaskName "RAG-CLI-Startup"

# View service logs
Get-Content logs/*.log -Tail 50

# Restart services
.\scripts\startup.ps1 -NoWait
```

---

## Support

For issues with auto-startup:

1. Check logs in `logs/` directory
2. View `config/services_status.json` for service info
3. Run `.\scripts/monitor.ps1 HEALTH` for system status
4. Consult the troubleshooting section above
