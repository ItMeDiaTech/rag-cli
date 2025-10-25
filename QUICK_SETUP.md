# RAG-CLI Auto-Start Quick Setup

## TL;DR - Just Use It!

**Good news:** Auto-start is already enabled by default!

When you first use any RAG command in Claude Code, the monitoring services will start automatically. No setup required.

---

## Want Auto-Start on Windows Boot?

Run this one command as Administrator:

```powershell
cd "C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
.\scripts\startup.ps1 -InstallAsTask
```

That's it! Services will now auto-start every time Windows boots.

---

## Check Status

**View Dashboard in Browser:**
```
http://localhost:5000
```

**Check Services Running:**
```powershell
Test-NetConnection -ComputerName localhost -Port 9999  # TCP Server
Test-NetConnection -ComputerName localhost -Port 5000  # Dashboard
```

**Watch Real-Time Metrics:**
```powershell
.\scripts\monitor.ps1 WATCH
```

---

## How It Works

### Default (No Setup)
1. You use a RAG command in Claude Code
2. The plugin automatically checks if services are running
3. If not, they start automatically
4. You use the service immediately

### With Windows Task (Optional)
1. Windows boots
2. Task Scheduler runs the startup script
3. Services start in background
4. You access dashboard anytime at `http://localhost:5000`

### With Claude Code MCP (Advanced)
Services start when Claude Code itself starts (needs manual config)

---

## Services Started

| Service | Port | Purpose |
|---------|------|---------|
| **TCP Monitoring Server** | 9999 | Data for PowerShell monitor |
| **Web Dashboard** | 5000 | Beautiful browser interface |

---

## Access Points

| Tool | URL | Usage |
|------|-----|-------|
| **Web Dashboard** | http://localhost:5000 | Real-time metrics & logs |
| **PowerShell Monitor** | `.\scripts\monitor.ps1 WATCH` | Terminal monitoring |
| **RAG Plugin** | Direct use in Claude Code | Automatic via hooks |

---

## Troubleshooting

### Services not starting?
```powershell
# Start them manually
.\scripts\startup.ps1

# With more details
.\scripts\startup.ps1 | Select-Object *
```

### Port in use error?
```powershell
# Find what's using port 9999
netstat -ano | findstr :9999
# Kill the process if needed
Stop-Process -Id <PID> -Force
```

### Task not running?
```powershell
# Check task exists
Get-ScheduledTask -TaskName "RAG-CLI-Startup"

# Run it manually
Start-ScheduledTask -TaskName "RAG-CLI-Startup"

# View history
Get-ScheduledTask -TaskName "RAG-CLI-Startup" | Get-ScheduledTaskInfo
```

---

## Full Documentation

See `STARTUP_GUIDE.md` for complete setup options and troubleshooting.

---

## Summary

**Default:** Just start using the RAG plugin - services auto-start when needed

**Better:** Install Windows Task (one command, requires admin):
```powershell
.\scripts\startup.ps1 -InstallAsTask
```

**Monitor:** View live dashboard at `http://localhost:5000`

That's all you need to know!
