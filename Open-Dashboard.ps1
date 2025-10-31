# PowerShell launcher for RAG-CLI Dashboard
Write-Host "RAG-CLI Dashboard Launcher" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: http://127.0.0.1:5000" -ForegroundColor Green
Write-Host ""

# Find Chrome
$chromePaths = @(
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)

$chromePath = $null
foreach ($path in $chromePaths) {
    if (Test-Path $path) {
        $chromePath = $path
        break
    }
}

if ($chromePath) {
    Write-Host "Opening in Chrome with localhost security bypass..." -ForegroundColor Yellow
    $tempDir = Join-Path $env:TEMP "chrome-rag-cli"
    Start-Process -FilePath $chromePath -ArgumentList "--disable-web-security","--user-data-dir=$tempDir","http://127.0.0.1:5000"
} else {
    Write-Host "Chrome not found. Opening in default browser..." -ForegroundColor Yellow
    Start-Process "http://127.0.0.1:5000"
}

Write-Host ""
Write-Host "If you see connection errors:" -ForegroundColor Cyan
Write-Host "  - Try hard refresh: Ctrl+Shift+R" -ForegroundColor White
Write-Host "  - Try a different browser (Edge, Firefox)" -ForegroundColor White
Write-Host "  - Check Windows Firewall settings" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
