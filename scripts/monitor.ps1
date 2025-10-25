#!/usr/bin/env pwsh
<#
.SYNOPSIS
    PowerShell monitoring interface for RAG-CLI TCP server.

.DESCRIPTION
    This script connects to the RAG-CLI monitoring server and provides
    real-time status, logs, and metrics information.

.PARAMETER Command
    The monitoring command to execute (STATUS, LOGS, METRICS, HEALTH, WATCH)

.PARAMETER Host
    The hostname or IP address of the monitoring server (default: localhost)

.PARAMETER Port
    The port number of the monitoring server (default: 9999)

.PARAMETER Follow
    For LOGS command, continuously follow new log entries

.EXAMPLE
    .\monitor.ps1 STATUS
    .\monitor.ps1 LOGS -Follow
    .\monitor.ps1 WATCH

.NOTES
    Requires PowerShell 5.0 or higher
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("STATUS", "LOGS", "METRICS", "HEALTH", "WATCH")]
    [string]$Command = "STATUS",

    [string]$HostName = "localhost",

    [int]$Port = 9999,

    [switch]$Follow
)

# Set error action preference
$ErrorActionPreference = "Stop"

# ANSI color codes for pretty output
$Colors = @{
    Reset     = "`e[0m"
    Bold      = "`e[1m"
    Red       = "`e[31m"
    Green     = "`e[32m"
    Yellow    = "`e[33m"
    Blue      = "`e[34m"
    Magenta   = "`e[35m"
    Cyan      = "`e[36m"
    White     = "`e[37m"
    BrightRed = "`e[91m"
    BrightGreen = "`e[92m"
    BrightYellow = "`e[93m"
    BrightBlue = "`e[94m"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host "$($Colors[$Color])$Message$($Colors.Reset)"
}

function Connect-RAGMonitor {
    param(
        [string]$Server,
        [int]$ServerPort,
        [string]$Request
    )

    try {
        # Create TCP client
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect($Server, $ServerPort)

        # Get network stream
        $stream = $tcpClient.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $reader = New-Object System.IO.StreamReader($stream)

        # Send request
        $writer.WriteLine($Request)
        $writer.Flush()

        # Read response
        $response = ""
        while ($stream.DataAvailable -or $response -eq "") {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            $response += $line + "`n"
        }

        # Close connections
        $writer.Close()
        $reader.Close()
        $stream.Close()
        $tcpClient.Close()

        return $response
    }
    catch {
        Write-ColorOutput "Failed to connect to monitoring server at ${Server}:${ServerPort}" "BrightRed"
        Write-ColorOutput "Error: $_" "Red"
        return $null
    }
}

function Show-Status {
    Write-ColorOutput "`n=== RAG-CLI System Status ===" "BrightCyan"
    Write-ColorOutput "Connecting to ${HostName}:${Port}..." "Yellow"

    $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "STATUS"

    if ($response) {
        try {
            $status = $response | ConvertFrom-Json

            Write-ColorOutput "`nSystem Information:" "BrightGreen"
            Write-Host "  Version:        $($status.version)"
            Write-Host "  Uptime:         $($status.uptime)"
            Write-Host "  Status:         $($status.status)"

            Write-ColorOutput "`nComponent Status:" "BrightGreen"
            Write-Host "  Vector Store:   $($status.components.vector_store)"
            Write-Host "  Embeddings:     $($status.components.embeddings)"
            Write-Host "  Retriever:      $($status.components.retriever)"
            Write-Host "  Claude API:     $($status.components.claude)"

            Write-ColorOutput "`nStatistics:" "BrightGreen"
            Write-Host "  Documents:      $($status.statistics.total_documents)"
            Write-Host "  Vectors:        $($status.statistics.total_vectors)"
            Write-Host "  Queries:        $($status.statistics.total_queries)"
            Write-Host "  Cache Hit Rate: $($status.statistics.cache_hit_rate)%"
        }
        catch {
            Write-Host $response
        }
    }
}

function Show-Logs {
    param([bool]$FollowLogs)

    Write-ColorOutput "`n=== RAG-CLI System Logs ===" "BrightCyan"

    if ($FollowLogs) {
        Write-ColorOutput "Following logs (Ctrl+C to stop)..." "Yellow"

        while ($true) {
            $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "LOGS"

            if ($response) {
                $logs = $response -split "`n"
                foreach ($log in $logs) {
                    if ($log -match "ERROR") {
                        Write-ColorOutput $log "BrightRed"
                    }
                    elseif ($log -match "WARNING") {
                        Write-ColorOutput $log "BrightYellow"
                    }
                    elseif ($log -match "INFO") {
                        Write-ColorOutput $log "Green"
                    }
                    elseif ($log -match "DEBUG") {
                        Write-ColorOutput $log "Cyan"
                    }
                    else {
                        Write-Host $log
                    }
                }
            }

            Start-Sleep -Seconds 2
        }
    }
    else {
        $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "LOGS"

        if ($response) {
            $logs = $response -split "`n" | Select-Object -Last 20
            foreach ($log in $logs) {
                if ($log -match "ERROR") {
                    Write-ColorOutput $log "BrightRed"
                }
                elseif ($log -match "WARNING") {
                    Write-ColorOutput $log "BrightYellow"
                }
                elseif ($log -match "INFO") {
                    Write-ColorOutput $log "Green"
                }
                else {
                    Write-Host $log
                }
            }
        }
    }
}

function Show-Metrics {
    Write-ColorOutput "`n=== RAG-CLI Performance Metrics ===" "BrightCyan"

    $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "METRICS"

    if ($response) {
        try {
            $metrics = $response | ConvertFrom-Json

            Write-ColorOutput "`nLatency Metrics (ms):" "BrightGreen"
            Write-Host ("  Vector Search:  {0,8:N2}" -f $metrics.latency.vector_search)
            Write-Host ("  Keyword Search: {0,8:N2}" -f $metrics.latency.keyword_search)
            Write-Host ("  Reranking:      {0,8:N2}" -f $metrics.latency.reranking)
            Write-Host ("  Claude API:     {0,8:N2}" -f $metrics.latency.claude_api)
            Write-Host ("  End-to-End:     {0,8:N2}" -f $metrics.latency.end_to_end)

            Write-ColorOutput "`nThroughput:" "BrightGreen"
            Write-Host "  Queries/min:    $($metrics.throughput.queries_per_minute)"
            Write-Host "  Docs/min:       $($metrics.throughput.docs_per_minute)"

            Write-ColorOutput "`nResource Usage:" "BrightGreen"
            Write-Host ("  Memory (MB):    {0,8:N2}" -f $metrics.resources.memory_mb)
            Write-Host ("  CPU (%):        {0,8:N2}" -f $metrics.resources.cpu_percent)
        }
        catch {
            Write-Host $response
        }
    }
}

function Show-Health {
    Write-ColorOutput "`n=== RAG-CLI Health Check ===" "BrightCyan"

    $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "HEALTH"

    if ($response) {
        try {
            $health = $response | ConvertFrom-Json

            if ($health.status -eq "healthy") {
                Write-ColorOutput "System Status: HEALTHY" "BrightGreen"
            }
            else {
                Write-ColorOutput "System Status: UNHEALTHY" "BrightRed"
            }

            if ($health.issues) {
                Write-ColorOutput "`nIssues Detected:" "BrightYellow"
                foreach ($issue in $health.issues) {
                    Write-Host "  - $issue"
                }
            }
        }
        catch {
            if ($response -match "OK") {
                Write-ColorOutput "System Status: HEALTHY" "BrightGreen"
            }
            else {
                Write-Host $response
            }
        }
    }
}

function Watch-System {
    Write-ColorOutput "`n=== RAG-CLI System Monitor ===" "BrightCyan"
    Write-ColorOutput "Watching system (Ctrl+C to stop)..." "Yellow"

    while ($true) {
        Clear-Host

        # Show current time
        Write-ColorOutput "RAG-CLI Monitor - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "BrightBlue"
        Write-ColorOutput ("=" * 60) "Blue"

        # Get and display metrics
        $response = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "METRICS"

        if ($response) {
            try {
                $metrics = $response | ConvertFrom-Json

                # Performance bar chart
                Write-ColorOutput "`nPerformance:" "BrightGreen"

                $maxLatency = 1000  # Max latency for scale (ms)
                $barWidth = 30

                foreach ($metric in @("vector_search", "keyword_search", "reranking", "claude_api")) {
                    $value = $metrics.latency.$metric
                    $percentage = [Math]::Min(($value / $maxLatency) * 100, 100)
                    $filled = [Math]::Floor($percentage * $barWidth / 100)
                    $empty = $barWidth - $filled

                    $bar = "[" + ("█" * $filled) + ("░" * $empty) + "]"

                    $label = $metric.Replace("_", " ").ToUpper()
                    Write-Host ("  {0,-15} {1} {2,7:N2} ms" -f $label, $bar, $value)
                }

                Write-ColorOutput "`nThroughput:" "BrightGreen"
                Write-Host "  Queries/min: $($metrics.throughput.queries_per_minute)"
                Write-Host "  Cache Hits:  $($metrics.cache_hit_rate)%"

                Write-ColorOutput "`nResources:" "BrightGreen"
                Write-Host ("  Memory: {0:N0} MB | CPU: {1:N1}%" -f $metrics.resources.memory_mb, $metrics.resources.cpu_percent)
            }
            catch {
                Write-Host "Metrics unavailable"
            }
        }

        # Get recent logs
        Write-ColorOutput "`nRecent Activity:" "BrightGreen"
        $logsResponse = Connect-RAGMonitor -Server $HostName -ServerPort $Port -Request "LOGS"

        if ($logsResponse) {
            $recentLogs = $logsResponse -split "`n" | Select-Object -Last 5
            foreach ($log in $recentLogs) {
                if ($log.Length -gt 80) {
                    $log = $log.Substring(0, 77) + "..."
                }
                Write-Host "  $log"
            }
        }

        Start-Sleep -Seconds 5
    }
}

# Main execution
Write-ColorOutput "RAG-CLI Monitoring Client" "BrightMagenta"
Write-ColorOutput ("=" * 40) "Magenta"

switch ($Command) {
    "STATUS" {
        Show-Status
    }
    "LOGS" {
        Show-Logs -FollowLogs:$Follow
    }
    "METRICS" {
        Show-Metrics
    }
    "HEALTH" {
        Show-Health
    }
    "WATCH" {
        Watch-System
    }
}

Write-Host ""