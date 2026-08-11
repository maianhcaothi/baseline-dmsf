<#
run_cluster.ps1 — launch an [N_edge, N_cloud] DMSF client fleet on one machine.

The server must already be waiting in its own terminal:

    d:\SplitInference\venv\Scripts\python.exe server.py

Then, from this directory:

    .\run_cluster.ps1                       # 9 edges + 3 clouds, all CPU
    .\run_cluster.ps1 -Edges 2 -Clouds 1    # smaller smoke test
    .\run_cluster.ps1 -Device cuda          # only if the GPU can hold every process

`server.clients` in config.yaml must match -Edges/-Clouds exactly, or the server
never dispatches START.
#>
param(
    [int]$Edges   = 9,
    [int]$Clouds  = 3,
    [string]$Device = 'cpu',
    [string]$Python = 'd:\SplitInference\venv\Scripts\python.exe',
    [int]$StaggerMs = 1500
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# --name doubles as the id tag for timing_*.log and metrics_raw_*.csv, so it has
# to be unique per process. Under `split-point: auto` it must ALSO be a key in
# the profile-cache JSON — these lists are exactly the keys in
# profiles/all_devices_bs32.json.
$edgeNames  = 2..10 | ForEach-Object { "machine-$_" }
$cloudNames = @('device-1', 'device-2', 'device-3', 'device-4', 'device-7')

if ($Edges  -gt $edgeNames.Count)  { throw "only $($edgeNames.Count) edge names available"  }
if ($Clouds -gt $cloudNames.Count) { throw "only $($cloudNames.Count) cloud names available" }

$logs = Join-Path $PSScriptRoot 'client-logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$procs = @()

function Start-Client($LayerId, $Name, $Role) {
    $p = Start-Process -FilePath $Python -PassThru -NoNewWindow `
        -ArgumentList 'client.py', '--layer_id', $LayerId, '--name', $Name, '--device', $Device `
        -RedirectStandardOutput "$logs\${Role}_${Name}.out" `
        -RedirectStandardError  "$logs\${Role}_${Name}.err"
    Write-Host ("  {0,-6} {1,-12} pid {2}" -f $Role, $Name, $p.Id)
    # Purely a load stagger: START is broadcast only after all clients register,
    # so launch order is irrelevant to correctness. 12 simultaneous torch
    # imports just thrash the disk.
    Start-Sleep -Milliseconds $StaggerMs
    return $p
}

Write-Host "Launching $Edges edge + $Clouds cloud on device=$Device"
foreach ($n in $cloudNames[0..($Clouds - 1)]) { $procs += Start-Client 2 $n 'cloud' }
foreach ($n in $edgeNames[0..($Edges  - 1)]) { $procs += Start-Client 1 $n 'edge'  }

Write-Host "`n$($procs.Count) clients up. Output in $logs"
Write-Host "Tail one with:  Get-Content $logs\edge_machine-2.out -Wait`n"
$procs | Wait-Process
Write-Host "All clients exited."
