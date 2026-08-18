<#
push_data.ps1 - replace the input video on every host in the fleet, over SSH.

    .\push_data.ps1 -Source dai@<host>:/home/dai/ntuanh/ExpandingVideo/output/video.mp4
                                       # DRY RUN: shows every file it would
                                       # copy, replace and delete. Default.
    .\push_data.ps1 -Source ... -Apply             # actually do it
    .\push_data.ps1 -Source ... -Apply -Only machine-2,machine-3
    .\push_data.ps1 -Source .\video.mp4 -Apply     # a local file, already staged

Reads hosts.json, exactly like deploy.ps1 (see hosts.example.json). That file
names YOUR machines, so it is gitignored like config.yaml and setup.json.

Why this is not deploy.ps1
--------------------------
deploy.ps1 moves CODE, by pull, and never copies files - because config.yaml and
setup.json are per-host and a copy would flatten every host's identity. Data is
the opposite case: *.mp4 is gitignored, so git cannot carry it and a copy is the
only way. Keeping the two apart keeps that distinction visible.

Three things it does that a bare scp loop does not
--------------------------------------------------
1. It ASKS EACH HOST where its old video is rather than assuming. `data:` in
   config.yaml is per-host and hand-edited, so the old path is a fact only that
   host holds. Nothing is deleted that did not come from that host's own config.
2. It VERIFIES THE COPY BEFORE DELETING ANYTHING. SHA-256 on both ends; a host
   whose hash does not match keeps its old file and is reported as failed.
3. It DELETES map/label/. Ground truth is generated FROM the video by
   make_map_labels.py, so a new video with the old labels still in place scores
   this video's detections against that video's frames - plausible numbers,
   entirely false. Same class of trap as a write-once cache surviving a run.

DRY RUN IS THE DEFAULT. Nothing is copied or deleted without -Apply.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]  $Source,                       # user@host:/path/to/video.mp4, or a local path
    [string]  $Inventory,                    # default: hosts.json beside this script
    [string[]]$Only,
    [string]  $DestName = 'video.mp4',       # filename to write on each host
    [string]  $DestDir,                      # default: the dir the host's old video is in
    [switch]  $UpdateConfig,                 # rewrite each host's config.yaml data: line
    [switch]  $KeepLabels,                   # do NOT delete map/label (read the warning first)
    [switch]  $Apply,
    [int]     $TimeoutSec = 60
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is EMPTY inside param() under `powershell -File`, which is the
# invocation this project has to use (execution policy AllSigned). Resolve it in
# the body instead -- deploy.ps1 had the same default and crashed on it.
$root = if ($PSScriptRoot) { $PSScriptRoot }
        else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $root
if (-not $Inventory) { $Inventory = Join-Path $root 'hosts.json' }

if (-not (Test-Path $Inventory)) {
    Write-Host ""
    Write-Host "No inventory at $Inventory" -ForegroundColor Red
    Write-Host "Create it from the template:" -ForegroundColor Yellow
    Write-Host "    Copy-Item hosts.example.json hosts.json"
    Write-Host "then fill in the ssh target, logical name and repo path per host."
    Write-Host ""
    exit 1
}
$inv   = Get-Content $Inventory -Raw | ConvertFrom-Json
$hosts = @($inv.hosts)
if ($Only) { $hosts = @($hosts | Where-Object { $Only -contains $_.name }) }
if ($hosts.Count -eq 0) { Write-Host "No hosts selected." -ForegroundColor Yellow; exit 1 }

# ------------------------------------------------------------------ helpers --
# The two python programs are here-strings on purpose: they contain quotes of
# both kinds, and inlining them into a PowerShell string is how this script
# failed to parse the first time.
$ReadDataProg = @'
import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['data'])
'@
# No double quote anywhere in this one - chr(34) supplies it - so the same
# command text survives both a POSIX login shell and cmd.exe on a Windows host.
$SetDataProg = @'
import re,io,sys;p=sys.argv[1];n=sys.argv[2];s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^data:.*$','data: '+chr(34)+n+chr(34),s,count=1);io.open(p,'w',encoding='utf-8').write(s)
'@

function Split-Target {
    # "user@host:2222" -> @(dest, portArgs). Same rule deploy.ps1 uses.
    param([string]$Target)
    if ($Target -match '^(.*):(\d+)$') { return @($Matches[1], @('-p', $Matches[2])) }
    return @($Target, @())
}

function Invoke-Ssh {
    param([string]$Target, [string]$Command)
    $dest, $portArgs = Split-Target $Target
    $sshArgs = @('-o', 'BatchMode=yes', '-o', "ConnectTimeout=$TimeoutSec",
                 '-o', 'StrictHostKeyChecking=accept-new') + $portArgs + @($dest, $Command)
    # Windows PowerShell wraps a native command's stderr in an ErrorRecord, so
    # with ErrorActionPreference=Stop the FIRST unreachable host would terminate
    # the whole fleet loop. One host failing must never stop the fleet -- same
    # rule deploy.ps1 states. Keep stderr (it carries the auth diagnosis) but
    # make it non-terminating.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try     { $out = & ssh @sshArgs 2>&1 }
    catch   { $out = $_.Exception.Message }
    finally { $ErrorActionPreference = $prev }
    return @{ ok = ($LASTEXITCODE -eq 0); out = ($out | Out-String).Trim() }
}

function Invoke-Native {
    # Same guard for scp: a transfer failure is a reported row, not a crash.
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try     { & $Exe @Arguments 2>&1 | Out-Null }
    catch   { $global:LASTEXITCODE = 1 }
    finally { $ErrorActionPreference = $prev }
    return ($LASTEXITCODE -eq 0)
}

function Copy-ToHost {
    param([string]$Local, [string]$Target, [string]$RemotePath)
    $dest, $portArgs = Split-Target $Target
    # scp takes the port as -P, ssh as -p. Getting this wrong is a silent
    # "connection refused" against port 22 on a host that does not listen there.
    $portArgs = @($portArgs | ForEach-Object { if ($_ -eq '-p') { '-P' } else { $_ } })
    $scpArgs = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new') +
               $portArgs + @($Local, ($dest + ':' + $RemotePath))
    return (Invoke-Native 'scp' $scpArgs)
}

function Test-Posix {
    param([string]$Target)
    (Invoke-Ssh $Target 'uname -s').ok        # false on a Windows OpenSSH host
}

function Get-RemoteHash {
    param([string]$Target, [string]$Path, [bool]$Posix)
    if ($Posix) {
        $cmd = "sha256sum '$Path'"
    } else {
        $cmd = 'powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 \"' + $Path + '\").Hash"'
    }
    $r = Invoke-Ssh $Target $cmd
    if (-not $r.ok) { return $null }
    return ($r.out -split '\s+')[0].ToLower()
}

# ----------------------------------------------------------- stage locally --
# One pull from the source, then a fan-out. The alternative - every host pulling
# from the source itself - needs host-to-source credentials on twelve machines
# and makes the source serve the file twelve times.
$remoteSource  = ($Source -match '^[^:]+@[^:]+:')
$staged        = $null
$cleanupStaged = $false
if ($remoteSource) {
    $staged = Join-Path ([IO.Path]::GetTempPath()) ("push_data_" + [guid]::NewGuid().ToString('N') + ".mp4")
    Write-Host "Staging $Source"
    if ($Apply) {
        $srcTarget, $srcPath = $Source -split ':', 2
        $dest, $portArgs = Split-Target $srcTarget
        $portArgs = @($portArgs | ForEach-Object { if ($_ -eq '-p') { '-P' } else { $_ } })
        $scpArgs = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new') +
                   $portArgs + @(($dest + ':' + $srcPath), $staged)
        if (-not (Invoke-Native 'scp' $scpArgs)) {
            Write-Host "  cannot fetch $Source" -ForegroundColor Red
            exit 1
        }
        $cleanupStaged = $true
    } else {
        Write-Host "  [dry run] would scp it to a temp file" -ForegroundColor DarkGray
    }
} else {
    if (-not (Test-Path $Source)) {
        Write-Host "No such file: $Source" -ForegroundColor Red
        exit 1
    }
    $staged = (Resolve-Path $Source).Path
}

$localHash = $null
if (Test-Path $staged) {
    $localHash = (Get-FileHash -Algorithm SHA256 $staged).Hash.ToLower()
    $sizeMb = [math]::Round((Get-Item $staged).Length / 1MB, 1)
    Write-Host ("  {0} MB  sha256 {1}..." -f $sizeMb, $localHash.Substring(0, 16))
}

Write-Host ""
if ($Apply) {
    Write-Host "$($hosts.Count) host(s)"
} else {
    Write-Host "$($hosts.Count) host(s)   [DRY RUN - nothing will change]" -ForegroundColor Yellow
}
Write-Host ""

# ----------------------------------------------------------------- per host --
$rows = @()
foreach ($h in $hosts) {
    Write-Host ("-> {0,-11} {1}" -f $h.name, $h.ssh)
    $row = [ordered]@{ Name = $h.name; Role = $h.role; Old = ''; New = ''; Status = '' }
    $repo = $h.repo.TrimEnd('/', '\')

    $posix = Test-Posix $h.ssh
    if (-not $posix) {
        $probe = Invoke-Ssh $h.ssh 'echo ok'
        if (-not $probe.ok) {
            $row.Status = 'ssh unreachable / auth refused'
            $rows += [pscustomobject]$row
            Write-Host "   $($row.Status)" -ForegroundColor Red
            continue
        }
    }

    # The old path is a fact only this host holds: config.yaml is per-host.
    $cfg = Invoke-Ssh $h.ssh ('python -c "' + $ReadDataProg + '" "' + $repo + '/config.yaml"')
    if (-not $cfg.ok -or -not $cfg.out) {
        $row.Status = 'cannot read data: from config.yaml'
        $rows += [pscustomobject]$row
        Write-Host "   $($row.Status)" -ForegroundColor Red
        continue
    }
    $oldPath = $cfg.out.Trim()
    $row.Old = $oldPath

    if ($DestDir) { $dir = $DestDir.TrimEnd('/', '\') }
    else          { $dir = ($oldPath -replace '[\\/][^\\/]+$', '') }
    $newPath = "$dir/$DestName"
    $row.New = $newPath

    if (-not $Apply) {
        if ($oldPath -eq $newPath) { $row.Status = 'would overwrite in place' }
        else                       { $row.Status = 'would copy, then delete old' }
        $rows += [pscustomobject]$row
        Write-Host "   old: $oldPath"
        Write-Host "   new: $newPath   ($($row.Status))" -ForegroundColor DarkGray
        continue
    }

    # 1 - copy to a temp name, so a failed transfer never leaves a truncated
    #     file where the run expects a video.
    $tmp = "$newPath.incoming"
    if (-not (Copy-ToHost $staged $h.ssh $tmp)) {
        $row.Status = 'scp failed - old file untouched'
        $rows += [pscustomobject]$row
        Write-Host "   $($row.Status)" -ForegroundColor Red
        continue
    }
    # 2 - verify BEFORE anything is deleted
    if ((Get-RemoteHash $h.ssh $tmp $posix) -ne $localHash) {
        if ($posix) { Invoke-Ssh $h.ssh "rm -f '$tmp'" | Out-Null }
        else        { Invoke-Ssh $h.ssh ('cmd /c del /q "' + $tmp + '"') | Out-Null }
        $row.Status = 'hash mismatch - old file untouched'
        $rows += [pscustomobject]$row
        Write-Host "   $($row.Status)" -ForegroundColor Red
        continue
    }
    # 3 - move into place, then remove the old file if it is a different one
    if ($posix) { $mv = "mv -f '$tmp' '$newPath'" }
    else        { $mv = 'cmd /c move /y "' + $tmp + '" "' + $newPath + '"' }
    $r = Invoke-Ssh $h.ssh $mv
    if (-not $r.ok) {
        $row.Status = "cannot move into place: $($r.out)"
        $rows += [pscustomobject]$row
        Write-Host "   $($row.Status)" -ForegroundColor Red
        continue
    }
    if ($oldPath -ne $newPath) {
        if ($posix) { Invoke-Ssh $h.ssh "rm -f '$oldPath'" | Out-Null }
        else        { Invoke-Ssh $h.ssh ('cmd /c del /q "' + $oldPath + '"') | Out-Null }
    }

    # 4 - the trap: ground truth generated from the OLD video still describes
    #     the old frames. Scoring the new video against it is silently wrong.
    if (-not $KeepLabels) {
        if ($posix) {
            Invoke-Ssh $h.ssh ("rm -rf '$repo/map/label' '$repo/map/pred' '$repo/map/collect'") | Out-Null
        } else {
            Invoke-Ssh $h.ssh ('cmd /c rmdir /s /q "' + $repo + '\map\label" "' + $repo + '\map\pred" "' + $repo + '\map\collect"') | Out-Null
        }
    }

    if ($UpdateConfig) {
        # config.yaml is per-host and hand-edited, so touch exactly the one line.
        Invoke-Ssh $h.ssh ('python -c "' + $SetDataProg + '" "' + $repo + '/config.yaml" "' + $newPath + '"') | Out-Null
    }

    $row.Status = 'replaced'
    $rows += [pscustomobject]$row
    Write-Host "   replaced -> $newPath" -ForegroundColor Green
}

# ------------------------------------------------------------------- report --
Write-Host ""
$rows | Format-Table -AutoSize
$ok  = @('replaced', 'would overwrite in place', 'would copy, then delete old')
$bad = @($rows | Where-Object { $ok -notcontains $_.Status })
if ($bad) {
    Write-Host "$($bad.Count) host(s) did NOT get the new data - the fleet is now MIXED." -ForegroundColor Red
    Write-Host "A run across a mixed fleet is not comparable with anything." -ForegroundColor Red
}
if ($Apply -and -not $KeepLabels) {
    Write-Host ""
    Write-Host "map/label was deleted on every host that succeeded." -ForegroundColor Yellow
    Write-Host "Regenerate it on the SERVER before any run with map.enable: true," -ForegroundColor Yellow
    Write-Host "or map.log scores this video's detections against the old video's frames:"
    Write-Host "    python make_map_labels.py"
}
if ($cleanupStaged) { Remove-Item $staged -Force -ErrorAction SilentlyContinue }
