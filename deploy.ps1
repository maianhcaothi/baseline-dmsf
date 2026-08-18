<#
deploy.ps1 — pull the current commit onto every host in the fleet, over SSH.

    .\deploy.ps1 -DryRun          # show what each host would do, change nothing
    .\deploy.ps1                  # fetch + pull --ff-only everywhere
    .\deploy.ps1 -Only machine-4,device-2
    .\deploy.ps1 -Ref v1.2        # pin a tag or SHA instead of origin/main

Reads hosts.json (see hosts.example.json). That file names YOUR machines, so it
is gitignored exactly like config.yaml and setup.json.

Why pull and not copy
---------------------
config.yaml and setup.json are per-host and gitignored. git cannot touch an
ignored file, so a pull updates the code and leaves each host's identity alone.
A file copy would overwrite both, every host would register under the same
name, and _id_tag would collide — timing_*.log and metrics_raw_*.csv are keyed
by it, so two hosts would silently overwrite each other's measurements.

--ff-only is deliberate: a host that has diverged FAILS instead of being merged
into some third state nobody has tested. Divergence is a fact worth seeing.

One host failing never stops the fleet. Every host is attempted, and the
summary at the end is the report.
#>
[CmdletBinding()]
param(
    [string]  $Inventory,
    [string[]]$Only,
    [string]  $Ref = 'origin/main',
    [switch]  $DryRun,
    [int]     $TimeoutSec = 30
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is EMPTY inside param() when the script is launched with
# `powershell -File` -- which is the form the README documents, because the
# execution policy here is AllSigned and the in-session `.\deploy.ps1` form is
# not available. The default therefore crashed the script on Join-Path before it
# read a single host. Resolve the root in the BODY, where the automatic variable
# is populated, and fall back to MyInvocation for the -File path.
$root = if ($PSScriptRoot) { $PSScriptRoot }
        else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $root
if (-not $Inventory) { $Inventory = Join-Path $root 'hosts.json' }

# ---------------------------------------------------------------- inventory --
if (-not (Test-Path $Inventory)) {
    Write-Host ""
    Write-Host "No inventory at $Inventory" -ForegroundColor Red
    Write-Host "Create it from the template:" -ForegroundColor Yellow
    Write-Host "    Copy-Item hosts.example.json hosts.json"
    Write-Host "then fill in the ssh target, logical name and repo path per host."
    Write-Host ""
    exit 1
}

$inv = Get-Content $Inventory -Raw | ConvertFrom-Json
$hosts = @($inv.hosts)
if ($Only) { $hosts = @($hosts | Where-Object { $Only -contains $_.name }) }
if ($hosts.Count -eq 0) { Write-Host "No hosts selected." -ForegroundColor Yellow; exit 1 }

# A duplicated logical name is the failure this whole design exists to prevent:
# two hosts sharing a name share an _id_tag, and their per-device timing logs
# and metrics CSVs overwrite each other. Catch it before touching any machine.
$dupes = $hosts | Group-Object name | Where-Object { $_.Count -gt 1 }
if ($dupes) {
    Write-Host "Duplicate host name(s) in the inventory: $($dupes.Name -join ', ')" -ForegroundColor Red
    Write-Host "Each host needs a unique 'name' — it is the per-process id tag." -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------- helpers --
function Invoke-Ssh {
    <#  One remote command, one ssh invocation.

        Never chain with '&&' or 'cd x; y' — the remote default shell may be
        cmd.exe or PowerShell 5.1, and 5.1 has no '&&' at all. Every step below
        is a single `git -C <repo>` call precisely so no remote shell syntax is
        involved.  #>
    param([string]$Target, [string]$Command)

    $sshArgs = @(
        '-o', 'BatchMode=yes'                       # never sit at a password prompt
        '-o', "ConnectTimeout=$TimeoutSec"
        '-o', 'StrictHostKeyChecking=accept-new'
    )
    if ($Target -match '^(.+?):(\d+)$') {
        $sshArgs += @('-p', $Matches[2]); $dest = $Matches[1]
    } else {
        $dest = $Target
    }
    $sshArgs += @($dest, $Command)

    $out = & ssh @sshArgs
    return [pscustomobject]@{
        Output   = ($out | Out-String).Trim()
        ExitCode = $LASTEXITCODE
    }
}

function Get-ShortSha { param([string]$s) if ($s.Length -ge 7) { return $s.Substring(0, 7) } return $s }

# ---------------------------------------------------------------------- run --
$localSha = (& git -C $PSScriptRoot rev-parse HEAD).Trim()
Write-Host ""
Write-Host "Fleet deploy — $($hosts.Count) host(s)" -ForegroundColor Cyan
Write-Host "  ref        $Ref"
Write-Host "  local HEAD $(Get-ShortSha $localSha)"
if ($DryRun) { Write-Host "  DRY RUN — no host will be modified" -ForegroundColor Yellow }
Write-Host ""

$results = @()

foreach ($h in $hosts) {
    Write-Host ("-> {0,-11} {1}" -f $h.name, $h.ssh)
    $repo = $h.repo
    $row = [ordered]@{
        Name = $h.name; Role = $h.role; Ssh = $h.ssh
        Before = ''; After = ''; Status = ''; Note = ''
    }

    # -- 1. reachable, and is that path actually a clone? ----------------------
    $head = Invoke-Ssh $h.ssh "git -C `"$repo`" rev-parse HEAD"
    if ($head.ExitCode -ne 0) {
        $row.Status = 'FAILED'
        if ($head.ExitCode -eq 255) {
            $row.Note = 'ssh unreachable / auth refused'
        } else {
            $row.Note = "not a git repo at $repo"
        }
        Write-Host ("   {0}: {1}" -f $row.Status, $row.Note) -ForegroundColor Red
        $results += [pscustomobject]$row
        continue
    }
    $row.Before = Get-ShortSha $head.Output

    # -- 2. does this host's setup.json agree with the inventory? -------------
    # Wrong name = wrong profile lookup under 'split-point: auto', and a
    # colliding _id_tag if two hosts share one. Report, never rewrite: the file
    # is that machine's own identity and this script does not own it.
    # Best-effort: any failure here (no python on PATH, no setup.json yet) just
    # skips the check rather than failing the host.
    $sj = Invoke-Ssh $h.ssh "python -c `"import json,sys;print(json.load(open(sys.argv[1]))['name'])`" `"$repo/setup.json`""
    if ($sj.ExitCode -eq 0 -and $sj.Output -and $sj.Output -ne $h.name) {
        $row.Note = "setup.json says '$($sj.Output)', inventory says '$($h.name)'"
        Write-Host ("   WARN: {0}" -f $row.Note) -ForegroundColor Yellow
    }

    # -- 3. local source drift? --ff-only will refuse a diverged host anyway,
    #       but naming the drift beats reading a bare 'pull failed'.
    $dirty = Invoke-Ssh $h.ssh "git -C `"$repo`" status --porcelain"
    if ($dirty.ExitCode -eq 0 -and $dirty.Output) {
        $n = ($dirty.Output -split "`n").Count
        $msg = "$n uncommitted change(s)"
        if ($row.Note) { $row.Note = "$($row.Note); $msg" } else { $row.Note = $msg }
        Write-Host ("   WARN: {0}" -f $msg) -ForegroundColor Yellow
    }

    if ($DryRun) {
        $row.Status = 'dry-run'
        $row.After = $row.Before
        Write-Host ("   at {0}, would fast-forward to {1}" -f $row.Before, $Ref)
        $results += [pscustomobject]$row
        continue
    }

    # -- 4. fetch, then fast-forward ------------------------------------------
    $fetch = Invoke-Ssh $h.ssh "git -C `"$repo`" fetch --prune origin"
    if ($fetch.ExitCode -ne 0) {
        $row.Status = 'FAILED'; $row.Note = 'fetch failed (no network, or no credentials for origin)'
        Write-Host ("   {0}: {1}" -f $row.Status, $row.Note) -ForegroundColor Red
        $results += [pscustomobject]$row
        continue
    }

    $merge = Invoke-Ssh $h.ssh "git -C `"$repo`" merge --ff-only $Ref"
    $after = Invoke-Ssh $h.ssh "git -C `"$repo`" rev-parse HEAD"
    $row.After = Get-ShortSha $after.Output

    if ($merge.ExitCode -ne 0) {
        $row.Status = 'FAILED'
        $extra = 'not a fast-forward — host has diverged, resolve by hand'
        if ($row.Note) { $row.Note = "$($row.Note); $extra" } else { $row.Note = $extra }
        Write-Host ("   {0}: {1}" -f $row.Status, $extra) -ForegroundColor Red
        $results += [pscustomobject]$row
        continue
    }

    if ($row.Before -eq $row.After) {
        $row.Status = 'current'
        Write-Host ("   already at {0}" -f $row.After) -ForegroundColor DarkGray
    } else {
        $row.Status = 'updated'
        Write-Host ("   {0} -> {1}" -f $row.Before, $row.After) -ForegroundColor Green
    }

    # -- 5. per-host identity must have survived the pull ---------------------
    # git cannot modify an ignored file, so this should never fire. It is here
    # because the day it does fire is the day someone committed config.yaml.
    $ident = Invoke-Ssh $h.ssh "git -C `"$repo`" ls-files --error-unmatch config.yaml setup.json"
    if ($ident.ExitCode -eq 0) {
        $extra = 'config.yaml/setup.json are TRACKED — per-host identity is being overwritten'
        if ($row.Note) { $row.Note = "$($row.Note); $extra" } else { $row.Note = $extra }
        Write-Host ("   WARN: {0}" -f $extra) -ForegroundColor Yellow
    }

    $results += [pscustomobject]$row
}

# ------------------------------------------------------------------ summary --
Write-Host ""
Write-Host ("=" * 78)
$results | Format-Table Name, Role, Before, After, Status, Note -AutoSize | Out-String | Write-Host
Write-Host ("=" * 78)

$updated = @($results | Where-Object { $_.Status -eq 'updated' }).Count
$current = @($results | Where-Object { $_.Status -eq 'current' }).Count
$failed  = @($results | Where-Object { $_.Status -eq 'FAILED'  }).Count
Write-Host ("  {0} updated   {1} already current   {2} failed" -f $updated, $current, $failed)

# A fleet running mixed code produces results that cannot be compared across
# hosts, so say it plainly rather than leaving it to be read off the table.
$shas = @($results | Where-Object { $_.Status -ne 'FAILED' } | Select-Object -ExpandProperty After -Unique)
if ($shas.Count -gt 1) {
    Write-Host ("  WARNING: fleet is on {0} different commits: {1}" -f $shas.Count, ($shas -join ', ')) -ForegroundColor Yellow
    Write-Host "  Runs from a mixed fleet are not comparable." -ForegroundColor Yellow
} elseif ($shas.Count -eq 1 -and -not $DryRun) {
    Write-Host ("  All reachable hosts on {0}" -f $shas[0]) -ForegroundColor Green
}
Write-Host ""

if ($failed -gt 0) { exit 1 }
exit 0
