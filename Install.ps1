#requires -Version 7.4
[CmdletBinding()]
param([string]$Python, [string]$Proxy = '')
$ErrorActionPreference = 'Stop'
$ragRoot = $PSScriptRoot
$ragVenv = Join-Path $ragRoot '.venv'
$ragPython = Join-Path $ragVenv 'Scripts\python.exe'
$ragEvidence = Join-Path $ragRoot 'evidence'
$ragInstallStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Path $ragEvidence,(Join-Path $ragRoot '.cache\pip') -Force | Out-Null

function Invoke-RagPython {
    param([string[]]$PythonArguments)
    & $ragPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Project Python command failed (exit $LASTEXITCODE)." }
}

if (-not (Test-Path -LiteralPath $ragPython)) {
    if (-not $Python) {
        $Python = (& py -3.13 -c 'import sys; print(sys.executable)')
        if ($LASTEXITCODE -ne 0 -or -not $Python) { throw 'Pass -Python with a Python 3.13 x64 executable.' }
        $Python = $Python.Trim()
    }
    & $Python -c 'import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize("P") == 8, "Python 3.13 x64 is required for this lock"'
    if ($LASTEXITCODE -ne 0) { throw 'Incompatible base Python.' }
    & $Python -m venv --prompt rag-lab $ragVenv
    if ($LASTEXITCODE -ne 0) { throw 'Creating the project virtual environment failed.' }
}
Invoke-RagPython -PythonArguments @('-c', 'import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize("P") == 8')

$ragPipArgs = @('-m','pip','--isolated','--disable-pip-version-check','--cache-dir',(Join-Path $ragRoot '.cache\pip'))
if ($Proxy) { $ragPipArgs += @('--proxy',$Proxy) }
$ragPreviousHttpProxy = [Environment]::GetEnvironmentVariable('HTTP_PROXY','Process')
$ragPreviousHttpsProxy = [Environment]::GetEnvironmentVariable('HTTPS_PROXY','Process')
try {
    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
    }
    Invoke-RagPython -PythonArguments ($ragPipArgs + @('install','--only-binary=:all:','--index-url','https://pypi.org/simple','pip==26.2.2'))
    Invoke-RagPython -PythonArguments ($ragPipArgs + @('install','--only-binary=:all:','--index-url','https://download.pytorch.org/whl/cpu','--report',(Join-Path $ragEvidence "torch-install-$ragInstallStamp.report.json"),'-r',(Join-Path $ragRoot 'requirements.cpu.lock.txt')))
    Invoke-RagPython -PythonArguments ($ragPipArgs + @('install','--only-binary=:all:','--index-url','https://pypi.org/simple','--report',(Join-Path $ragEvidence "dependencies-install-$ragInstallStamp.report.json"),'-r',(Join-Path $ragRoot 'requirements.lock.txt')))
    Invoke-RagPython -PythonArguments @('-m','pip','--isolated','check')
    Invoke-RagPython -PythonArguments @((Join-Path $ragRoot 'download_model.py'))
    Invoke-RagPython -PythonArguments @((Join-Path $ragRoot 'verify_environment.py'))
} finally {
    [Environment]::SetEnvironmentVariable('HTTP_PROXY',$ragPreviousHttpProxy,'Process')
    [Environment]::SetEnvironmentVariable('HTTPS_PROXY',$ragPreviousHttpsProxy,'Process')
}
Write-Host "RAG dependency environment is ready: $ragPython"
