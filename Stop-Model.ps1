# 停止 RAG 使用的本地 Huihui 模型服务。
# 用法：.\Stop-Model.ps1
# 现有停止入口会核对进程路径和启动时间，避免停止无关进程。
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$modelLauncher = Join-Path $workspaceRoot 'local-model-uncensored\Model.cmd'

if (-not (Test-Path -LiteralPath $modelLauncher -PathType Leaf)) {
    throw "Local model launcher was not found: $modelLauncher"
}

& $modelLauncher Stop
if ($LASTEXITCODE -ne 0) {
    throw "Stopping the local model failed (exit $LASTEXITCODE)."
}
