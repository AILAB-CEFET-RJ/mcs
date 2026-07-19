param(
    [string]$OutputTag = "BASELINE_CORRECTED",
    [string]$ConfigName = ""
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$projectRoot = $PSScriptRoot
$logDirectory = Join-Path $projectRoot "runs\stage1_baseline"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$configs = Get-ChildItem (Join-Path $projectRoot "config") -Filter "train_*.yaml" |
    Sort-Object Name

if ($ConfigName) {
    $configs = $configs | Where-Object Name -EQ $ConfigName
}

if (-not $configs) {
    throw "Nenhuma configuração de treino encontrada."
}

Push-Location $projectRoot
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = "utf-8"
try {
    foreach ($config in $configs) {
        $logPath = Join-Path $logDirectory ($config.BaseName + ".log")
        Write-Host "`n=== Etapa 1: $($config.Name) ==="

        & python "src/train_pipeline.py" `
            --config $config.FullName `
            --output-tag $OutputTag 2>&1 |
            Tee-Object -FilePath $logPath

        if ($LASTEXITCODE -ne 0) {
            throw "Falha em $($config.Name). Consulte $logPath"
        }
    }
}
finally {
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    Pop-Location
}

Write-Host "`nEtapa 1 concluída para $($configs.Count) configuração(ões)."
