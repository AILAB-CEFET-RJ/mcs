param(
    [string]$LogDirectory = "runs\dataset_build_method_corrected_v2"
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$projectRoot = $PSScriptRoot
$logRoot = Join-Path $projectRoot $LogDirectory
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

# O FULL deve sempre anteceder o CASEONLY correspondente, pois fornece o
# suporte comum de datas e unidades usado na comparação controlada.
$configs = @(
    "config/config_rj_daily.yaml",
    "config/config_rj_daily_casesonly.yaml",
    "config/config_natal_daily.yaml",
    "config/config_natal_daily_casesonly.yaml",
    "config/config_rj_weekly.yaml",
    "config/config_rj_weekly_casesonly.yaml",
    "config/config_natal_weekly.yaml",
    "config/config_natal_weekly_casesonly.yaml"
)

Push-Location $projectRoot
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = "utf-8"
try {
    foreach ($config in $configs) {
        $configName = [System.IO.Path]::GetFileNameWithoutExtension($config)
        $logPath = Join-Path $logRoot "$configName.log"
        Write-Host "`n=== Construindo $config ==="

        & python "src/data_handling/build_dataset.py" --config $config 2>&1 |
            Tee-Object -FilePath $logPath

        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao construir $config. Consulte $logPath"
        }
    }
}
finally {
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    Pop-Location
}

Write-Host "`nConstrução corrigida concluída para os $($configs.Count) datasets."
