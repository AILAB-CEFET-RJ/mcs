param(
    [int]$StartYear = 2024,
    [int]$EndYear = 2025,
    [string]$CnesReference = "2512",
    [switch]$SkipSinan,
    [switch]$SkipCnes,
    [switch]$SkipEra5
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$root = $PSScriptRoot
$outputRoot = Join-Path $root "data\raw\holdout_2024_2025"
$sinanDir = Join-Path $outputRoot "sinan"
$cnesDir = Join-Path $outputRoot "cnes"
$era5RjDir = Join-Path $outputRoot "era5\rj_monthly"
$era5NatalDir = Join-Path $outputRoot "era5\natal_monthly"

New-Item -ItemType Directory -Force -Path $sinanDir, $cnesDir, $era5RjDir, $era5NatalDir | Out-Null

Push-Location $root
try {
    # O Pipfile fica no diretório pai. O Pipenv o encontra subindo a árvore.
    & pipenv run python -c "import cdsapi, pysus; print('Dependencias de download: OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Ambiente incompleto. Execute 'pipenv install' na raiz SLR antes do download."
    }

    if (-not $SkipSinan) {
        foreach ($year in $StartYear..$EndYear) {
            Write-Host "`n=== SINAN DENG $year ==="
            & pipenv run python src/ingestion/download/download_sinan.py `
                DENG $year $sinanDir --log INFO
            if ($LASTEXITCODE -ne 0) {
                throw "Falha no download do SINAN para $year. Verifique se o arquivo anual já foi publicado pelo DATASUS."
            }
        }
    }

    if (-not $SkipCnes) {
        Write-Host "`n=== CNES RJ $CnesReference ==="
        & pipenv run python src/ingestion/download/download_cnes.py `
            ST RJ $CnesReference (Join-Path $cnesDir "STRJ$CnesReference.dbc") --log INFO
        if ($LASTEXITCODE -ne 0) { throw "Falha no download do CNES/RJ." }

        Write-Host "`n=== CNES RN $CnesReference ==="
        & pipenv run python src/ingestion/download/download_cnes.py `
            ST RN $CnesReference (Join-Path $cnesDir "STRN$CnesReference.dbc") --log INFO
        if ($LASTEXITCODE -ne 0) { throw "Falha no download do CNES/RN." }
    }

    if (-not $SkipEra5) {
        Write-Host "`n=== ERA5 single-levels: Rio de Janeiro ==="
        & pipenv run python src/ingestion/download/download_era5.py `
            --dataset single-levels `
            --start-year $StartYear --end-year $EndYear `
            --north -22.0 --south -23.0 --west -44.0 --east -42.0 `
            --file-prefix RJ --out-dir $era5RjDir --zero-pad-month
        if ($LASTEXITCODE -ne 0) { throw "Falha no download ERA5/RJ." }

        Write-Host "`n=== ERA5-Land: Natal ==="
        & pipenv run python src/ingestion/download/download_era5.py `
            --dataset era5-land `
            --start-year $StartYear --end-year $EndYear `
            --north -5.45 --south -6.15 --west -35.60 --east -34.90 `
            --file-prefix NATAL --out-dir $era5NatalDir --zero-pad-month
        if ($LASTEXITCODE -ne 0) { throw "Falha no download ERA5/Natal." }
    }
}
finally {
    Pop-Location
}

Write-Host "`nDownloads concluídos em: $outputRoot"
Write-Host "Os arquivos ainda precisam ser validados, consolidados e pré-processados antes da construção dos datasets."
