param(
    [string]$Dataset = "mlg-ulb/creditcardfraud",
    [string]$OutputDir = "data/raw"
)

$ErrorActionPreference = "Stop"

Write-Host "Dataset alvo: $Dataset"
Write-Host "Pasta destino: $OutputDir"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$destination = Join-Path $projectRoot $OutputDir
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleConfig = Join-Path $kaggleDir "kaggle.json"
$kaggleAccessToken = Join-Path $kaggleDir "access_token"

if ((-not (Test-Path $kaggleConfig)) -and (-not (Test-Path $kaggleAccessToken)) -and (-not $env:KAGGLE_API_TOKEN)) {
    Write-Host ""
    Write-Host "Credenciais do Kaggle nao encontradas em nenhum destes locais:"
    Write-Host $kaggleConfig
    Write-Host $kaggleAccessToken
    Write-Host "Variavel de ambiente KAGGLE_API_TOKEN"
    Write-Host ""
    Write-Host "Como resolver:"
    Write-Host "1. Entre em https://www.kaggle.com/settings"
    Write-Host "2. Crie um token de API"
    Write-Host "3. Salve o token novo em $kaggleAccessToken ou o arquivo legado em $kaggleConfig"
    Write-Host "4. Rode este script de novo"
    exit 1
}

$kaggleCommand = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggleCommand) {
    Write-Host ""
    Write-Host "Kaggle CLI nao encontrado."
    Write-Host "Instale com:"
    Write-Host "python -m pip install kaggle"
    exit 1
}

kaggle datasets download -d $Dataset -p $destination --unzip

$csvPath = Join-Path $destination "creditcard.csv"
if (Test-Path $csvPath) {
    Write-Host ""
    Write-Host "Dataset real pronto em: $csvPath"
    Write-Host "Agora rode:"
    Write-Host "python -m src.train"
}
else {
    Write-Host ""
    Write-Host "Download finalizado, mas creditcard.csv nao foi encontrado. Verifique a pasta data/raw."
    exit 1
}
