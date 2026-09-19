[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectId,

    [ValidateNotNullOrEmpty()]
    [string]$Region = "asia-southeast1",

    [ValidatePattern('^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')]
    [string]$ServiceName = "train-condition-monitoring"
)

$ErrorActionPreference = "Stop"

$gcloudCommand = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloudCommand) {
    $standardUserInstall = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path -LiteralPath $standardUserInstall) {
        $gcloudCommand = $standardUserInstall
    }
}

if (-not $gcloudCommand) {
    throw "Google Cloud CLI (gcloud) is required. Install it from https://cloud.google.com/sdk/docs/install"
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot

try {
    & $gcloudCommand config set project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Could not select GCP project '$ProjectId'." }

    & $gcloudCommand services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --quiet
    if ($LASTEXITCODE -ne 0) { throw "Could not enable the required GCP APIs." }

    & $gcloudCommand run deploy $ServiceName `
        --source . `
        --project $ProjectId `
        --region $Region `
        --allow-unauthenticated `
        --port 8080 `
        --cpu 2 `
        --memory 2Gi `
        --concurrency 1 `
        --timeout 600 `
        --max-instances 8 `
        --execution-environment gen2 `
        --quiet
    if ($LASTEXITCODE -ne 0) { throw "Cloud Run deployment failed." }

    $serviceUrl = & $gcloudCommand run services describe $ServiceName `
        --project $ProjectId `
        --region $Region `
        --format "value(status.url)"
    if ($LASTEXITCODE -ne 0) { throw "Deployment completed, but the service URL could not be read." }

    Write-Host "Deployment complete: $serviceUrl"
    Write-Host "Health check: $serviceUrl/api/health"
}
finally {
    Pop-Location
}
