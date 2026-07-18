$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$out = Join-Path (Split-Path -Parent $repo) "rcm-python.tar.gz"

if (Test-Path $out) {
    Remove-Item -LiteralPath $out
}

tar -czf $out -C $repo `
    --exclude=.git `
    --exclude=.github `
    --exclude=.venv `
    --exclude=frontend `
    --exclude=tests `
    --exclude=build `
    --exclude=dist `
    --exclude=__pycache__ `
    --exclude=.pytest_cache `
    --exclude=.pytest-tmp `
    .

Write-Host "Created $out"
