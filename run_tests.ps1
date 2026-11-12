# Runs the DriveWatch API tests inside the slim API image (it has fastapi, duckdb and lightgbm).
# From the repo root:   powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
# Docker Desktop must be running. Needs the image drivewatch-api:slim (docker build -f docker/api/Dockerfile.slim -t drivewatch-api:slim docker/api).

docker rm -f dwtest *> $null
docker run -d --rm --name dwtest -v "${PWD}:/workspace" drivewatch-api:slim sleep 600 | Out-Null
docker exec dwtest pip install -q pytest httpx 2>&1 | Select-String -NotMatch "WARNING|notice"
docker exec dwtest python -m pytest -q -p no:cacheprovider tests/test_predict.py tests/test_feedback.py tests/test_fleet.py tests/test_fresh_clone.py
$code = $LASTEXITCODE
docker rm -f dwtest | Out-Null
exit $code
