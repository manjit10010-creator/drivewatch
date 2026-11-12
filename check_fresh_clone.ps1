# Checks how the API behaves on a fresh clone, where data/gold/drivewatch.duckdb does not exist.
# It points the API at a warehouse path that does not exist (nothing of yours is touched), waits for start-up,
# asks /health and /feedback, shows the end of the API log, and removes the container.
# From the repo root:   powershell -ExecutionPolicy Bypass -File .\check_fresh_clone.ps1
# Needs the image drivewatch-api:slim and a free port 8000 (stop a running API first).

docker rm -f dwfresh *> $null
docker run -d --rm --name dwfresh -p 8000:8000 -e DRIVEWATCH_DB=data/gold/nope.duckdb -v "${PWD}:/workspace" drivewatch-api:slim | Out-Null
Start-Sleep -Seconds 10
Write-Host "--- GET /health  (expect HTTP 503 and 'could not be opened')"
curl.exe -s -w "\nHTTP %{http_code}\n" http://localhost:8000/health
Write-Host "--- GET /feedback  (expect HTTP 200 and [])"
curl.exe -s -w "\nHTTP %{http_code}\n" http://localhost:8000/feedback
Write-Host "--- end of the API log (a traceback for the missing file is expected)"
docker logs dwfresh --tail 12
docker rm -f dwfresh | Out-Null
