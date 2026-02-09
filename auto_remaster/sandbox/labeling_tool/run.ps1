Write-Host "Starting Labeling Tool Backend..."
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
Read-Host -Prompt "Press Enter to exit"
