timestamp=$(date +%s)

cd .. && python comfyui_sandbox/process_folder.py > comfy_$timestamp.log 2>&1 &