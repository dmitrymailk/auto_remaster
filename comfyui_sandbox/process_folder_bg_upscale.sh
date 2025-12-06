timestamp=$(date +%s)

# cd .. && python comfyui_sandbox/process_folder_upscale_v1.py > comfy_$timestamp.log 2>&1 &
cd .. && python comfyui_sandbox/process_folder_upscale_v2.py > comfy_upscale_$timestamp.log 2>&1 &