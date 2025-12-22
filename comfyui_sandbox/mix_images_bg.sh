timestamp=$(date +%s)

cd .. && python comfyui_sandbox/mix_images.py > comfy_upscale_$timestamp.log 2>&1 &