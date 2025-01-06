timestamp=$(date +%s)

# cd ../ComfyUI && python main.py --cuda-device 0 2>&1 &
# cd ../ComfyUI && python main.py --cuda-device 1 --port 1337  > comfy_$timestamp.log 2>&1 &
# cd ../ComfyUI && python main.py --cuda-device 2 --port 1338  > comfy_$timestamp.log 2>&1 &
cd ../ComfyUI && python main.py --cuda-device 3 --port 1339  > comfy_$timestamp.log 2>&1 &