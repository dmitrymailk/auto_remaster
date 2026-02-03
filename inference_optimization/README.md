- https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers/models/stable_diffusion
- https://blog.cerebrium.ai/improve-stable-diffusion-inference-by-50-with-tensorrt-or-aitemplate-429aa95b1709
- https://github.com/facebookincubator/AITemplate/tree/main/examples/05_stable_diffusion
- https://github.com/leejet/stable-diffusion.cpp


pip3 install torch==2.9.1 torchvision==0.24.1

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install typing-extensions==4.12.2

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

pip install -r inference_optimization\requirements_win.txt