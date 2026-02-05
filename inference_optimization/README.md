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


### Model optimization
- [Stable Fast performance optimization](https://github.com/chengzeyi/stable-fast)
- [Triton fork for Windows support](https://github.com/woct0rdho/triton-windows)
- [Lightweight inference library for ONNX files, written in C++. It can run Stable Diffusion XL 1.0 on a RPI Zero 2 (or in 298MB of RAM) but also Mistral 7B on desktops and servers. ARM, x86, WASM, RISC-V supported. Accelerated by XNNPACK. Python, C# and JS(WASM) bindings available.](https://github.com/vitoplantamura/OnnxStream)
- [AITemplate](https://facebookincubator.github.io/AITemplate/install/index.html)
- [TensorRT for RTX](https://github.com/NVIDIA/TensorRT-RTX)
- [acceleration of Stable Diffusion and ControlNet pipeline using TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion)
- [ONNX NVIDIA TensorRT RTX Execution Provider](https://thevishalagarwal.github.io/onnxruntime/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)
- [TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro)
- [LiteRT, successor to TensorFlow Lite. is Google's On-device framework for high-performance ML & GenAI deployment on edge platforms, via efficient conversion, runtime, and optimization](https://github.com/google-ai-edge/LiteRT)
- [Support PyTorch model conversion with LiteRT](https://github.com/google-ai-edge/litert-torch)
- [Zero-copy with GPU acceleration](https://ai.google.dev/edge/litert/next/gpu)
- [Open Machine Learning Compiler Framework](https://github.com/apache/tvm/)
- [MNN is a blazing fast, lightweight deep learning framework, battle-tested by business-critical use cases in Alibaba.](https://github.com/alibaba/MNN) 
- [black-forest-labs/FLUX.1-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx)
- https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- [A Library to Quantize and Compress Deep Learning Models for Optimized Inference on Native Windows RTX GPUs](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows)
- [AI Model Optimization Toolkit for the ONNX Runtime](https://github.com/microsoft/Olive)
- [This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion)
- [AI Model Efficiency Toolkit (AIMET)](https://github.com/quic/aimet)
- [AI Model Efficiency Toolkit Model Optimization for Snapdragon Devices](https://docs.qualcomm.com/doc/80-64748-1/topic/model_updates.html)

### Screen capture
- [How to render Direct3D scene to texture, process it with CUDA and render result to screen?](https://stackoverflow.com/questions/77237723/how-to-render-direct3d-scene-to-texture-process-it-with-cuda-and-render-result)
- [Screen capture sample, clarified](https://windowsasusual.blogspot.com/2020/12/screen-capture-sample-clarified-few.html)
- [DXcam A Python high-performance screen capture library for Windows using Desktop Duplication API](https://github.com/ra1nty/DXcam)
