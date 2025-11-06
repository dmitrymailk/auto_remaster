pip install -r requirements.txt
# cd ao && pip install -e . && cd ..
# cd cut-cross-entropy && pip install -e . && cd ..
# cd Liger-Kernel && pip install -e . && cd ..
# cd torchtune && pip install -e . && cd ..

export MAX_JOBS=10
# for 5090(reduce arch list)
export FLASH_ATTN_CUDA_ARCHS="80;120"
pip install flash-attn==2.8.3 --no-build-isolation