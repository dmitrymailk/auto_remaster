scp -r /home/dimweb/auto_remaster/dataset dimweb@192.168.120.25:/home/dimweb/auto_remaster

scp -r /home/dimweb/auto_remaster/ComfyUI/custom_nodes dimweb@192.168.120.25:/home/dimweb/auto_remaster/ComfyUI/custom_nodes

scp /home/dimweb/auto_remaster/ComfyUI/models/checkpoints/RealVisXL_V4.0_Lightning.safetensors dimweb@192.168.120.25:/home/dimweb/auto_remaster/ComfyUI/models/checkpoints/

scp -r /home/dimweb/auto_remaster/ComfyUI/models/controlnet/ dimweb@192.168.120.25:/home/dimweb/auto_remaster/ComfyUI/models/

scp -r /home/dimweb/auto_remaster/ComfyUI/models/upscale_models/ dimweb@192.168.120.25:/home/dimweb/auto_remaster/ComfyUI/models/

cat render_nfs_4screens_6_sdxl_1_part_1.tar.gz | tar -xvf -
cat render_nfs_4screens_6_sdxl_1_part_2.tar.gz | tar -xvf -
cat render_nfs_4screens_6_sdxl_1_part_3.tar.gz | tar -xvf -

scp -r /home/dimweb/auto_remaster/vpn dimweb@192.168.120.25:/home/dimweb/auto_remaster
scp -r /home/dimweb/auto_remaster/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_upscale_2x.zip dimweb@192.168.120.210:/home/dimweb/auto_remaster/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_upscale_2x.zip
