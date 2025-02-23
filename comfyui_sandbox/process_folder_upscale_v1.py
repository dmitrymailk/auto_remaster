import os
from pathlib import Path
from PIL import Image
from more_itertools import chunked
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import requests
import io


class ComfyUIImageAPIUpscaleV1:
    def __init__(
        self,
        server_address="127.0.0.1:8188",
        workflow_path="",
    ):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.workflow_path = workflow_path

    def queue_prompt(
        self,
        prompt,
    ):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            "http://{}/prompt".format(self.server_address), data=data
        )
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(
        self,
        filename,
        subfolder,
        folder_type,
    ):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            "http://{}/view?{}".format(self.server_address, url_values)
        ) as response:
            return response.read()

    def get_history(
        self,
        prompt_id,
    ):
        with urllib.request.urlopen(
            "http://{}/history/{}".format(self.server_address, prompt_id)
        ) as response:
            return json.loads(response.read())

    def get_images(
        self,
        ws,
        prompt,
    ):
        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history["outputs"]:
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                images_output = []
                if "images" in node_output:
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def process_image_folder(
        self,
        original_images_path="",
        batch_size=4,
        target_save_path_1="",
        target_save_path_2="",
        total_parts=4,
        part_num=0,
    ):
        self.client_id = str(uuid.uuid4())

        original_images = sorted(list(Path(original_images_path).glob("*.png")))
        target_images = sorted(list(Path(target_save_path_1).glob("*.png")))

        # total_parts = 4
        original_images_parts = list(
            chunked(
                original_images,
                len(original_images) // total_parts + 1,
            )
        )

        # part_num = 0
        original_images_part = original_images_parts[part_num]
        target_images_names = set([item.stem for item in target_images])
        images_to_process = [
            item
            for item in original_images_part
            if not item.stem in target_images_names
        ][:batch_size]
        if len(images_to_process) == 0:
            return "END"

        with open(self.workflow_path, "r") as f:
            workflow = json.load(f)

        # diffustion steps
        # workflow["240"]["inputs"]["steps"] = 5
        # workflow["201"]["inputs"]["batch"] = True
        # input images
        workflow["268"]["inputs"]["paths"] = "\n".join(
            [str(item) for item in images_to_process]
        )
        print(workflow["268"]["inputs"]["paths"])
        # save prefix
        save_prefix = f"nfs_4screens_5_sdxl_{self.client_id}"
        workflow["261"]["inputs"]["filename_prefix"] = save_prefix
        workflow["252"]["inputs"]["filename_prefix"] = save_prefix + "v2"

        print(self.client_id)
        ws = websocket.WebSocket()
        ws.connect(
            "ws://{}/ws?clientId={}".format(
                self.server_address,
                self.client_id,
            )
        )

        images = self.get_images(ws, workflow)
        for node_id in images:
            for image_data, image_original_path in zip(
                images[node_id],
                images_to_process,
            ):

                image = Image.open(io.BytesIO(image_data))
                if int(node_id) == 272:
                    image.save(f"{target_save_path_1}/{image_original_path.stem}.png")
                if int(node_id) == 270:
                    image.save(f"{target_save_path_2}/{image_original_path.stem}.png")

        # clean output images from comfyui
        # os.system(f"rm /code/ComfyUI/output/{save_prefix}*.png")
        # os.system(f"rm /code/ComfyUI/output/{save_prefix+'v2'}*.png")


comfy_images_process = ComfyUIImageAPIUpscaleV1(
    server_address="127.0.0.1:8188",
    workflow_path="/code/showcases/showcase_10/workflow_ultimate_upscale_simple_nfs_mix_api_v3.json",
)
original_images_path = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_5_sdxl_1"
)
target_save_path_1 = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_5_sdxl_1_upscale_1x"
)
target_save_path_2 = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_5_sdxl_1_upscale_2x"
)

# port = 8188
# port = 1337
# port = 1338
# port = 1339
# comfy_images_process = ComfyUIImageAPI(
#     server_address=f"127.0.0.1:{port}",
#     # workflow_path="showcases/showcase_9/nfs_canny_normal_map_sdxl_batch_list_api.json",
#     workflow_path="showcases/showcase_9/nfs_canny_normal_map_sdxl_batch_list_api_v2.json",
# )
# original_images_path = "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6"
# target_save_path = "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1"

if __name__ == "__main__":
    result = ""
    while result != "END":
        result = comfy_images_process.process_image_folder(
            original_images_path=original_images_path,
            batch_size=1,
            target_save_path_1=target_save_path_1,
            target_save_path_2=target_save_path_2,
            total_parts=1,
            part_num=0,
        )
