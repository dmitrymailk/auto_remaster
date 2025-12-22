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
import tempfile
import os
import shutil


class ComfyUIImageAPIMixImagesV1:
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
        part=0,
        parts=8,
        smooth_images="",
        crisp_images="",
        target_save_path_1="",
    ):
        os.system(f"mkdir -p {target_save_path_1}")

        # генерируем случайный номер для исполнителя
        self.client_id = str(uuid.uuid4())

        target_images = sorted(list(Path(target_save_path_1).glob("*.png")))
        target_images = sorted([int(num.stem) for num in target_images])

        smooth_images_files = sorted(list(Path(smooth_images).glob("*.png")))
        crisp_images_files = sorted(list(Path(crisp_images).glob("*.png")))

        current_part = list(
            chunked(
                # ["{:07d}".format(i) for i in range(len(smooth_images_files))],
                [i for i in range(len(smooth_images_files))],
                len(smooth_images_files) // parts,
            ),
        )
        current_part = current_part[part]
        current_part_set = set(current_part)

        current_part_completed = [
            num for num in target_images if num in current_part_set
        ]
        # first time
        if len(current_part_completed) == 0:
            last_frame = current_part[0]
        else:
            last_frame = current_part_completed[-1]
            last_frame += 1
        print("last_frame", last_frame, current_part[-1])

        if last_frame > current_part[-1]:
            return "END"

        with open(self.workflow_path, "r") as f:
            workflow = json.load(f)

        # print(f"New image")
        # final images
        with tempfile.TemporaryDirectory() as dataset_images_dir:
            workflow["125"]["inputs"]["image"] = str(
                smooth_images_files[last_frame].absolute()
            )
            workflow["126"]["inputs"]["image"] = str(
                crisp_images_files[last_frame].absolute()
            )
            temp_output_img = f""
            target_mix_image = f"{target_save_path_1}/{'{:07d}'.format(last_frame)}.png"
            save_num = f"{'{:07d}'.format(last_frame)}.png"
            workflow["131"]["inputs"]["filename_prefix"] = dataset_images_dir + "/"
            workflow["131"]["inputs"]["current_frame"] = last_frame

            prompt_id = self.queue_prompt(workflow)["prompt_id"]

            # print(self.client_id)
            ws = websocket.WebSocket()
            ws.connect(
                "ws://{}/ws?clientId={}".format(
                    self.server_address,
                    self.client_id,
                )
            )

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

            mix_images = sorted(list(Path(dataset_images_dir).glob("*.png")))
            print(mix_images[-1])
            mix_images[-1].rename(f"{dataset_images_dir}/{save_num}")
            shutil.copyfile(f"{dataset_images_dir}/{save_num}", target_mix_image)


ports_parts = [
    [1340, 0],
    [1341, 1],
    [1342, 2],
    [1343, 3],
    [1344, 4],
    [1345, 5],
    [1346, 6],
    [1347, 7],
    [1348, 8],
]
num = 4
port = ports_parts[num][0]
part = ports_parts[num][1]
parts = 8
comfy_images_process = ComfyUIImageAPIMixImagesV1(
    server_address=f"127.0.0.1:{port}",
    workflow_path="/code/showcases/showcase_11/nfs_mix_wan_sdxl_road_v4_api.json",
)
target_save_path = (
    # "/code/comfyui_sandbox/video_renders/render_nfs_4screens_5_sdxl_1_wan"
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_wan_mix"
)


smooth_images = "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_wan/"
crisp_images = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_dataset"
)

if __name__ == "__main__":
    result = ""

    while result != "END":
        result = comfy_images_process.process_image_folder(
            part=part,
            parts=parts,
            smooth_images=smooth_images,
            crisp_images=crisp_images,
            target_save_path_1=target_save_path,
        )
