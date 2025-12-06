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


class ComfyUIImageAPIUpscaleV2:
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
        dataset: list = None,
        batch_size=4,
        target_save_path_1="",
        target_save_path_2="",
        total_parts=4,
        part_num=0,
    ):
        os.system(f"mkdir -p {target_save_path_1}")
        os.system(f"mkdir -p {target_save_path_2}")

        # генерируем случайный номер для исполнителя
        self.client_id = str(uuid.uuid4())

        # original_images = sorted(list(Path(original_images_path).glob("*.png")))
        original_images = dataset
        target_images = sorted(list(Path(target_save_path_1).glob("*.png")))

        # разбиваем на части
        # total_parts = 4
        original_images_parts = list(
            chunked(
                ["{:07d}".format(i) for i in range(len(original_images))],
                len(original_images) // total_parts + 1,
            ),
        )

        # part_num = 0
        # берем в текущем процессе только нужную часть
        original_images_part = original_images_parts[part_num]
        # составляем уникальный список имен чтобы с ними больше не работать
        target_images_names = set([item.stem for item in target_images])
        # фильтруем
        images_to_process = [
            item for item in original_images_part if not item in target_images_names
        ][:batch_size]
        if len(images_to_process) == 0:
            return "END"

        with open(self.workflow_path, "r") as f:
            workflow = json.load(f)

        # diffustion steps
        # workflow["240"]["inputs"]["steps"] = 5
        # workflow["201"]["inputs"]["batch"] = True
        # input images
        # создаем временную папку чтобы данные картинки удалились после
        # обработки
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary folder created at: {temp_dir}")
            for im_num in images_to_process:
                dataset[int(im_num)]["edited_image"].save(f"{temp_dir}/{im_num}.png")

            images_to_process = [
                f"{temp_dir}/{im_num}.png" for im_num in images_to_process
            ]

            # создаем батч
            workflow["268"]["inputs"]["paths"] = "\n".join(
                # [str(item) for item in images_to_process]
                [item for item in images_to_process]
            )
            print(workflow["268"]["inputs"]["paths"])
            # save prefix
            # save_prefix = f"nfs_4screens_5_sdxl_{self.client_id}"
            # сохраняем
            save_prefix = f"nfs_4screens_6_sdxl_{self.client_id}"
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
            # получаем изображения
            images = self.get_images(ws, workflow)
            for node_id in images:
                for image_data, image_original_path in zip(
                    images[node_id],
                    images_to_process,
                ):

                    image = Image.open(io.BytesIO(image_data))
                    if int(node_id) == 272:
                        image.save(
                            # f"{target_save_path_1}/{image_original_path.stem}.png"
                            f"{target_save_path_1}/{Path(image_original_path).stem}.png"
                        )
                    if int(node_id) == 270:
                        image.save(
                            f"{target_save_path_2}/{Path(image_original_path).stem}.png"
                        )

            # clean output images from comfyui
            # os.system(f"rm /code/ComfyUI/output/{save_prefix}*.png")
            # os.system(f"rm /code/ComfyUI/output/{save_prefix+'v2'}*.png")


# port = 8188
port = 1337
part_num = 1
comfy_images_process = ComfyUIImageAPIUpscaleV2(
    server_address=f"127.0.0.1:{port}",
    workflow_path="/code/showcases/showcase_10/workflow_ultimate_upscale_simple_nfs_mix_api_v3.json",
)
# original_images_path = (
#     "/code/comfyui_sandbox/video_renders/render_nfs_4screens_5_sdxl_1"
# )
target_save_path_1 = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_upscale_1x"
)
target_save_path_2 = (
    "/code/comfyui_sandbox/video_renders/render_nfs_4screens_6_sdxl_1_upscale_2x"
)


from datasets import load_dataset

# dataset_name = "dim/nfs_pix2pix_1920_1080_v5"
dataset_name = "dim/nfs_pix2pix_1920_1080_v6"
dataset = load_dataset(
    dataset_name,
    # cache_dir="/code/dataset/nfs_pix2pix_1920_1080_v5",
    cache_dir="/code/dataset/nfs_pix2pix_1920_1080_v6",
)
dataset = dataset["train"]
if __name__ == "__main__":
    result = ""
    while result != "END":
        result = comfy_images_process.process_image_folder(
            dataset=dataset,
            batch_size=1,
            target_save_path_1=target_save_path_1,
            target_save_path_2=target_save_path_2,
            total_parts=50,
            part_num=part_num,
        )
