from safetensors.torch import load_file, save_file
import torch
import json

path = "/code/ComfyUI/models/controlnet/Shakker-Labs_FLUX.1-dev-ControlNet-Union-Pro.safetensors"  # input file


# read safetensors metadata
def read_safetensors_metadata(path):
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        metadata = header.get("__metadata__", {})
        return metadata


metadata = read_safetensors_metadata(path)
print(json.dumps(metadata, indent=4))  # show metadata

sd_pruned = dict()  # initialize empty dict

state_dict = load_file(path)  # load safetensors file
for key in state_dict:  # for each key in the safetensors file
    sd_pruned[key] = state_dict[key].to(torch.float8_e4m3fn)  # convert to fp8

# save the pruned safetensors file
save_file(
    sd_pruned,
    "/code/ComfyUI/models/controlnet/Shakker-Labs_FLUX.1-dev-ControlNet-Union-Pro-fp8.safetensors",
    metadata={"format": "pt", **metadata},
)
