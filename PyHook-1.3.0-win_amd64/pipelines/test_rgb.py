from pipeline_utils import build_variable, read_value, use_local_python
import numpy as np

# print("NIGGGGGERS")
# print(sys.modules['numpy.random._generator'])
# # print(sys.modules.keys())
with use_local_python():
    import torch
    print(torch.tensor(1, device='cuda'))


name = "[Test] RGB"
version = "1.2.9"
desc = "Dummy pipeline for testing. Tests RGB manipulation and multiple UI settings."
supports = [32, 64]

settings = {
    "R": build_variable(0, -255, 255, 1, "Red channel modifier."),
    "G": build_variable(0, -255, 255, 1, "Green channel modifier."),
    "B": build_variable(0, -255, 255, 1, "Blue channel modifier."),
    "Brightness": build_variable(0.5, 0.0, 1.0, 0.01, "Brightness setting."),
    "Invert": build_variable(False, None, None, None, "Invert all colors."),
}


def before_change_settings(key: str, value: float) -> None:
    print(f"BEFORE: Settings change: {key} = {value}.")


def after_change_settings(key: str, value: float) -> None:
    print(f"AFTER: Settings change: {key} = {value}.")


def on_load() -> None:
    print(f'Pipeline="{name}" was loaded.')


def on_frame_process(
    frame: np.array, width: int, height: int, frame_num: int
) -> np.array:
    # bright_val = (read_value(settings, "Brightness") - 0.5) * 255 * 2
    # modifiers = [
    #     int(read_value(settings, "R") + bright_val),
    #     int(read_value(settings, "G") + bright_val),
    #     int(read_value(settings, "B") + bright_val)
    # ]
    # for i in range(3):
    #     if modifiers[i] < 0:
    #         frame[:, :, i] -= np.minimum(frame[:, :, i], -modifiers[i])
    #     else:
    #         frame[:, :, i] += np.minimum(255 - frame[:, :, i], modifiers[i])
    # if read_value(settings, "Invert"):
    frame = torch.from_numpy(frame).cuda()
    # print(frame.device)
    # frame =  255 - frame
    return frame.cpu().numpy()
    # return frame


def on_unload() -> None:
    print(f'Pipeline="{name}" was unloaded.')
