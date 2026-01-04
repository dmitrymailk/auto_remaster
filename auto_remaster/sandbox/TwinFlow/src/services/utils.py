import torch
import typing
from typing import Optional, Union

from collections import OrderedDict, defaultdict

from .tools import create_logger

logger = create_logger(__name__)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            ema_name = (
                model_name.replace("module.", "")
                if model_name.startswith("module.")
                else f"module.{model_name}"
            )
            if ema_name in ema_params:
                ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                raise KeyError(f"Parameter name {model_name} not found in EMA model!")