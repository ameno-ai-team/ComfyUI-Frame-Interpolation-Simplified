import os
import sys
import math
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
from comfy.model_management import get_torch_device
from .rife_arch import IFNet
from typing import *
import torch
import os

DEVICE = get_torch_device()

def load_model_file():
    return os.path.abspath(
        os.path.join(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), './ckpts'
                )
            ), 'rife49.pth'
        )
    )

def preprocess_frames(frames):
    return frames[..., :3].permute(0, 3, 1, 2)

def postprocess_frames(frames):
    return frames.permute(0, 2, 3, 1)[..., :3].cpu()

def best_multiplier(multiplier: float) -> float:
    """
    Snap multiplier to next integer if fractional part >= 0.75.
    Above 0.75, ceilling and trimming is cheaper than running fractional.
    Below 0.75, run as-is — ceiling wastes too many frames with no gain.
    """
    frac = multiplier % 1.0
    if frac == 0:
        return multiplier
    if frac >= 0.75:
        return float(math.ceil(multiplier))
    return multiplier

def generic_frame_loop(
    model,
    frames,
    multiplier: float,
    dtype=torch.float16
):
    n = len(frames)
    out_count = round((n - 1) * multiplier) + 1
    output_frames = torch.zeros(out_count, *frames.shape[1:], dtype=dtype)
    frames = frames.to(dtype=dtype, device=DEVICE)

    inv_multiplier = 1.0 / multiplier

    for out_idx in range(out_count):
        t = out_idx * inv_multiplier
        src_idx = int(t)
        frac = t - src_idx

        if frac < 1e-6 or src_idx >= n - 1:
            output_frames[out_idx] = frames[min(src_idx, n - 1)]
        else:
            frame0 = frames[src_idx:src_idx + 1]
            frame1 = frames[src_idx + 1:src_idx + 2]
            output_frames[out_idx] = model(
                frame0, frame1, frac, [8, 4, 2, 1], True, True
            ).detach()
    return output_frames

class LoadInterpolationModel:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = ("RIFE_VFI_MODEL", )
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"
    
    def execute(self):
        model_path = load_model_file()
        model = IFNet(arch_ver='4.7')
        model.load_state_dict(torch.load(model_path))
        model.eval().to(DEVICE).half()
        return (model, )

class Interpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("RIFE_VFI_MODEL", ),
                "frames": ("IMAGE", ),
                "multiplier": ("FLOAT", {"default": 2.0, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"
    
    def execute(
        self,
        model,
        frames: torch.Tensor,
        multiplier: float = 2.0,
    ):
        n = len(frames)
        target_count = round((n - 1) * multiplier) + 1

        run_multiplier = best_multiplier(multiplier)
        snapped = run_multiplier != multiplier

        frames = preprocess_frames(frames)
        output_frames = generic_frame_loop(model, frames, run_multiplier)

        if snapped:
            output_frames = output_frames[:target_count]

        return (postprocess_frames(output_frames),)
    
NODE_CLASS_MAPPINGS = {
    "Interpolate": Interpolate,
    "LoadInterpolationModel": LoadInterpolationModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Interpolate": "Interpolate",
    "LoadInterpolationModel": "LoadInterpolationModel"
}