import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
from comfy.model_management import soft_empty_cache, get_torch_device
from .rife_arch import IFNet
from typing import *
import einops
import torch
import os

DEVICE = get_torch_device()

def load_model_file():
    return os.path.abspath(
        os.path.join(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), './ckpts', 'rife'
                )
            ), 'rife49.pth'
        )
    )

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def generic_frame_loop(
    frames,
    multiplier: Union[SupportsInt, List],
    return_middle_frame_function,
    *return_middle_frame_function_args,
    dtype = torch.bfloat16
):
    output_frames = torch.zeros(multiplier * frames.shape[0], *frames.shape[1:], dtype=dtype)
    out_len = 0
    frames = frames.to(dtype=dtype, device=DEVICE)

    for frame_itr in range(len(frames) - 1):
        frame0 = frames[frame_itr:frame_itr+1]
        frame1 = frames[frame_itr+1:frame_itr+2]
        output_frames[out_len] = frame0
        out_len += 1

        for middle_i in range(1, multiplier):
            middle_frame = return_middle_frame_function(
                frame0,
                frame1,
                middle_i / multiplier,
                *return_middle_frame_function_args
            ).detach()
            output_frames[out_len] = middle_frame
            out_len += 1

    output_frames[out_len] = frames[-1:]
    out_len += 1
    soft_empty_cache()
    return output_frames[:out_len]

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
                "multiplier": ("INT", {"default": 2, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"
    
    def execute(
        self,
        model,
        frames: torch.Tensor,
        multiplier: SupportsInt = 2,
    ):
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
            return model(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)
        
        args = [model, [8, 4, 2, 1], True, True]
        out = postprocess_frames(
            generic_frame_loop(frames, multiplier, return_middle_frame, *args)
        )
        return (out,)
    
NODE_CLASS_MAPPINGS = {
    "Interpolate": Interpolate,
    "Load Interpolation Model": LoadInterpolationModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Interpolate": "Interpolate",
    "LoadInterpolationModel": "LoadInterpolationModel"
}