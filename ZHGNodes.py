import os
import torch
import numpy as np
import math
import cv2
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo


from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
from .reactor_swapper import swap_face, get_current_faces_model, analyze_faces, get_face_single
from .reactor_logger import logger
from .reactor_utils import tensor_to_pil

import folder_paths
from nodes import MAX_RESOLUTION

class SaveZHGImage:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.disable_metadata = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "output_dir": ("STRING", {"default": "d:\\ZHG"}),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "ZHG Nodes"

    def save_images(self, images, output_dir="d:\\ZHG", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not self.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": os.path.join(full_output_folder, file),
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class MaskCombineOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine"

    CATEGORY = "ZHG Nodes"

    def combine(self, mask1, mask2, index):
        
        if (index == 0):
            result = mask1
        else:
            result = mask2

        return (result,)

class MaskCoverOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("MASK",),
                "source": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "cover"

    CATEGORY = "ZHG Nodes"

    def cover(self, destination, source, x, y):
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))

        left, top = (x, y,)
        right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = destination[:, top:bottom, left:right]

        output[:, top:bottom, left:right] = destination_portion * source_portion
        
        output_ones = output.norm(1)
        destination_ones = destination_portion.norm(1)
        if (output_ones / destination_ones) < 0.2:
            result = 1
        else:
            result = 0

        return (result,)

class GetFaceIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "getindex"

    CATEGORY = "ZHG Nodes"

    def getindex(self, input_image):
        target_img = tensor_to_pil(input_image)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        target_faces = analyze_faces(target_img)
        target_face, wrong_gender = get_face_single(target_img, target_faces)
        if target_face is not None:
            result = 1
        else:
            result = 0
        return (result,)

NODE_CLASS_MAPPINGS = {
    "Combine ZHGMasks": MaskCombineOp,
    "Cover ZHGMasks": MaskCoverOp,
    "ZHG FaceIndex": GetFaceIndex,
    "ZHG SaveImage": SaveZHGImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Combine ZHGMasks": "Combine ZHGMasks",
    "Cover ZHGMasks": "Cover ZHGMasks",
    "ZHG FaceIndex": "ZHG FaceIndex",
    "ZHG SaveImage": "ZHG SaveImage",
}
