import os
import torch
import numpy as np
import math
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
from .reactor_swapper import swap_face, get_current_faces_model, analyze_faces, get_face_single
from .reactor_logger import logger
from .reactor_utils import tensor_to_pil


DELIMITER = '|'
cached_clipseg_model = None
VERY_BIG_SIZE = 1024 * 1024


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
    "ZHG FaceIndex": GetFaceIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Combine ZHGMasks": "Combine ZHGMasks",
    "ZHG FaceIndex": "ZHG FaceIndex",
}
