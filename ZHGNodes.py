import os
import torch
import numpy as np
import math
import cv2
import json
import copy
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import gaussian_filter
from skimage import exposure
import folder_paths
from nodes import MAX_RESOLUTION
import comfy

from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
from .reactor_swapper import swap_face, get_current_faces_model, analyze_faces, get_face_single
from .reactor_logger import logger

from .utils import tensor_to_pil, pil_to_tensor, tensor2pil, pil2tensor, pil2mask
from .modules import shared
from .modules.scripts import USDUMode, USDUSFMode, Script
from .modules.processing import StableDiffusionProcessing
from .modules.upscaler import UpscalerData

import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import comfy.model_management
from .sam_hq.predictor import SamPredictorHQ
from .sam_hq.build_sam_hq import sam_model_registry
from .local_groundingdino.datasets import transforms as T
from .local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from .local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from .local_groundingdino.models import build_model as local_groundingdino_build_model
import glob

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    return [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break

logger = logging.getLogger('comfyui_zhihuige')

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(
    sam_model,
    image,
    boxes
):
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)


class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            }
        }
    CATEGORY = "ZHG Nodes/segment"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model, )


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            }
        }
    CATEGORY = "ZHG Nodes/segment"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt1": ("STRING", {}),
                "prompt2": ("STRING", {}),
                "prompt3": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "ZHG Nodes/segment"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt1, prompt2, prompt3, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt1,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            
            H, W = image.size[1], image.size[0]
            for i in range(boxes.size(0)):
                boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
                boxes[i][:2] -= boxes[i][2:] / 2
                boxes[i][2:] += boxes[i][:2]
            return boxes

            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))


class GroundingDinoPIPESAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhg_pipe": ('ZHG_PIPE', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "ZHG Nodes/segment"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, zhg_pipe, image, prompt, threshold):
        sam_model, grounding_dino_model = zhg_pipe
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = "ZHG Nodes/mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask):
        out = 1.0 - mask
        return (out,)

class ToZHGSAMPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "sam_model": ('SAM_MODEL', {}),
                    "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                    },
                }

    RETURN_TYPES = ("ZHG_PIPE", )
    RETURN_NAMES = ("zhg_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ZHG Nodes/Pipe"

    def doit(sam_model, grounding_dino_model):
        pipe = sam_model, grounding_dino_model
        return (pipe, )


class FromZHGSAMPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"zhg_pipe": ("ZHG_PIPE",), }, }

    RETURN_TYPES = ("SAM_MODEL", "GROUNDING_DINO_MODEL")
    RETURN_NAMES = ("sam model", "grounding dino model")
    FUNCTION = "doit"

    CATEGORY = "ZHG Nodes/Pipe"

    def doit(self, zhg_pipe):
        sam_model, grounding_dino_model = zhg_pipe
        return sam_model, grounding_dino_model


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return prepare_inputs(USDU_base_inputs())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ZHG Nodes/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers = [None]
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode
        )

        #
        # Running the script
        #
        script = Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


class UltimateSDUpscaleNoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ZHG Nodes/upscaling"

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):

        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None
        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(upscaled_image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1, force_uniform_tiles, tiled_decode
        )

        script = Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1)

        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

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
                "filename": os.path.join(full_output_folder, file) + ' [zhihuige]',
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

    CATEGORY = "ZHG Nodes/mask"

    def combine(self, mask1, mask2, index=0):
        
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

    CATEGORY = "ZHG Nodes/mask"

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


class GetMaskArea:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask": ("MASK",),
                "h_offset": ("INT", {"default": 100, "step":10}),
                "h_cutoff": ("FLOAT", {"default": 0, "step":0.001}),
                "max_width": ("INT", {"default": 1600, "min": 0, "max": MAX_RESOLUTION, "step": 100}),
                "min_height": ("INT", {"default": 2400, "min": 0, "max": MAX_RESOLUTION, "step": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    FUNCTION = "getimage"

    CATEGORY = "ZHG Nodes/mask"

    def get_mask_aabb(self, masks):
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

        b = masks.shape[0]

        bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
        is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
        for i in range(b):
            mask = masks[i]
            if mask.numel() == 0:
                continue
            if torch.max(mask != 0) == False:
                is_empty[i] = True
                continue
            y, x = torch.where(mask)
            bounding_boxes[i, 0] = torch.min(x)
            bounding_boxes[i, 1] = torch.min(y)
            bounding_boxes[i, 2] = torch.max(x)
            bounding_boxes[i, 3] = torch.max(y)

        return bounding_boxes, is_empty

    def getimage(self, image1, image2, mask, h_offset = 100, h_cutoff=0, max_width=1600, min_height=2400):
        bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
        boxes, is_empty = self.get_mask_aabb(bounds)

        padding = 0.02
        box = boxes[0]
        H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
        hh = int(int(H.item()) * (1.0 - h_cutoff))
        yy = int(Y.item()) - h_offset
        hh = int((hh + 50) * (1 + padding))
        ww = int(int(W.item()) * (1 + 20 * padding))
        #X = int(X.item()) - 10
        #W = int(W.item()) + 20
        xx = max(int(int(X.item()) - ww / ( 1 + 20 * padding) * 10 * padding), 0)
        image1_copy = copy.deepcopy(image1)
        image11 = image1_copy[:,yy:yy+hh,xx:xx+ww]
        image2_copy = copy.deepcopy(image2)
        image22 = image2_copy[:,yy:yy+hh,xx:xx+ww]
        mask = mask[:,yy:yy+hh,xx:xx+ww]
        return (image11, image22, mask)

# IMAGE LEVELS NODE

class ZHG_Image_Levels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                "mid_level": ("FLOAT", {"default": 127.5, "min": 0.0, "max": 255.0, "step": 0.1}),
                "white_level": ("FLOAT", {"default": 255, "min": 0.0, "max": 255.0, "step": 0.1}),
                "bright_thresold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 255.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_image_levels"

    CATEGORY = "ZHG Nodes/Image/Adjustment"

    def apply_image_levels(self, image, black_level, mid_level, white_level, bright_thresold):
        tensor_images = []
        for timg in image:
            img = tensor2pil(timg)
            levels = self.AdjustLevels(black_level, mid_level, white_level)
            tensor_images.append(pil2tensor(levels.adjust(img, bright_thresold)))
        tensor_images = torch.cat(tensor_images, dim=0)

        # Return adjust image tensor
        return (tensor_images, )


    class AdjustLevels:
        def __init__(self, min_level, mid_level, max_level):
            self.min_level = min_level
            self.mid_level = mid_level
            self.max_level = max_level

        def isBright(self, pil_image, thresold):
            gray_img = np.array(pil_image.convert('L'))
            
            # 获取形状以及长宽
            img_shape = gray_img.shape
            height, width = img_shape[0], img_shape[1]
            size = gray_img.size
            # 灰度图的直方图
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            
            # 计算灰度图像素点偏离均值(128)程序
            a = 0
            ma = 0
            reduce_matrix = np.full((height, width), 128)
            shift_value = gray_img - reduce_matrix
            shift_sum = sum(map(sum, shift_value))

            da = shift_sum / size
            # 计算偏离128的平均偏差
            for i in range(256):
                ma += (abs(i-128-da) * hist[i])
            m = abs(ma / size)
            # 亮度系数
            k = abs(da) / m
            # print(k)
            if k[0] > thresold:
                # 过亮
                if da > 0:
                    #print("过亮")
                    return True
                else:
                    #print("过暗")
                    return False
            else:
                #print("亮度正常")
                return False

        def adjust(self, im, thresold = 1.0):

            if not self.isBright(im, thresold):
                im_arr = np.array(im)
                im_arr[im_arr < self.min_level] = self.min_level
                im_arr = (im_arr - self.min_level) * \
                    (255 / (self.max_level - self.min_level))
                im_arr[im_arr < 0] = 0
                im_arr[im_arr > 255] = 255
                im_arr = im_arr.astype(np.uint8)
                
                im = Image.fromarray(im_arr)
                im = ImageOps.autocontrast(im, cutoff=self.max_level)

            return im

class SmoothEdge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "sigma": ("FLOAT", {"default":1.5, "min":0.0, "max":12.0, "step":0.1}),
                "gamma": ("INT", {"default": 20}),
            },
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "smooth"

    CATEGORY = "ZHG Nodes/mask"

    def img2tensor(self, img, bgr2rgb=True, float32=True):

        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    def smooth_region(self, image, tolerance):
        from scipy.ndimage import gaussian_filter
        image = image.convert("L")
        mask_array = np.array(image)
        smoothed_array = gaussian_filter(mask_array, sigma=tolerance)
        threshold = np.max(smoothed_array) / 2
        smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
        smoothed_mask = exposure.adjust_gamma(smoothed_mask, gamma=20)
        smoothed_image = Image.fromarray(smoothed_mask, mode="L")
        return ImageOps.invert(smoothed_image.convert("RGB"))

    def smooth(self, masks, sigma=128, gamma=20):
        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = self.smooth_region(pil_image, sigma)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            regions_tensor = torch.cat(regions, dim=0)
            return (regions_tensor,)
        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(mask_np, mode="L")
            region_mask = self.smooth_region(pil_image, sigma)
            region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
            return (region_tensor,)

NODE_CLASS_MAPPINGS = {
    "Combine ZHGMasks": MaskCombineOp,
    "Cover ZHGMasks": MaskCoverOp,
    "ZHG FaceIndex": GetFaceIndex,
    "ZHG SaveImage": SaveZHGImage,
    "ZHG SmoothEdge": SmoothEdge,
    "ZHG GetMaskArea": GetMaskArea,
    "ZHG Image Levels": ZHG_Image_Levels,
    "ZHG UltimateSDUpscale": UltimateSDUpscale,
    'SAMModelLoader (zhihuige)': SAMModelLoader,
    'GroundingDinoModelLoader (zhihuige)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (zhihuige)': GroundingDinoSAMSegment,
    'GroundingDinoPIPESegment (zhihuige)': GroundingDinoPIPESAMSegment,
    'InvertMask (zhihuige)': InvertMask,
    'From ZHG pip': FromZHGSAMPipe,
    'To ZHG pip': ToZHGSAMPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Combine ZHGMasks": "Combine ZHGMasks",
    "Cover ZHGMasks": "Cover ZHGMasks",
    "ZHG FaceIndex": "ZHG FaceIndex",
    "ZHG SaveImage": "ZHG SaveImage",
    "ZHG SmoothEdge": "ZHG SmoothEdge",
    "ZHG GetMaskArea": "ZHG GetMaskArea",
    "ZHG Image Levels": "ZHG Image Levels",
    "ZHG UltimateSDUpscale": "ZHG Ultimate SD Upscale",
    'SAMModelLoader (zhihuige)': 'SAMModelLoader (zhihuige)',
    'GroundingDinoModelLoader (zhihuige)': 'GroundingDinoModelLoader (zhihuige)',
    'GroundingDinoSAMSegment (zhihuige)': 'GroundingDinoSAMSegment (zhihuige)',
    'GroundingDinoPIPESegment (zhihuige)': 'GroundingDinoPIPESegment (zhihuige)',
    'InvertMask (zhihuige)': 'InvertMask (zhihuige)',
    'From ZHG pip': 'From ZHG pip',
    'To ZHG pip': 'To ZHG pip',
}
