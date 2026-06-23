import functools
import gc
import multiprocessing as mp
import os
import pdb
import time
import traceback as tb
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool, Process, cpu_count
from multiprocessing.pool import Pool
from typing import Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from accelerate.logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

timings = {}
BATCH_SIZE = 64


def load_model(checkpoint, use_torchscript=False):
    model = torch.load(checkpoint, map_location='cuda')
    return model
    # if use_torchscript:
    #     return torch.jit.load(checkpoint)
    # else:
    #     return torch.export.load(checkpoint).module()




DEFAULT_CHECKPOINT='data/hand3d/zh174/pretrained_models/hamba/checkpoints/hamba.ckpt'

def load_hamba(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()
    elif model_cfg.MODEL.BACKBONE.TYPE == 'vmamba' \
        or model_cfg.MODEL.BACKBONE.TYPE == 'fastvit_ma36':
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 224, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 224 for vmamba backbone"
        model_cfg.MODEL.BBOX_SHAPE = [224,224]
        model_cfg.freeze()
        print(">" * 50)

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        if 'PRETRAINED_WEIGHTS_INIT_REGRESSION' in model_cfg.MODEL.keys():
            model_cfg.MODEL.pop('PRETRAINED_WEIGHTS_INIT_REGRESSION')
        model_cfg.freeze()

    print("checkpoint_path: ", checkpoint_path)
    model = HAMBA.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg




class HambaWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        encoder_feat_dim: int = 384,
        resolution=1024,
        antialias: bool = True,
    ):
        super().__init__()
        from Hamba.hamba.models import HAMBA, download_models, load_hamba, DEFAULT_CHECKPOINT
        self.model = self._build_hamba(model_name)
        self.model = self.model['state_dict']
        self.resolution = resolution

        # self.antialias = antialias
        # self.register_buffer(
        #     "mean", torch.Tensor([0.4844, 0.4570, 0.4062]), persistent=False
        # )
        # self.register_buffer(
        #     "std", torch.Tensor([0.2295, 0.2236, 0.2256]), persistent=False
        # )
        
        freeze = True
        if freeze:
            self._freeze()
        else:
            raise NotImplementedError(
                "Fine-tuning is not supported yet."
            )  # sapiens is too larger to finetune the model end-to-end.


    def _freeze(self):
        logger.warning(f"======== Freezing Hamba Model ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    
    @staticmethod
    def _build_hamba(model_name: str, pretrained: bool = True):

        logger.debug(f"Using Hamba model: {model_name}")
        USE_TORCHSCRIPT = "_torchscript" in model_name

        # build the model from a checkpoint file
        model = load_model(model_name, use_torchscript=USE_TORCHSCRIPT)
        # if not USE_TORCHSCRIPT:
        #     raise NotImplementedError
        # else:
        #     dtype = torch.float32  # TorchScript models use float32
        #     model = model.cuda()
        return model


    def _preprocess_image(
        self, image: torch.tensor, resolution: int = 1024
    ) -> torch.Tensor:

        _, __, H, W = image.shape
        max_size = max(H, W)
        H_pad = max_size - H
        W_pad = max_size - W
        pad_size = (
            W_pad // 2,
            max_size - (W + W_pad // 2),
            H_pad // 2,
            max_size - (H + H_pad // 2),
            0,
            0,
            0,
            0,
        )

        image = F.pad(image, pad_size, value=1)

        image = kornia.geometry.resize(
            image,
            (resolution, resolution),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        image = kornia.enhance.normalize(image, self.mean, self.std)

        return image


    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized

        image = self._preprocess_image(image, self.resolution)

        # NOTE that, only supports
        patch_h, patch_w = (
            image.shape[-2] // 16,
            image.shape[-1] // 16,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (out_local,) = self.model(image)

        out_global = None
        if out_global is not None:
            raise NotImplementedError("Global feature is not supported yet.")
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)

        return ret
