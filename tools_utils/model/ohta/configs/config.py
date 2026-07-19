# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..
# *************************************************************************


import os
import sys
from pathlib import Path
import argparse
import torch
from yacs.config import CfgNode as CN


_C = CN(new_allowed=True)


_C.resume = False


_C.eval_iter = 10000000

_C.render_folder_name = ""
_C.ignore_non_rigid_motions = False
_C.render_skip = 1
_C.render_frames = 100

_C.num_workers = 4


def get_cfg_defaults():
    return _C.clone()

def parse_cfg(cfg):
    cfg.logdir = os.path.join('./output', cfg.category, cfg.task, cfg.subject.replace('/', '_'), cfg.experiment)

def determine_primary_secondary_gpus(cfg):
    print("------------------ GPU Configurations ------------------")
    cfg.n_gpus = torch.cuda.device_count()
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        if cfg.n_gpus > 1:
            cfg.secondary_gpus = [g for g in all_gpus]
        else:
            cfg.secondary_gpus = cfg.primary_gpus
        print(f"Primary GPUs: {cfg.primary_gpus}")
        print(f"Secondary GPUs: {cfg.secondary_gpus}")
    else:
        print(f"CPU job")
    print("--------------------------------------------------------")

def make_cfg():
    cfg = get_cfg_defaults()
    config_dir = Path(__file__).resolve().parent
    cfg.merge_from_file(str(config_dir / 'default.yaml'))
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_dir / 'interhand' / 'ohta_train.yaml'))
    cfg.smpl_cfg.lbs_weights = str(
        config_dir.parents[3] / 'pretrained_models' / 'manohd_lbs_weights.pth'
    )
    parse_cfg(cfg)

    determine_primary_secondary_gpus(cfg)

    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default='./tools_utils/model/ohta/configs/interhand/ohta_train.yaml', type=str)
parser.add_argument("--type", default="freepose", type=str)
parser.add_argument("--input", default='', type=str)
parser.add_argument("--edit", action='store_true')
parser.add_argument("--checkpoint", default='', type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

cfg = make_cfg()
