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
    cfg.logdir = os.path.join('handavatar/out', cfg.category, cfg.task, cfg.subject.replace('/', '_'), cfg.experiment)

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
    cfg.merge_from_file(str(config_dir / 'interhand' / 'test_cap0.yaml'))
    cfg.smpl_cfg.lbs_weights = str(
        config_dir.parents[3] / 'pretrained_models' / 'manohd_lbs_weights.pth'
    )
    parse_cfg(cfg)

    determine_primary_secondary_gpus(cfg)

    return cfg

def make_cfg_left():
    cfg = get_cfg_defaults()
    config_dir = Path(__file__).resolve().parent
    cfg.merge_from_file(str(config_dir / 'default_left.yaml'))
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_dir / 'interhand' / 'test_cap0_left.yaml'))
    cfg.smpl_cfg.lbs_weights = str(
        config_dir.parents[3] / 'pretrained_models' / 'manohd_lbs_weights.pth'
    )
    parse_cfg(cfg)

    determine_primary_secondary_gpus(cfg)

    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default='./tools_utils/model/handavatar/configs/interhand/test_cap0.yaml', type=str)
parser.add_argument("--type", default="freepose", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
parser.add_argument('--conf', default='./code/confs/subject1.conf', type=str)
parser.add_argument('--is_continue', default=False, action="store_true",
                    help='If set, indicates continuing from a previous run.')
parser.add_argument('--checkpoint', default='latest', type=str,
                    help='The checkpoint epoch number in case of continuing from a previous run.')

cfg = make_cfg()
cfg_left = make_cfg_left()
