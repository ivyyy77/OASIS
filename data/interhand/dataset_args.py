# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..
# *************************************************************************

from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    if cfg.category in ['ohta',] and cfg.task == 'interhand':
        dataset_attrs.update({
            "interhand_train": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": f'/data/hand3d/HandAvatar-main/data/InterHand/{cfg.interhand.fps}',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "interhand_val": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": f'/data/hand3d/HandAvatar-main/data/InterHand/{cfg.interhand.fps}',
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'interhand',
            },
            "interhand_test": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": f'/data/hand3d/HandAvatar-main/data/InterHand/{cfg.interhand.fps}',
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'interhand',
            },
        })
    else:
        dataset_attrs.update({
            "our_data": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": '/scratch/groups/su004-neuralnet/zh174/hand-data/our_data',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
            "our_data_test": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": '/home/z/zh174/our_data_eval',
                # "dataset_path": '/home/z/zh174/eval_debug',
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                # "src_type": 'interhand',
                },
            "interhand_test": {
                # "dataset_path": f'data/InterHand/{cfg.interhand.fps}',
                "dataset_path": f'/data/hand3d/HandAvatar-main/data/InterHand/{cfg.interhand.fps}',
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                # "src_type": 'interhand',
            },
        })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
