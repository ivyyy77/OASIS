# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import pdb
import sys

from sklearn.preprocessing import maxabs_scale

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.interhand.train import Dataset, HandAvatarDataset, HanCo, MixedDataset, Mix_Dataset, Hands11k_Dataset
from data.debug import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
import time
import json
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP
# from splatformer.utils import gpu_utils, gs_utils, loss_utils
# from splatformer.utils.optimizers import build_optimizer, build_scheduler
# from splatformer.utils.metrics import MetricComputer
# from splatformer.utils.log_utils import ProcessSafeLogger
import wandb
# wandb.login(key=os.getenv("WANDB_API_KEY"))  # configure locally
from tqdm import tqdm
from data.hand_dataset import merge_batch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from LHM.losses import _is_better
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)



flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb/', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
# flags.DEFINE_integer('iter', '2', 'Training iteration')
# flags.DEFINE_integer('iter', '11757', 'Training iteration')
flags.DEFINE_integer('iter', '20000', 'Training iteration')


FLAGS = flags.FLAGS

@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(argv):

    # train_loader = HandDataset(split='train')
    # # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=1)
    # # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)


    # data_iterator = iter(dataloader)
    # # data_iterator_1 = iter(dataloader_1)
    # batches = next(data_iterator)


    # FLAGS.parse_args()  # 手动解析命令行参数
    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("runner", type=str, help="Runner to launch")
    args, unknown = parser.parse_known_args()

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    runner = RunnerClass()

    # # features_offsets = ['means',  'scales', 'opacities', 'quats', 'features_dc', 'features_rest']
    # # features_activation = {'means': Tanh(), 'features_dc': Identity(), 'features_rest': Identity(), 'scales': Identity(), 'opacities': Identity(), 'quats': Identity()}
    # features_offsets = ['xyz',  'scaling', 'opacity', 'rotation', 'shs']
    # features_activation = {'xyz': Tanh(), 'shs': Identity(), 'scaling': Identity(), 'opacity': Identity(), 'rotation': Identity()}
    # # model = FeaturePredictor()
    # model = FeaturePredictor(backbone_type='PT', grid_resolution=384,
    #                          input_embed_to_mlp=False, input_feat_to_mlp=True, input_features=features_offsets, max_scale_normalized=0.01,
    #                          output_features=features_offsets, output_features_type='res',
    #                          # output_head_nlayer=4, output_head_type='mlp-relu', output_head_width=128,
    #                          output_head_nlayer=3, output_head_type='mlp-relu', output_head_width=128,
    #                          res_feature_activation=features_activation, resume_ckpt=None, sh_degree=1, zeroinit=True)

    # ==============================================================================================
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500',#'env://',
                            rank=0,       # Set appropriate rank per process
                            world_size=1,
                            # master_addr="127.0.0.1",
                            # master_port=29500
                            )   # Total number of processes)
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    # gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, bindings=None)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    iteration = FLAGS.iter
    # ========================================= DEBUG =========================================
    # model = runner.model.renderer.gs_net

    # ========================================= DEBUG =========================================
    #
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {num_params}")
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # # print(RunnerClass)
    # # sys.exit()
    # model = model.to(device_id)
    # model = DDP(model, device_ids=[device_id])
    # if FLAGS.only_eval == False:
    #   model.train()
    #   # 3. Optimizer
    #   optimizer, scheduler = {}, {}
    #   with gin.config_scope('pretrain'):
    #     optimizer['pretrain'] = build_optimizer(model.module)
    #     scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
    #   with gin.config_scope('train2D'):
    #     optimizer['train2D'] = build_optimizer(model.module)
    #     scheduler['train2D'] = build_scheduler(optimizer['train2D'])
    # # if rank==0:
    # #     wandb_run = wandb.init(project='debug_new_project', dir=FLAGS.wandb_dir)
    # #     if FLAGS.output_dir[-1] == '/':
    # #         FLAGS.output_dir = FLAGS.output_dir[:-1]
    # #     wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
    # # ==============================================================================================
    # # if rank == 0:
    # #     wandb_run = wandb.init(project='sign', dir=FLAGS.wandb_dir)  # resume=?

    # train_loader = ReconstructionDataset(split='train')
    # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)

    # from data.hand_dataset import create_dataset, merge_batch
    # print('===============================')
    # print("\033[91m[Loading MANO Training Dataset]\033[0m")
    # hand_train_loader = create_dataset('train')
    # dataloader = make_dataloader(hand_train_loader, shuffle=True, batch_size=2)

    #
    # train_loader = HandDataset(split='train')
    # # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)
    # # dataloader_1 = make_dataloader(train_loader, shuffle=True, batch_size=2)
    #
    #
    # data_iterator = iter(dataloader)
    # # data_iterator_1 = iter(dataloader_1)

    
    # train_loader = Dataset(dataset_path='./data/interhand',
    #                        data_type='train')
    # test_loader = Dataset(dataset_path='./data/interhand',
    #                        data_type='val',
    #                        skip=100)
    # train_loader = HanCo(dataset_path='./data/hanco')
    # hands11k_ds = Hands11k_Dataset(dataset_path='./data/hands11k/processed_test',
    #                                data_type='train')
    # train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)

    interhand_ds = Dataset(dataset_path='./data/interhand',
                           data_type='train')
    handco_ds = HanCo(dataset_path='./data/hanco', data_type='train')

    # train_loader = MixedDataset([interhand_ds, handco_ds, hands11k_ds], [0.3, 0.2, 0.5])
    train_loader = MixedDataset([interhand_ds, handco_ds], [0.5, 0.5])
    
    train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)

    # mixed_ds = Mix_Dataset([interhand_ds, handco_ds])

    # batch_sampler = DistributedSameDatasetBatchSampler(
    #     dataset_lengths=[len(interhand_ds), len(handco_ds)],
    #     ratios=[0.7, 0.3],     # 控制 InterHand/HanCo 采样比例
    #     batch_size=2,          # 你要的 batch_size=2（每个 rank 的本地 batch）
    #     num_batches=20000,     # 每个 epoch 的全局 batch 数（所有 rank 总和）
    #     seed=2025
    # )

    # train_dataloader = DataLoader(
    #     mixed_ds,
    #     batch_sampler=batch_sampler,
    #     num_workers=0,
    #     pin_memory=True,
    #     persistent_workers=False
    # )

    # train_dataloader = make_dataloader(mixed_ds, shuffle=True, batch_size=2)

    # test_dataloader = make_dataloader(test_loader, shuffle=True, batch_size=1)

    # test_hand = HandAvatarDataset(dataset_path='/scratch/groups/su004-neuralnet/zh174/InterHand/5',
    #                                             data_type='progress',
    #                                             skip=200)
    
    test_hand = HanCo(dataset_path='./data/hanco',
                            split='test')
    
    test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)


    data_iterator = iter(train_dataloader)
    # data_iterator_1 = iter(dataloader_1)
    writer = SummaryWriter(log_dir=os.path.join(FLAGS.output_dir, 'exp1'))

    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(False)
    # ==============================================================================================
    # pbar = tqdm(range(1, 2000 + 1))
    pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)
    # num_epochs = 20
    for iteration in pbar:
    # for epoch in range(num_epochs):
        # batches = next(data_iterator)   # 随机采样一个 subject ID, 并查找当前 ID 的所有视频
        # batches = merge_batch(batches)
        try:
            batches = next(data_iterator)
            # batches_1 = next(data_iterator_1)
        except:
            data_iterator = iter(train_dataloader)
            batches = next(data_iterator)
        # batches = merge_batch(batches)
        runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)    # hand
        # runner.run(batch=batches, scaler=scaler, optimizer=optimizer, scheduler=scheduler)    # human
        # runner.run()    # infer funtion


        # if iteration % 5 == 0:
        if iteration >= 1000 and iteration % 1000 == 0:

            # runner.save(iteration=iteration, is_latest=False)

            runner.save(is_latest=True)

            runner.load(is_latest=True)
            print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

            psnr_full = 0
            ssim_full = 0
            lpips_full = 0
            image_full = 0

            count = 0

            infer_batch = test_hand.get_img()
            # infer_img = infer_batch['original_image'].permute(1,2,0).to('cuda')
            # infer_pts = infer_batch['big_pose_world_vertex'].to('cuda')
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)


            # 外层枚举整个 test_dataloader
            for idx, test_batch in enumerate(test_dataloader):
                # 每个 batch 单独一个 tqdm，并且 position = idx+1 保证占用新的一行
                with tqdm(
                    total=1,                              # 只处理一个 batch
                    desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
                    leave=True,                            # 保留进度条
                    position=idx + 1,                       # 每个 batch 占一行
                    dynamic_ncols=True
                ) as pbar:

                    # 关闭梯度避免显存爆炸
                    with torch.no_grad():
                        loss_dict = runner.test_handavatar( gs_model_list, gs_densify_list, query_points,
                                                            test_batch, idx, iteration, pbar)

                    psnr_full  += loss_dict['psnr'].item()
                    ssim_full  += loss_dict['ssim'].item()
                    lpips_full += loss_dict['lpips'].item()
                    # image_full += loss_dict['image_l1'].item()
                    count += 1

                    # 在该 batch 的进度条上显示平均值与当前值
                    pbar.set_postfix({
                        "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                        "ssim":  f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                        "psnr":  f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                        # "L1":    f"{image_full/count:.4f} ({loss_dict['image_l1'].item():.4f})",
                    })

                    pbar.update(1)

                    # 及时释放 GPU 缓存
                    torch.cuda.empty_cache()

            print(f"\033[91m[Summary @iter {iteration}]\033[0m "
                f"PSNR: {psnr_full/count:.4f}, "
                f"SSIM: {ssim_full/count:.4f}, "
                f"LPIPS: {lpips_full/count:.4f}, "
                f"L1: {image_full/count:.4f}\n")


            # aggregate metrics
            current_metrics = {
                'lpips': lpips_full / count,
                'ssim': ssim_full / count,
                'psnr': psnr_full / count,
                'image_l1': image_full / count,
                'iteration_tested_on': int(iteration),
                'timestamp': time.time()
            }

            # create test results folder
            test_results_dir = os.path.join(runner.checkpoint_path, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            latest_results_path = os.path.join(test_results_dir, 'latest_test_results.json')
            best_results_path = os.path.join(test_results_dir, 'best_test_results.json')

            # save latest results
            with open(latest_results_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)

            # # load best if exists
            # best_metrics = None
            # if os.path.isfile(best_results_path):
            #     with open(best_results_path, 'r') as f:
            #         best_metrics = json.load(f)

            # 5) 判断是否已有历史 best
            if os.path.exists(best_results_path):
                try:
                    with open(best_results_path, 'r') as f:
                        best_metrics = json.load(f)
                except Exception as e:
                    print(f"[Warning] failed to load best_results_path ({best_results_path}): {e}")
                    best_metrics = None
            else:
                # 第一次测试：没有历史 best -> 直接把当前设为 best（并保存 checkpoint）
                best_metrics = None

            # ---- 比较是否更优 ----
            # 6) 若没有历史 best（第一次）或当前比 best 更优，则更新 best 并保存 iteration_ckpt
            if (best_metrics is None) or _is_better(current_metrics, best_metrics):
                # 更新 best json
                best_metrics = current_metrics
                with open(best_results_path, 'w') as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"\033[92m[+] New best model found at iteration {iteration}! Saving...]\033[0m")
                runner.save_checkpoint(iteration=iteration, is_latest=False)
            else:
                print(f"[ ] No improvement at iter {iteration}. Best stays at iteration {best_metrics.get('iteration_tested_on', 'N/A')} (LPIPS {best_metrics['lpips']:.6f}, PSNR {best_metrics['psnr']:.4f})")



    runner.save(iteration)
    writer.close()





if __name__ == "__main__":
    # main()
    app.run(main)



# python train_interhand.py infer.hand_lrm  model_name=LHM-1B