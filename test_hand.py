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

from tqdm import tqdm
from sklearn.preprocessing import maxabs_scale

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.debug import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
# from splatformer.utils import gpu_utils, gs_utils, loss_utils
# from splatformer.utils.optimizers import build_optimizer, build_scheduler
# from splatformer.utils.metrics import MetricComputer
# from splatformer.utils.log_utils import ProcessSafeLogger
import wandb
# wandb.login(key=os.getenv("WANDB_API_KEY"))  # configure locally
from tqdm import tqdm
from data.hand_dataset import merge_batch

from torch.utils.tensorboard import SummaryWriter
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
flags.DEFINE_integer('iter', '50000', 'Training iteration')


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

    # ==============================================================================================
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29501',#'env://',
                            rank=0,       # Set appropriate rank per process
                            world_size=1,
                            # master_addr="127.0.0.1",
                            # master_port=29500
                            )   # Total number of processes)
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    iteration = FLAGS.iter
    # ========================================= DEBUG =========================================
    # model = runner.model.renderer.gs_net


    # print("\033[91m[========== Loading Our Training Dataset ==========]\033[0m")
    # train_loader = HandDataset(split='train')
    # print("\033[91m[========== Loading Our Testing Dataset ==========]\033[0m")
    test_loader = HandDataset(split='test')
    # train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)
    test_dataloader = make_dataloader(test_loader, shuffle=True, batch_size=2)



    # ================================  Train ==============================================================
    # ================================  Train ==============================================================
    # ================================  Train ==============================================================
    
    # pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)
    # # num_epochs = 20
    # for iteration in pbar:
    # # for epoch in range(num_epochs):
    #     # batches = next(data_iterator)   # 随机采样一个 subject ID, 并查找当前 ID 的所有视频
    #     # batches = merge_batch(batches)
    #     try:
    #         batches = next(data_iterator)
    #         # batches_1 = next(data_iterator_1)
    #     except:
    #         data_iterator = iter(train_dataloader)
    #         batches = next(data_iterator)
    #     # batches = merge_batch(batches)
    #     runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)    # hand
    #     # runner.run(batch=batches, scaler=scaler, optimizer=optimizer, scheduler=scheduler)    # human
    #     # runner.run()    # infer funtion

    #     # if iteration % 5 == 0:
    #     if iteration % 5000 == 0:
    #         runner.save(iteration)
    #         # runner.test(merge_batch(batch_test), iteration)

    #         # # 测试：对整个 test_dataloader 做一次逐-batch 测试，用一个临时的子进度条
    #         # test_pbar = tqdm(test_dataloader, desc=f"Test @iter {iteration}", leave=False, position=1, dynamic_ncols=True)
    #         # for test_batch in test_pbar:
    #         #     # runner.test 应该处理单个 batch 的推理并返回/记录度量
    #         #     # runner.save(iteration)
    #         #     runner.test(test_batch, iteration, test_pbar)   # hand
                
    #         #     # 可选择把一些实时 metric 填到 test_pbar.postfix（如果 runner.test 返回 metric）
    #         #     # test_pbar.set_postfix({'some_metric': f"{metric:.4f}"})
    #         # test_pbar.close()            
    #         # # runner.test(batches, iteration)
    #         # # exit(0)
    # runner.save(iteration)
    # writer.close()

    # ================================  TEST ==============================================================
    # ================================  TEST ==============================================================
    # ================================  TEST ==============================================================

    # 需要测试的迭代列表
    test_iters = {30000}

    for iteration in test_iters:
        runner.load_model(iteration)
        print(f"\033[91m[========== Running test for iteration {iteration} ==========]\033[0m")

        psnr_full = 0
        ssim_full = 0
        lpips_full = 0
        image_full = 0

        count = 0

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
                    loss_dict = runner.test(test_batch, idx, iteration, pbar)

                psnr_full  += loss_dict['psnr'].item()
                ssim_full  += loss_dict['ssim'].item()
                lpips_full += loss_dict['lpips'].item()
                image_full += loss_dict['image_l1'].item()
                count += 1

                # 在该 batch 的进度条上显示平均值与当前值
                pbar.set_postfix({
                    "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                    "ssim":  f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                    "psnr":  f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                    "L1":    f"{image_full/count:.4f} ({loss_dict['image_l1'].item():.4f})",
                })
                pbar.update(1)

                # 及时释放 GPU 缓存
                torch.cuda.empty_cache()

        print(f"\033[91m[Summary @iter {iteration}]\033[0m "
            f"PSNR: {psnr_full/count:.4f}, "
            f"SSIM: {ssim_full/count:.4f}, "
            f"LPIPS: {lpips_full/count:.4f}, "
            f"L1: {image_full/count:.4f}\n")
        # f"\033[91m[========== Running test for iteration {iteration} ==========]\033[0m"


if __name__ == "__main__":
    # main()
    app.run(main)



