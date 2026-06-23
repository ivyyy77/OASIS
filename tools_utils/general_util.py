
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)



# argparse.ArgumentParser(description="OpenLRM launcher")
#    parser.add_argument("runner", type=str, help="Runner to launch")
#    args, unknown = parser.parse_known_args()

#    if args.runner not in REGISTRY_RUNNERS:
 #       raise ValueError("Runner {} not found".format(args.runner))

#    RunnerClass = REGISTRY_RUNNERS[args.runner]
#    runner = RunnerClass()

    # features_offsets = ['means',  'scales', 'opacities', 'quats', 'features_dc', 'features_rest']
    # features_activation = {'means': Tanh(), 'features_dc': Identity(), 'features_rest': Identity(), 'scales': Identity(), 'opacities': Identity(), 'quats': Identity()}
 #   features_offsets = ['xyz',  'scaling', 'opacity', 'rotation', 'shs']
  #  features_activation = {'xyz': Tanh(), 'shs': Identity(), 'scaling': Identity(), 'opacity': Identity(), 'rotation': Identity()}
    # model = FeaturePredictor()
    # model = FeaturePredictor(backbone_type='PT', grid_resolution=384,
    #                          input_embed_to_mlp=False, input_feat_to_mlp=True, input_features=features_offsets, max_scale_normalized=0.01,
    #                          output_features=features_offsets, output_features_type='res',
    #                          # output_head_nlayer=4, output_head_type='mlp-relu', output_head_width=128,
    #                          output_head_nlayer=3, output_head_type='mlp-relu', output_head_width=128,
    #                          res_feature_activation=features_activation, resume_ckpt=None, sh_degree=1, zeroinit=True)

    # ==============================================================================================
#    dist.init_process_group(backend='nccl', init_method='env://',
#                            rank=0,       # Set appropriate rank per process
 #                           world_size=1,
                            # master_addr="127.0.0.1",
                            # master_port=29500
 #                           )   # Total number of processes)
  #  rank = dist.get_rank()
   # torch.cuda.set_device(rank % torch.cuda.device_count())
    # print(f"Start running basic DDP example on rank {rank}.")
    #device_id = rank % torch.cuda.device_count()
    # gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
#    gin.parse_config_files_and_bindings(FLAGS.gin_file, bindings=None)
 #   os.makedirs(FLAGS.output_dir, exist_ok=True)
 #   set_seed(42)

    # ========================================= DEBUG =========================================
    # model = runner.model.renderer.gs_net
  #  model = runner.model.renderer.grid_offset
    # ========================================= DEBUG =========================================

   # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Number of trainable parameters: {num_params}")
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print(RunnerClass)
    # sys.exit()
    #model = model.to(device_id)
#    model = DDP(model, device_ids=[device_id])
 #   if FLAGS.only_eval == False:
 #     model.train()
      # 3. Optimizer
 #     optimizer, scheduler = {}, {}
 #     with gin.config_scope('pretrain'):
  #      optimizer['pretrain'] = build_optimizer(model.module)
   #     scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
    #  with gin.config_scope('train2D'):
     #   optimizer['train2D'] = build_optimizer(model.module)
      #  scheduler['train2D'] = build_scheduler(optimizer['train2D'])
    # if rank==0:
    #     wandb_run = wandb.init(project='debug_new_project', dir=FLAGS.wandb_dir)
    #     if FLAGS.output_dir[-1] == '/':
    #         FLAGS.output_dir = FLAGS.output_dir[:-1]
    #     wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
    # ==============================================================================================
    # if rank == 0:
    #     wandb_run = wandb.init(project='sign', dir=FLAGS.wandb_dir)  # resume=?

#    train_loader = ReconstructionDataset(split='train')
 #   test_loader = ReconstructionDataset(split='test')

  #  dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)
   # data_iterator = iter(dataloader)

    #scaler = torch.cuda.amp.GradScaler()
    #torch.autograd.set_detect_anomaly(False)
    # ==============================================================================================
    #pbar = tqdm(range(1, 200 + 1))
  #  for iteration in pbar:
   #     batches = next(data_iterator)   # 随机采样一个 subject ID, 并查找当前 ID 的所有视频
        # try:
        #     batches = next(data_iterator)
        # except:
        # runner.run()    # infer funtion


    # model.eval()
    # for test_dataset, test_loader in build_testloader().items():
    #     metrics, metrics_input = evaluation(model, test_loader=test_loader,
    #                                         output_dir=FLAGS.output_dir + f'/{FLAGS.eval_subdir}/{test_dataset}',
    #                                         compare_with_input=FLAGS.compare_with_input,
    #                                         save_as_single=True,
    #                                         save_viewer=FLAGS.save_viewer,
    #                                         output_gt=True, compare_with_pseudo=False)
    #     if dist.get_rank() == 0:
    #         logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, FLAGS.eval_subdir, 'eval.log')).get_logger()
    #         metric_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    #         logger.info(f'Test-{test_dataset}: {metric_str}')
    #         if FLAGS.compare_with_input:
    #             metric_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics_input.items()])
    #             logger.info(f'Input 3DGS: Test-{test_dataset}: {metric_str}')
    #     dist.barrier()
    #
    # dist.destroy_process_group()



