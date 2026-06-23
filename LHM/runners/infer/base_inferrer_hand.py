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


from abc import abstractmethod

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from LHM.runners.abstract import Runner

logger = get_logger(__name__)


class Inferrer_hand(Runner):

    EXP_TYPE: str = None

    def __init__(self):
        super().__init__()

        torch._dynamo.config.disable = True
        self.accelerator = Accelerator()

        self.model: torch.nn.Module = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def device(self):
        return self.accelerator.device

    @abstractmethod
    def _build_model(self, cfg):
        pass

    @abstractmethod
    def infer_single(self, *args, **kwargs):
        pass
    
    # # @abstractmethod
    def save(self, iteration=None, is_latest=False):
        # self.save(iteration)
        self.save_checkpoint(iteration=iteration, is_latest=is_latest)

    # @abstractmethod
    def load(self, iteration=None, is_latest=False, checkpoint_path=None):
        # self.load_model(iteration, checkpoint_path)
        self.load_checkpoint(iteration=iteration, is_latest=is_latest, checkpoint_path=checkpoint_path)
    

    def load_model(self, iteration=None, checkpoint_path=None):
        self.load_model(iteration, checkpoint_path)
    
    
    # @abstractmethod
    def finetune_model(self, iteration, checkpoint_path=None):
        self.finetune_model(iteration, checkpoint_path)

    @abstractmethod
    def test(self,batch, idx, iteration, pbar):
        self.test(batch=batch, idx=idx, iteration=iteration, pbar=pbar)

    @abstractmethod
    def test_handavatar(self, gs_model_list, gs_densify_list, query_points, batch, idx, iteration, pbar):
        self.test_handavatar( gs_model_list=gs_model_list, gs_densify_list=gs_densify_list, query_points=query_points,
            batch=batch, idx=idx, iteration=iteration, pbar=pbar)

    @abstractmethod
    def infer_handavatar(self, batch):
        self.infer_handavatar(batch=batch)

    # def run(self, splatformer_model=None, batch=None, optimizer=None, scheduler=None, scaler=None):
    def run(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None):
        self.infer(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)

    def run_edit(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)

    def run_edit_wild(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_edit_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)

    def run_edit_wild_inversion(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        """Inversion stage for edit wild: mask out edit region, fit global color."""
        self.finetune_edit_wild_inversion(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)

    def run_edit_wild_stage2(self, batch=None, pseudo_batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None,
                              edit_unmask_iter=300, edit_mask_weight=30.0, pseudo_view_weight=0.1):
        """Stage2 for edit wild: masked pseudo-GT learning -> unmasked edit learning with vis masking."""
        self.finetune_edit_wild_stage2(batch=batch, pseudo_batch=pseudo_batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar,
                                        total_iters=total_iters, edit_unmask_iter=edit_unmask_iter,
                                        edit_mask_weight=edit_mask_weight, pseudo_view_weight=pseudo_view_weight)

    def run_wild(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)
        # self.finetune_wild_with_boundary_color(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)

    def run_wild_ohta(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_wild_inversion(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)
        # self.finetune_wild_with_boundary_color(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)

    def run_wild_stage_2(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None,
                         is_text_to_avatar=None, enable_pose_refine=False, pose_refine_lr=1e-4,
                         edit_mask_weight=0.0):
        if is_text_to_avatar is not None:
            setattr(self, '_is_text_to_avatar', bool(is_text_to_avatar))
        # self.infer_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        # [FIXED v5] More conservative scaling constraints.
        # Invisible region threshold 0.012 (was 0.015) to balance texture preservation vs. stability.
        self.finetune_wild_stage2_only(
            batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters,
            # enable_scaling_constraint=False,
            enable_scaling_constraint=True,
            # scaling_min_threshold=0.004,  # Global min scale (original value)
            # invisible_scaling_threshold=0.008,  # Invisible points (moderately increased from 0.008)
            # visible_scaling_threshold=0.008,  # Visible points (unchanged)
            # for Hanco dataset 
            scaling_min_threshold=0.003,  # Global min scale (original value)
            invisible_scaling_threshold=0.005,  # Invisible points (moderately increased from 0.008)
            visible_scaling_threshold=0.002,  # Visible points (unchanged)
            # 0.012 & 0.008 对于 interhand的微调
            # invisible_scaling_threshold=0.012,  # Invisible points (moderately increased from 0.008)
            # visible_scaling_threshold=0.008,  # Visible points (unchanged)
            enable_pose_refine=enable_pose_refine,
            pose_refine_lr=pose_refine_lr,
            edit_mask_weight=edit_mask_weight,
        )
        # self.finetune_wild_with_boundary_color(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)


    def run_wild_stage_2_interhand(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None,
                         is_text_to_avatar=None):
        if is_text_to_avatar is not None:
            setattr(self, '_is_text_to_avatar', bool(is_text_to_avatar))
        # self.infer_wild(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        # [FIXED v5] More conservative scaling constraints.
        # Invisible region threshold 0.012 (was 0.015) to balance texture preservation vs. stability.
        self.finetune_wild_stage2_only(
            batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters,
            enable_scaling_constraint=True,
            scaling_min_threshold=0.004,  # Global min scale (original value)
            # invisible_scaling_threshold=0.008,  # Invisible points (moderately increased from 0.008)
            # visible_scaling_threshold=0.005,  # Visible points (unchanged)
            # for Hanco dataset 
            # scaling_min_threshold=0.003,  # Global min scale (original value)
            # invisible_scaling_threshold=0.005,  # Invisible points (moderately increased from 0.008)
            # visible_scaling_threshold=0.002,  # Visible points (unchanged)
            # 0.012 & 0.008 对于 interhand的微调
            invisible_scaling_threshold=0.012,  # Invisible points (moderately increased from 0.008)
            visible_scaling_threshold=0.008,  # Visible points (unchanged)
        )

    def run_visible_only(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None,
                         gradient_mask_invisible=True, pretrain_regularization=0.1):
        """
        New strategy: Only learn from input view with gradient masking on invisible regions.
        This preserves pretrained prior for invisible areas instead of corrupting with pseudo-GT.
        """
        self.finetune_visible_only(
            batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters,
            use_lora=True,
            gradient_mask_invisible=gradient_mask_invisible,
            pretrain_regularization=pretrain_regularization
        )

    def run_wild_inversion_stage2(self, batch=None, pseudo_batch=None, scaler=None, iteration=None, 
                                   writer=None, pbar=None, total_iters=None, color_consistency_weight=1.0):
        """
        Inversion stage 2: optimize color params with pseudo-view color consistency.
        """
        self.finetune_wild_inversion_stage2(
            batch=batch, pseudo_batch=pseudo_batch, scaler=scaler, iter=iteration, 
            writer=writer, pbar=pbar, total_iters=total_iters, 
            color_consistency_weight=color_consistency_weight
        )


    def run_interhand(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_interhand(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)


    def run_t2a(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_t2a(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)
        
    def run_nail(self, batch=None, scaler=None, iteration=None, writer=None, pbar=None, total_iters=None):
        # self.infer_edit(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar)
        self.finetune_nail(batch=batch, scaler=scaler, iter=iteration, writer=writer, pbar=pbar, total_iters=total_iters)