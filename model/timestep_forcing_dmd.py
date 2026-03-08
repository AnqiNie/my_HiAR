"""
Timestep Forcing DMD Model (with optional Progressive Distillation)

Inherits from DMD and overrides the _initialize_inference_pipeline method,
replacing SelfForcingTrainingPipeline with TimestepForcingTrainingPipeline.

Progressive Distillation (PD) features:
- Uses PD loss (velocity regression / Euler endpoint) for the first pd_num_steps denoising steps
- Subsequent steps continue to use DMD loss
- PD and DMD steps do not overlap
- Supports two computation modes: single_forward (all frames in one forward pass) and rollout (block-by-block KV-cache inference)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from pipeline import TimestepForcingTrainingPipeline
from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import FlowMatchScheduler


class TimestepForcingDMD(DMD):
    """
    DMD with Timestep-First rollout and optional Progressive Distillation.

    Differences from the original DMD:
    1. Uses TimestepForcingTrainingPipeline for Timestep-First rollout
    2. Optional Progressive Distillation loss to constrain the first few denoising steps
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        
        # ============================================================
        # Progressive Distillation configuration
        # ============================================================
        self.pd_enabled = getattr(args, 'pd_enabled', False)
        self.pd_num_steps = getattr(args, 'pd_num_steps', 0)
        self.pd_loss_weight = getattr(args, 'pd_loss_weight', 1.0)
        self.pd_loss_type = getattr(args, 'pd_loss_type', 'euler_endpoint')
        
        # PD x0 loss target mode (only effective when pd_loss_type='x0'):
        # - 'pred_x0': linearly extrapolate from two adjacent teacher ODE save points to sigma=0 to get the teacher's pred_x0 (default)
        # - 'gt_x0': directly use the ground truth x0 from the teacher ODE's final denoised output (i.e., ode_latent[:, -1])
        self.pd_x0_target_mode = getattr(args, 'pd_x0_target_mode', 'pred_x0')
        assert self.pd_x0_target_mode in ('pred_x0', 'gt_x0'), \
            f"pd_x0_target_mode must be 'pred_x0' or 'gt_x0', got: {self.pd_x0_target_mode}"
        
        # PD computation mode: single_forward (all frames in one forward pass) or rollout (block-by-block KV-cache inference)
        self.pd_compute_mode = getattr(args, 'pd_compute_mode', 'single_forward')
        assert self.pd_compute_mode in ('single_forward', 'rollout'), \
            f"pd_compute_mode must be 'single_forward' or 'rollout', got: {self.pd_compute_mode}"
        
        # Dedicated PD rollout pipeline (lazily initialized)
        self.pd_rollout_pipeline: Optional[TimestepForcingTrainingPipeline] = None
        
        if self.pd_enabled and self.pd_num_steps > 0:
            # Precompute the (input_index, target_index) mapping in the teacher trajectory for each PD step
            # The teacher ODE trajectory saves save_indices=[0,12,24,36,-1], a total of 5 points
            # denoising_step_list=[1000,750,500,250], a total of 4 steps
            # step 0: 1000->750 corresponds to teacher trajectory index 0->1 (x_T -> x_12)
            # step 1: 750->500 corresponds to teacher trajectory index 1->2 (x_12 -> x_24)
            # step 2: 500->250 corresponds to teacher trajectory index 2->3 (x_24 -> x_36)
            # step 3: 250->0   corresponds to teacher trajectory index 3->4 (x_36 -> x_0)
            num_denoising_steps = len(self.denoising_step_list)
            assert self.pd_num_steps <= num_denoising_steps, \
                f"pd_num_steps ({self.pd_num_steps}) cannot exceed the number of denoising steps ({num_denoising_steps})"
            
            # ============================================================
            # Initialize the teacher's scheduler to obtain precise sigma values
            # Teacher sampling uses shift=8.0, 48 steps; save_indices=[0,12,24,36,-1]
            # These parameters must be consistent with generate_ode_pairs.py
            # ============================================================
            pd_teacher_timestep_shift = getattr(args, 'pd_teacher_timestep_shift', 8.0)
            pd_teacher_num_inference_steps = getattr(args, 'pd_teacher_num_inference_steps', 48)
            pd_teacher_save_indices = getattr(args, 'pd_teacher_save_indices', [0, 12, 24, 36, -1])
            
            teacher_scheduler = FlowMatchScheduler(
                shift=pd_teacher_timestep_shift, sigma_min=0.0, extra_one_step=True
            )
            teacher_scheduler.set_timesteps(num_inference_steps=pd_teacher_num_inference_steps)
            
            # Precompute the teacher sigma corresponding to each save point
            # sigma is the actual value encountered during teacher sampling
            self.pd_teacher_sigmas = []
            for si in pd_teacher_save_indices:
                if si == -1:
                    self.pd_teacher_sigmas.append(0.0)
                else:
                    self.pd_teacher_sigmas.append(teacher_scheduler.sigmas[si].item())
            
            # PD segment mapping: pd_step_index -> (teacher_input_idx, teacher_target_idx, input_timestep, target_timestep)
            self.pd_segments = []
            for i in range(self.pd_num_steps):
                input_timestep = self.denoising_step_list[i]
                # target_timestep: the timestep of the next step; for the last step, the target is 0
                if i + 1 < num_denoising_steps:
                    target_timestep = self.denoising_step_list[i + 1]
                else:
                    target_timestep = torch.tensor(0, dtype=torch.long)
                self.pd_segments.append({
                    'teacher_input_idx': i,       # Input index in the teacher trajectory
                    'teacher_target_idx': i + 1,  # Target index in the teacher trajectory
                    'input_timestep': input_timestep,
                    'target_timestep': target_timestep,
                    # Precise sigma values of the teacher at these two save points
                    'teacher_sigma_input': self.pd_teacher_sigmas[i],
                    'teacher_sigma_target': self.pd_teacher_sigmas[i + 1],
                })

    def _initialize_inference_pipeline(self):
        """
        Replace SelfForcingTrainingPipeline with TimestepForcingTrainingPipeline.

        Key point: min_exit_step is automatically set to pd_num_steps to ensure PD and DMD do not overlap.
        """
        # If PD is enabled, DMD's min_exit_step is automatically set to pd_num_steps
        # Modification: still follow min_exit_step
        min_exit_step = getattr(self.args, 'min_exit_step', 0)
        
        self.inference_pipeline = TimestepForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_rollout_frames,
            num_gradient_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            use_ode_trajectory=getattr(self.args, 'use_ode_trajectory', False),
            always_clean_context=getattr(self.args, 'always_clean_context', False),
            min_exit_step=min_exit_step,
            denoising_order=getattr(self.args, 'denoising_order', 'timestep_first'),
        )

    # ================================================================
    # PD Rollout related methods
    # ================================================================

    def _initialize_pd_rollout_pipeline(self, input_timestep: torch.Tensor):
        """
        Lazily initialize the dedicated PD rollout TimestepForcingTrainingPipeline.

        Key differences from DMD's inference_pipeline:
        - denoising_step_list only contains the current PD step's input_timestep (single step)
        - last_step_only=True, ensuring exit at the only step
        - min_exit_step=0, since there is only one step, it must exit at step 0
        - Other parameters (num_frame_per_block, use_ode_trajectory, always_clean_context, etc.)
          remain consistent with the DMD pipeline to ensure aligned inference behavior

        Args:
            input_timestep: The input timestep of the current PD step (e.g., 1000, 750, etc.)
        """
        self.pd_rollout_pipeline = TimestepForcingTrainingPipeline(
            denoising_step_list=input_timestep.unsqueeze(0),  # Keep the original dtype after warp, avoid float->long truncation
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=False,  # PD data has no independent first frame
            same_step_across_blocks=True,   # Must be True for Timestep-First mode
            last_step_only=True,            # Single-step pipeline, must exit at the last (only) step
            num_max_frames=self.num_training_frames,  # PD data frame count = training frame count, no need for a larger KV cache
            num_gradient_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            use_ode_trajectory=getattr(self.args, 'use_ode_trajectory', False),
            always_clean_context=getattr(self.args, 'always_clean_context', False),
            min_exit_step=0,                # Single-step pipeline, do not skip any step
        )

    def _pd_rollout(
        self,
        teacher_input: torch.Tensor,
        input_timestep: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Execute block-by-block rollout starting from a teacher save point.

        Core workflow:
        1. Create a pipeline with denoising_step_list=[input_timestep] (single step)
        2. Pass teacher_input as the noise parameter to inference_with_trajectory
        3. The pipeline executes the only step: block-by-block KV-cache-based forward, then exits
        4. Return the denoised output (x0_pred) of all blocks

        Differences from DMD rollout:
        - DMD rollout starts from pure noise and undergoes multi-step denoising
        - PD rollout starts from a teacher save point and performs only one denoising step
        - Both share the same block-by-block KV-cache inference logic

        Args:
            teacher_input: A save point from the teacher trajectory [B, F, C, H, W]
            input_timestep: The timestep of the current PD step (scalar tensor, e.g., 1000)
            conditional_dict: Conditioning information dict

        Returns:
            denoised_output: Denoised latent from the rollout [B, F, C, H, W]
        """
        # Lazily initialize or update the pipeline's denoising_step_list
        # Each call must ensure denoising_step_list matches the current input_timestep
        if self.pd_rollout_pipeline is None:
            self._initialize_pd_rollout_pipeline(input_timestep)
        else:
            # Update denoising_step_list (avoid recreating the pipeline object)
            self.pd_rollout_pipeline.denoising_step_list = input_timestep.unsqueeze(0)  # Keep original dtype
        
        # Execute rollout
        # teacher_input is passed as the noise parameter; the pipeline treats it as the noisy latent at the current timestep
        # initial_latent=None: PD data does not need I2V conditioning
        output, _, _ = self.pd_rollout_pipeline.inference_with_trajectory(
            noise=teacher_input.to(self.dtype),
            initial_latent=None,
            **conditional_dict,
        )
        
        # output shape: [B, F, C, H, W] (since initial_latent=None, output frame count == input frame count)
        return output

    # ================================================================
    # Progressive Distillation Loss
    # ================================================================

    def progressive_distillation_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Progressive Distillation loss.

        Supports two computation modes:
        - single_forward: All frames in one forward pass (blockwise causal attention)
        - rollout: Block-by-block KV-cache-based inference (aligned with inference behavior)

        For each step in pd_num_steps, adjacent point pairs are taken from the teacher trajectory,
        and the student is trained to predict results that approximate the teacher's next point
        using the teacher's input point as input.

        Args:
            ode_latent: Teacher ODE trajectory [B, num_saves, F, C, H, W]
                        Arranged by save_indices, from noisy to clean
            conditional_dict: Conditioning information dict

        Returns:
            pd_loss: PD loss scalar
            pd_log_dict: Logging information
        """
        if not self.pd_enabled or self.pd_num_steps == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        # Randomly select one PD step for training (similar to ODE regression strategy, reduces memory usage)
        pd_step_idx = torch.randint(0, self.pd_num_steps, (1,), device=self.device)
        # Broadcast to synchronize, ensuring all ranks use the same step
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pd_step_idx, src=0)
        pd_step_idx = pd_step_idx.item()
        
        segment = self.pd_segments[pd_step_idx]
        
        # Extract input and target from the teacher trajectory
        # ode_latent: [B, num_saves, F, C, H, W]
        teacher_input = ode_latent[:, segment['teacher_input_idx']]    # [B, F, C, H, W]
        teacher_target = ode_latent[:, segment['teacher_target_idx']]  # [B, F, C, H, W]
        
        input_timestep = segment['input_timestep']
        target_timestep = segment['target_timestep']
        
        # Get student prediction based on computation mode
        if self.pd_compute_mode == 'single_forward':
            flow_pred, x0_pred = self._pd_single_forward(
                teacher_input, input_timestep, conditional_dict)
        elif self.pd_compute_mode == 'rollout':
            flow_pred, x0_pred = self._pd_rollout_forward(
                teacher_input, input_timestep, conditional_dict)
        else:
            raise ValueError(f"Unknown pd_compute_mode: {self.pd_compute_mode}")
        
        # Get GT x0 (teacher ODE final denoised output, i.e., the last save point in the trajectory)
        gt_x0 = ode_latent[:, -1]  # [B, F, C, H, W]
        
        # Compute loss
        pd_loss, log_dict = self._compute_pd_loss(
            flow_pred=flow_pred,
            x0_pred=x0_pred,
            teacher_input=teacher_input,
            teacher_target=teacher_target,
            segment=segment,
            gt_x0=gt_x0,
        )
        
        log_dict.update({
            'pd_loss': pd_loss.detach(),
            'pd_step_idx': torch.tensor(pd_step_idx, device=self.device).float(),
            'pd_compute_mode': torch.tensor(
                0.0 if self.pd_compute_mode == 'single_forward' else 1.0,
                device=self.device),
            'pd_input_timestep': torch.tensor(input_timestep, device=self.device).float() 
                if isinstance(input_timestep, (int, float)) 
                else input_timestep.float(),
        })
        
        return pd_loss, log_dict

    def _pd_single_forward(
        self,
        teacher_input: torch.Tensor,
        input_timestep,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PD single_forward mode: All frames in one forward pass (blockwise causal attention).

        This is the original computation method where all frames are computed in a single
        forward pass through a blockwise causal mask.

        Args:
            teacher_input: [B, F, C, H, W]
            input_timestep: The timestep of the current step (scalar or tensor)
            conditional_dict: Conditioning information

        Returns:
            flow_pred: [B, F, C, H, W]
            x0_pred: [B, F, C, H, W]
        """
        batch_size, num_frames = teacher_input.shape[:2]
        
        timestep = torch.ones(
            [batch_size, num_frames], device=self.device, dtype=torch.long
        ) * input_timestep
        
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=teacher_input.to(self.dtype),
            conditional_dict=conditional_dict,
            timestep=timestep
        )
        
        return flow_pred, x0_pred

    def _pd_rollout_forward(
        self,
        teacher_input: torch.Tensor,
        input_timestep,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PD rollout mode: Block-by-block KV-cache-based inference.

        Fully aligned with inference behavior: blocks are input sequentially, using KV cache
        to carry historical information, where earlier blocks' information flows to later blocks
        via KV cache.

        Note: The rollout returns denoised output (x0_pred); flow_pred needs to be derived
        from x0_pred (used for velocity/euler_endpoint loss types).

        Args:
            teacher_input: [B, F, C, H, W]
            input_timestep: The timestep of the current step (scalar or tensor)
            conditional_dict: Conditioning information

        Returns:
            flow_pred: [B, F, C, H, W] (derived from x0_pred)
            x0_pred: [B, F, C, H, W] (denoised output from rollout)
        """
        batch_size, num_frames = teacher_input.shape[:2]
        
        # Execute rollout to get denoised output
        x0_pred = self._pd_rollout(
            teacher_input=teacher_input,
            input_timestep=input_timestep,
            conditional_dict=conditional_dict,
        )
        
        # Derive flow_pred from x0_pred (used for velocity/euler_endpoint loss types)
        timestep = torch.ones(
            [batch_size, num_frames], device=self.device, dtype=torch.long
        ) * input_timestep
        
        flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=x0_pred.flatten(0, 1),
            xt=teacher_input.to(self.dtype).flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        
        return flow_pred, x0_pred

    def _compute_pd_loss(
        self,
        flow_pred: torch.Tensor,
        x0_pred: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_target: torch.Tensor,
        segment: dict,
        gt_x0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute PD loss based on pd_loss_type.

        This method is independent of the computation mode (single_forward/rollout)
        and only depends on the student's predictions and the teacher's targets.

        Args:
            flow_pred: Student's flow prediction [B, F, C, H, W]
            x0_pred: Student's x0 prediction [B, F, C, H, W]
            teacher_input: Input point from the teacher trajectory [B, F, C, H, W]
            teacher_target: Target point from the teacher trajectory [B, F, C, H, W]
            segment: PD segment information dict
            gt_x0: Ground truth x0 from teacher ODE final denoised output [B, F, C, H, W]
                   (only used when pd_x0_target_mode='gt_x0')

        Returns:
            pd_loss: Loss scalar
            log_dict: Logging information
        """
        batch_size, num_frames = teacher_input.shape[:2]
        input_timestep = segment['input_timestep']
        target_timestep = segment['target_timestep']
        log_dict = {}
        
        if self.pd_loss_type == 'euler_endpoint':
            # Euler step: x_next = x_t + flow_pred * (sigma_next - sigma_t)
            timestep = torch.ones(
                [batch_size, num_frames], device=self.device, dtype=torch.long
            ) * input_timestep
            
            euler_result = self.scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                teacher_input.to(self.dtype).flatten(0, 1),
                target_timestep=target_timestep
            ).unflatten(0, (batch_size, num_frames))
            
            pd_loss = F.mse_loss(euler_result.float(), teacher_target.float())
            log_dict['pd_euler_loss'] = pd_loss.detach()
            
        elif self.pd_loss_type == 'velocity':
            # Velocity regression: flow_pred ≈ (teacher_target - teacher_input) / (sigma_next - sigma_t)
            timestep = torch.ones(
                [batch_size, num_frames], device=self.device, dtype=torch.long
            ) * input_timestep
            
            self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
            
            t_flat = timestep.flatten(0, 1)
            timestep_id = torch.argmin(
                (self.scheduler.timesteps.unsqueeze(0) - t_flat.unsqueeze(1)).abs(), dim=1)
            sigma_t = self.scheduler.sigmas[timestep_id].reshape(-1, 1, 1, 1)
            
            if not isinstance(target_timestep, torch.Tensor):
                target_ts = torch.tensor([target_timestep], device=self.device)
            else:
                target_ts = target_timestep.to(self.device)
            target_id = torch.argmin(
                (self.scheduler.timesteps.unsqueeze(0) - target_ts.float().unsqueeze(1)).abs(), dim=1)
            sigma_next = self.scheduler.sigmas[target_id].reshape(-1, 1, 1, 1)
            
            # v_target = (x_next - x_t) / (sigma_next - sigma_t)
            delta_sigma = sigma_next - sigma_t
            target_velocity = (teacher_target.flatten(0, 1).double() - teacher_input.flatten(0, 1).double()) / delta_sigma.double()
            target_velocity = target_velocity.unflatten(0, (batch_size, num_frames)).float()
            
            pd_loss = F.mse_loss(flow_pred.float(), target_velocity)
            log_dict['pd_velocity_loss'] = pd_loss.detach()
            
        elif self.pd_loss_type == 'x0':
            # ============================================================
            # X0 Regression: student's x0_pred approximates teacher's x0 target
            #
            # Two target modes are supported (controlled by pd_x0_target_mode):
            #
            # 1. pred_x0 (default):
            #    Derive teacher's pred_x0 from two save points on the teacher ODE path:
            #      v_teacher = (x_{t'} - x_t) / (sigma_{t'} - sigma_t)    (chord flow)
            #      x0_teacher = x_t - sigma_t * v_teacher                   (extrapolate to sigma=0)
            #    Geometric meaning: In (sigma, x) space, two save points define a chord;
            #                       extending it to sigma=0 gives the teacher's estimate of x0.
            #
            # 2. gt_x0:
            #    Directly use the ground truth x0 from the teacher ODE's final denoised output
            #    (ode_latent[:, -1]).
            #    This is the final clean latent after the teacher's full 48-step ODE sampling,
            #    representing the teacher's final and most accurate estimate of x0.
            #    All PD steps share the same gt_x0 target, providing a more direct signal.
            # ============================================================
            
            if self.pd_x0_target_mode == 'gt_x0':
                # GT X0 mode: directly use teacher ODE final denoised output
                assert gt_x0 is not None, \
                    "pd_x0_target_mode='gt_x0' requires gt_x0 (ode_latent[:, -1]) to be provided"
                teacher_x0 = gt_x0.to(self.dtype)
                
                pd_loss = F.mse_loss(x0_pred.float(), teacher_x0.float())
                log_dict['pd_x0_loss'] = pd_loss.detach()
                log_dict['pd_x0_target_mode'] = torch.tensor(1.0, device=self.device)  # 1.0 = gt_x0
            else:
                # pred_x0 mode (default): linearly extrapolate from two save points
                sigma_t = segment['teacher_sigma_input']
                sigma_t_next = segment['teacher_sigma_target']
                delta_sigma = sigma_t_next - sigma_t  # Negative (from noisy to clean)
                
                # Compute teacher's pred_x0 in double precision to avoid numerical errors
                teacher_x0 = (
                    teacher_input.double() 
                    - sigma_t * (teacher_target.double() - teacher_input.double()) / delta_sigma
                ).to(self.dtype)
                
                pd_loss = F.mse_loss(x0_pred.float(), teacher_x0.float())
                log_dict['pd_x0_loss'] = pd_loss.detach()
                log_dict['pd_teacher_sigma_input'] = torch.tensor(sigma_t, device=self.device).float()
                log_dict['pd_teacher_sigma_target'] = torch.tensor(sigma_t_next, device=self.device).float()
                log_dict['pd_x0_target_mode'] = torch.tensor(0.0, device=self.device)  # 0.0 = pred_x0
            
        else:
            raise ValueError(f"Unknown pd_loss_type: {self.pd_loss_type}")
        
        return pd_loss, log_dict
