import gc
import logging
import json
import datetime
import sys

from utils.dataset import ShardingLMDBDataset, PDPairDataset, cycle
from utils.dataset import TextDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD, TimestepForcingDMD
import torch
import wandb
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video
from einops import rearrange
from pipeline import CausalInferencePipeline


class TeeLogger:
    """Tee stdout to both the console and a log file for persistent logging.

    Usage: sys.stdout = TeeLogger(log_file_path)
    After this, all print() output will be written to both the console and the file.
    """
    def __init__(self, log_path, mode='a'):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, mode, buffering=1)  # line-buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def fileno(self):
        return self.terminal.fileno()


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Initialize log file (before TensorBoard, to start recording as early as possible)
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            log_file_path = os.path.join(self.output_path, "train.log")
            sys.stdout = TeeLogger(log_file_path, mode='a')
            sys.stderr = TeeLogger(os.path.join(self.output_path, "train_stderr.log"), mode='a')
            print(f"\n{'='*60}")
            print(f"Log session started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"All stdout -> {log_file_path}")
            print(f"{'='*60}")

        # Initialize TensorBoard
        if self.is_main_process:
            tb_log_dir = os.path.join(config.logdir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard log directory: {tb_log_dir}")
        else:
            self.tb_writer = None

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "timestep_forcing_dmd":
            self.model = TimestepForcingDMD(config, device=self.device)
        else:
            raise ValueError(f"Invalid distribution_loss: {config.distribution_loss}")

        # Count model parameters
        if self.is_main_process:
            def _count_params(m):
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                return total, trainable
            gen_total, gen_train = _count_params(self.model.generator)
            fs_total, fs_train = _count_params(self.model.fake_score)
            rs_total, _ = _count_params(self.model.real_score)
            print(f"Model Parameter Summary:")
            print(f"  Generator : {gen_total/1e6:.1f}M total, {gen_train/1e6:.1f}M trainable")
            print(f"  Fake Score: {fs_total/1e6:.1f}M total, {fs_train/1e6:.1f}M trainable")
            print(f"  Real Score: {rs_total/1e6:.1f}M total (frozen)")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=0 if not config.load_raw_video else 8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 3b: Initialize PD dataloader (if Progressive Distillation is enabled)
        self.pd_enabled = getattr(config, 'pd_enabled', False)
        self.pd_dataloader = None
        if self.pd_enabled:
            pd_data_path = getattr(config, 'pd_data_path', '')
            if pd_data_path and os.path.isdir(pd_data_path):
                pd_dataset = PDPairDataset(pd_data_path)
                pd_sampler = torch.utils.data.distributed.DistributedSampler(
                    pd_dataset, shuffle=True, drop_last=True)
                pd_dataloader = torch.utils.data.DataLoader(
                    pd_dataset,
                    batch_size=config.batch_size,
                    sampler=pd_sampler,
                    num_workers=0)
                self.pd_dataloader = cycle(pd_dataloader)
                if dist.get_rank() == 0:
                    print(f"PD DATASET SIZE {len(pd_dataset)} (from {pd_data_path})")
                    print(f"PD loss weight: {getattr(config, 'pd_loss_weight', 1.0)}")
                    print(f"PD num steps: {getattr(config, 'pd_num_steps', 0)}")
                    print(f"PD loss type: {getattr(config, 'pd_loss_type', 'euler_endpoint')}")
            else:
                if dist.get_rank() == 0:
                    print(f"WARNING: pd_enabled=true but pd_data_path='{pd_data_path}' is not a valid directory. PD disabled.")
                self.pd_enabled = False

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        
        # ============================================================
        # Weight loading logic (three-level priority):
        #   1. Resume checkpoint (fully restore training state)
        #   2. generator_ckpt (load only generator pretrained weights, start from step 0)
        #   3. Start from random initialization (not recommended, usually a config error)
        # ============================================================
        resume_path = getattr(config, "resume_from", None)
        checkpoint_loaded = False

        # Parse resume_path (supports "auto" mode)
        if resume_path == "auto":
            latest_link = os.path.join(self.output_path, "latest_checkpoint", "model.pt")
            if os.path.exists(latest_link):
                resume_path = latest_link
                if self.is_main_process:
                    print(f"Auto-detected latest checkpoint: {resume_path}")
            else:
                if self.is_main_process:
                    print(f"No latest checkpoint found at {latest_link}, will fallback to generator_ckpt")
                resume_path = None

        # Try to load full checkpoint
        if resume_path and os.path.exists(resume_path):
            if self.is_main_process:
                print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")

            # Load generator weights
            if "generator" in checkpoint:
                self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
                if self.is_main_process:
                    print("  - Loaded generator weights")
            
            # Load critic weights
            if "critic" in checkpoint:
                self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                if self.is_main_process:
                    print("  - Loaded critic weights")
            
            # Load training step
            if "step" in checkpoint:
                self.step = checkpoint["step"]
                if self.is_main_process:
                    print(f"  - Resumed from step {self.step}")
            else:
                # Parse step number from checkpoint file path
                import re
                match = re.search(r'checkpoint_model_(\d+)', resume_path)
                if match:
                    self.step = int(match.group(1))
                    if self.is_main_process:
                        print(f"  - Parsed step from checkpoint path: {self.step}")
                else:
                    if self.is_main_process:
                        print("  - Warning: Could not determine step from checkpoint, starting from step 0")
            
            # Load generator optimizer state (FSDP-aware)
            if "generator_optimizer" in checkpoint:
                try:
                    optim_state_to_load = FSDP.optim_state_dict_to_load(
                        self.model.generator,
                        self.generator_optimizer,
                        checkpoint["generator_optimizer"]
                    )
                    self.generator_optimizer.load_state_dict(optim_state_to_load)
                    if self.is_main_process:
                        print("  - Loaded generator optimizer state (FSDP full optim)")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  - Warning: Failed to load generator optimizer with FSDP method: {e}")
                        print(f"  - Skipping generator optimizer state (will use fresh optimizer)")
            
            # Load critic optimizer state (FSDP-aware)
            if "critic_optimizer" in checkpoint:
                try:
                    optim_state_to_load = FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        checkpoint["critic_optimizer"]
                    )
                    self.critic_optimizer.load_state_dict(optim_state_to_load)
                    if self.is_main_process:
                        print("  - Loaded critic optimizer state (FSDP full optim)")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  - Warning: Failed to load critic optimizer with FSDP method: {e}")
                        print(f"  - Skipping critic optimizer state (will use fresh optimizer)")
            
            # Load EMA state
            if "generator_ema" in checkpoint and self.step >= config.ema_start_step:
                if self.generator_ema is None:
                    self.generator_ema = EMA_FSDP(self.model.generator, decay=config.ema_weight)
                self.generator_ema.load_state_dict(checkpoint["generator_ema"])
                if self.is_main_process:
                    print("  - Loaded generator EMA state")

            if self.is_main_process:
                print(f"Successfully resumed from step {self.step}")
            checkpoint_loaded = True
            del checkpoint
            torch.cuda.empty_cache()

        # Fallback: load generator pretrained weights (not a full resume)
        if not checkpoint_loaded and getattr(config, "generator_ckpt", False):
            if self.is_main_process:
                print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )
            checkpoint_loaded = True
            if self.is_main_process:
                print(f"  - Generator weights loaded, training from step 0")
        
        if not checkpoint_loaded and self.is_main_process:
            print("WARNING: No checkpoint or generator_ckpt loaded! Training from random initialization.")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # Compute gradient accumulation steps
        # total_batch_size = batch_size * world_size * gradient_accumulation_steps
        effective_batch_per_step = config.batch_size * self.world_size
        if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 0:
            self.gradient_accumulation_steps = config.gradient_accumulation_steps
        elif hasattr(config, 'total_batch_size') and config.total_batch_size > effective_batch_per_step:
            self.gradient_accumulation_steps = config.total_batch_size // effective_batch_per_step
        else:
            self.gradient_accumulation_steps = 1
        
        if self.is_main_process:
            print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            print(f"Effective batch size: {config.batch_size * self.world_size * self.gradient_accumulation_steps}")

        # ============================================================
        # Save full config and training metadata to output directory
        # ============================================================
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            # Save merged full config (YAML format for readability)
            config_save_path = os.path.join(self.output_path, "config.yaml")
            OmegaConf.save(config, config_save_path)
            print(f"Full config saved to {config_save_path}")
            
            # Save training metadata (JSON format for programmatic access)
            train_meta = {
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_step": self.step,
                "world_size": self.world_size,
                "batch_size_per_gpu": config.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": config.batch_size * self.world_size * self.gradient_accumulation_steps,
                "distribution_loss": config.distribution_loss,
                "lr_generator": config.lr,
                "lr_critic": getattr(config, 'lr_critic', config.lr),
                "denoising_step_list": config.denoising_step_list,
                "num_training_frames": getattr(config, 'num_training_frames', 21),
                "num_rollout_frames": getattr(config, 'num_rollout_frames', 0),
                "num_frame_per_block": getattr(config, 'num_frame_per_block', 1),
                "mixed_precision": config.mixed_precision,
                "sharding_strategy": config.sharding_strategy,
                "resume_from": getattr(config, 'resume_from', None),
                "generator_ckpt": getattr(config, 'generator_ckpt', None),
                "pd_enabled": getattr(config, 'pd_enabled', False),
                "pd_num_steps": getattr(config, 'pd_num_steps', 0),
                "denoising_order": getattr(config, 'denoising_order', 'timestep_first'),
            }
            meta_save_path = os.path.join(self.output_path, "train_meta.json")
            with open(meta_save_path, "w") as f:
                json.dump(train_meta, f, indent=2, default=str)
            
            # Print training configuration summary
            print("\n" + "=" * 60)
            print("Training Configuration Summary")
            print("=" * 60)
            print(f"  Distribution Loss : {config.distribution_loss}")
            print(f"  Denoising Steps   : {config.denoising_step_list}")
            print(f"  Denoising Order   : {getattr(config, 'denoising_order', 'timestep_first')}")
            print(f"  World Size        : {self.world_size}")
            print(f"  Batch / GPU       : {config.batch_size}")
            print(f"  Grad Accum Steps  : {self.gradient_accumulation_steps}")
            print(f"  Effective Batch   : {config.batch_size * self.world_size * self.gradient_accumulation_steps}")
            print(f"  LR (generator)    : {config.lr}")
            print(f"  LR (critic)       : {getattr(config, 'lr_critic', config.lr)}")
            print(f"  EMA Weight        : {config.ema_weight} (start at step {config.ema_start_step})")
            print(f"  Mixed Precision   : {config.mixed_precision}")
            print(f"  Num Frames        : {getattr(config, 'num_training_frames', 21)}")
            _rf = getattr(config, 'num_rollout_frames', 0)
            _effective_rollout = _rf if _rf > 0 else getattr(config, 'num_training_frames', 21)
            print(f"  Rollout Frames    : {_effective_rollout}")
            print(f"  Frames / Block    : {getattr(config, 'num_frame_per_block', 1)}")
            print(f"  Log Interval      : {config.log_iters} steps")
            print(f"  Gen/Critic Ratio  : 1:{config.dfake_gen_update_ratio}")
            if getattr(config, 'pd_enabled', False):
                print(f"  PD Enabled        : True")
                print(f"  PD Num Steps      : {getattr(config, 'pd_num_steps', 0)}")
                print(f"  PD Loss Type      : {getattr(config, 'pd_loss_type', 'euler_endpoint')}")
                print(f"  PD Loss Weight    : {getattr(config, 'pd_loss_weight', 1.0)}")
                print(f"  PD Compute Mode   : {getattr(config, 'pd_compute_mode', 'single_forward')}")
            print(f"  Starting Step     : {self.step}")
            print(f"  Output Dir        : {self.output_path}")
            print("=" * 60 + "\n")

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        # Save optimizer state using FSDP-aware method (full non-sharded state)
        # This allows correct training resumption across different GPU counts
        generator_optim_state = FSDP.full_optim_state_dict(
            self.model.generator, self.generator_optimizer)
        critic_optim_state = FSDP.full_optim_state_dict(
            self.model.fake_score, self.critic_optimizer)

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
            "step": self.step,
            "generator_optimizer": generator_optim_state,
            "critic_optimizer": critic_optim_state,
        }
        
        if self.config.ema_start_step <= self.step and self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
            # Also save a latest checkpoint symlink for easy resume
            latest_link = os.path.join(self.output_path, "latest_checkpoint")
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            elif os.path.exists(latest_link):
                import shutil
                shutil.rmtree(latest_link)
            os.symlink(f"checkpoint_model_{self.step:06d}", latest_link)
            print(f"Latest checkpoint link updated: {latest_link}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # Progressive Distillation loss (if enabled)
            pd_loss = torch.tensor(0.0, device=self.device)
            pd_log_dict = {}
            if self.pd_enabled and self.pd_dataloader is not None and hasattr(self.model, 'progressive_distillation_loss'):
                pd_batch = next(self.pd_dataloader)
                pd_prompts = pd_batch["prompts"]
                pd_ode_latent = pd_batch["ode_latent"].to(device=self.device, dtype=self.dtype)
                
                # Encode PD prompts text embeddings
                with torch.no_grad():
                    pd_conditional_dict = self.model.text_encoder(
                        text_prompts=pd_prompts)
                
                pd_loss, pd_log_dict = self.model.progressive_distillation_loss(
                    ode_latent=pd_ode_latent,
                    conditional_dict=pd_conditional_dict,
                )
                pd_loss = pd_loss * getattr(self.config, 'pd_loss_weight', 1.0)

            # Combine DMD loss and PD loss
            total_loss = generator_loss + pd_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / self.gradient_accumulation_steps
            scaled_loss.backward()

            # Log the unscaled loss for monitoring
            generator_log_dict.update({"generator_loss": generator_loss})
            if self.pd_enabled:
                generator_log_dict.update({
                    "pd_loss": pd_loss.detach(),
                    "total_generator_loss": total_loss.detach(),
                })
                generator_log_dict.update(pd_log_dict)

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        # Scale loss for gradient accumulation
        scaled_loss = critic_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Log the unscaled loss for monitoring
        critic_log_dict.update({"critic_loss": critic_loss})

        return critic_log_dict

    def generate_video(self, prompts, image=None):
        """
        Generate videos for visualization. Builds an inference pipeline using the
        generator, text_encoder, and vae from training.
        Note: This function should be called within a torch.no_grad() context.

        Selects inference method based on distribution_loss type:
        - timestep_forcing_dmd: uses inference_hybrid (frame_first_steps=0, i.e. timestep-first)
        - others: uses the original inference (frame-first)

        Args:
            prompts: list of text prompts
            image: optional initial image (for I2V)

        Returns:
            video: numpy array, shape [B, T, H, W, C], value range [0, 255]
        """
        batch_size = len(prompts)
        
        # Build inference pipeline (temporary, using components from training)
        # Note: temporarily set generator to eval mode
        was_training = self.model.generator.training
        self.model.generator.eval()
        
        # Create a temporary inference pipeline
        pipeline = CausalInferencePipeline(
            args=self.config,
            device=self.device,
            generator=self.model.generator,
            text_encoder=self.model.text_encoder,
            vae=self.model.vae
        )
        
        # ====== Override pipeline / model parameters to support extra-long frame sampling ======
        # By setting local_attn_size, KV cache is allocated per local window (not global 32760),
        # enabling inference with arbitrary frame counts without modifying causal_inference.py.
        num_training_frames = getattr(self.config, 'num_training_frames', 21)
        sample_multiplier = 6  # generate N times the training frame count for visualization
        num_output_frames = num_training_frames * sample_multiplier  # e.g. 21 * 6 = 126

        # Ensure frame count satisfies the num_frame_per_block divisibility constraint
        num_frame_per_block = getattr(self.config, 'num_frame_per_block', 3)
        independent_first_frame = getattr(self.config, 'independent_first_frame', False)
        if independent_first_frame:
            remainder = (num_output_frames - 1) % num_frame_per_block
            if remainder != 0:
                num_output_frames += (num_frame_per_block - remainder)
        else:
            remainder = num_output_frames % num_frame_per_block
            if remainder != 0:
                num_output_frames += (num_frame_per_block - remainder)
        
        # Set local attention window (in frames), so KV cache = local_attn_size * frame_seq_length
        local_attn_window = num_training_frames  # local window size = training frame count (e.g. 21)

        # Save original values (generator is shared during training, must restore after inference)
        orig_pipeline_local_attn = pipeline.local_attn_size
        orig_block_attrs = []
        for block in pipeline.generator.module.model.blocks:
            orig_block_attrs.append({
                'local_attn_size': block.self_attn.local_attn_size,
                'max_attention_size': block.self_attn.max_attention_size,
            })
            block.self_attn.local_attn_size = local_attn_window
            block.self_attn.max_attention_size = local_attn_window * 1560  # frame_seq_length = 1560
        
        pipeline.local_attn_size = local_attn_window
        pipeline.num_frame_per_block = num_frame_per_block
        pipeline.use_ode_trajectory = False
        # Force pipeline to reallocate KV cache (since local_attn_size changed)
        pipeline.kv_cache1 = None
        
        if image is not None:
            # I2V mode
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=self.dtype)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=self.device, dtype=self.dtype)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_output_frames - 1, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        else:
            # T2V mode
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_output_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        
        # Select inference method based on distribution_loss type
        use_timestep_first = (self.config.distribution_loss == "timestep_forcing_dmd")
        
        if use_timestep_first:
            # Timestep Forcing training: use timestep-first inference
            # frame_first_steps=0 means all steps use timestep-first order
            video, _ = pipeline.inference_hybrid(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent,
                frame_first_steps=0,  # all timestep-first
            )
        else:
            # Original training: use frame-first inference
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent
            )
        
        # Clean up VAE cache
        pipeline.vae.model.clear_cache()
        
        # ====== Restore generator model's original parameters ======
        # Generator is shared during training, must restore local attention params
        for i, block in enumerate(pipeline.generator.module.model.blocks):
            block.self_attn.local_attn_size = orig_block_attrs[i]['local_attn_size']
            block.self_attn.max_attention_size = orig_block_attrs[i]['max_attention_size']
        
        # Clean up KV cache allocated during inference to free memory for training
        pipeline.kv_cache1 = None
        if hasattr(pipeline, 'crossattn_cache'):
            pipeline.crossattn_cache = None
        del pipeline
        torch.cuda.empty_cache()
        
        # Restore generator's training state
        if was_training:
            self.model.generator.train()
        
        # Convert to numpy format, value range [0, 255]
        # video shape: [B, T, C, H, W] -> [B, T, H, W, C]
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step
        train_start_time = time.time()
        
        if self.is_main_process:
            print(f"\nTraining started at step {start_step}")
        
        # Default sampling prompts
        default_sample_prompt_list = [
            "Two NBA legends, Michael Jordan and Kobe Bryant, competing intensely on a basketball court. Both players are mid-action, with Jordan leaping towards the hoop, showcasing his iconic hang time, and Kobe driving to the basket, displaying his agility and determination. They are both wearing their respective team uniforms from their prime years, Jordan in his Chicago Bulls jersey and Kobe in his Los Angeles Lakers attire. The court is vividly lit with spectators cheering loudly in the background. The focus is on the dynamic interaction between the two players, capturing the essence of competitive sportsmanship. Medium shot, emphasizing their physical engagement and the intensity of the moment.",
            "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it's tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.",
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
 
        ]

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            # Train the generator (with gradient accumulation)
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for accum_step in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    # Use no_sync context manager to skip gradient sync during accumulation, sync only on last step
                    if accum_step < self.gradient_accumulation_steps - 1:
                        with self.model.generator.no_sync():
                            extra = self.fwdbwd_one_step(batch, True)
                    else:
                        extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                # Perform clip_grad_norm after gradient accumulation is complete
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                generator_log_dict = merge_dict_list(extras_list)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic (with gradient accumulation)
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for accum_step in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                # Use no_sync context manager to skip gradient sync during accumulation, sync only on last step
                if accum_step < self.gradient_accumulation_steps - 1:
                    with self.model.fake_score.no_sync():
                        extra = self.fwdbwd_one_step(batch, False)
                else:
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            # Perform clip_grad_norm after gradient accumulation is complete
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict = merge_dict_list(extras_list)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                
                # Sample videos (all processes participate since FSDP models require collective ops)
                try:
                    if self.is_main_process:
                        sample_output_dir = os.path.join(self.output_path, "samples")
                        os.makedirs(sample_output_dir, exist_ok=True)
                    
                    # All processes execute generate_video (required by FSDP)
                    # Use no_grad to disable gradient computation
                    with torch.no_grad():
                        sample_video = self.generate_video(prompts=default_sample_prompt_list)
                    
                    # Only the main process saves videos and logs
                    if self.is_main_process:
                        # sample_video shape: [B, T, H, W, C], save each prompt's video
                        num_samples = sample_video.shape[0]
                        for idx in range(num_samples):
                            video_path = os.path.join(sample_output_dir, f"sample_step_{self.step:06d}_p{idx}.mp4")
                            write_video(video_path, torch.from_numpy(sample_video[idx]).to(torch.uint8), fps=16)
                        print(f"Saved {num_samples} sample videos to {sample_output_dir} (step {self.step})")
                        
                        # Log to TensorBoard: first frame of each prompt
                        if self.tb_writer is not None:
                            for idx in range(num_samples):
                                first_frame = sample_video[idx, 0]  # [H, W, C]
                                first_frame = torch.from_numpy(first_frame).permute(2, 0, 1) / 255.0  # [C, H, W], [0, 1]
                                self.tb_writer.add_image(f"samples/prompt_{idx}_first_frame", first_frame, self.step)
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    if self.is_main_process:
                        print(f"Warning: Failed to generate sample video at step {self.step}: {e}")
                        import traceback
                        traceback.print_exc()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    generator_loss_mean = generator_log_dict["generator_loss"].mean().item()
                    generator_grad_norm_mean = generator_log_dict["generator_grad_norm"].mean().item()
                    dmdtrain_gradient_norm_mean = generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                    
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_loss_mean,
                            "generator_grad_norm": generator_grad_norm_mean,
                            "dmdtrain_gradient_norm": dmdtrain_gradient_norm_mean
                        }
                    )
                    
                    # TensorBoard: log generator metrics
                    self.tb_writer.add_scalar("loss/generator", generator_loss_mean, self.step)
                    self.tb_writer.add_scalar("grad_norm/generator", generator_grad_norm_mean, self.step)
                    self.tb_writer.add_scalar("grad_norm/dmdtrain_gradient", dmdtrain_gradient_norm_mean, self.step)
                    
                    # TensorBoard: log PD metrics (if enabled)
                    if self.pd_enabled and "pd_loss" in generator_log_dict:
                        pd_loss_mean = generator_log_dict["pd_loss"].mean().item()
                        self.tb_writer.add_scalar("loss/pd", pd_loss_mean, self.step)
                        wandb_loss_dict["pd_loss"] = pd_loss_mean
                        if "total_generator_loss" in generator_log_dict:
                            total_gen_loss_mean = generator_log_dict["total_generator_loss"].mean().item()
                            self.tb_writer.add_scalar("loss/total_generator", total_gen_loss_mean, self.step)
                            wandb_loss_dict["total_generator_loss"] = total_gen_loss_mean
                    
                    # TensorBoard: log per-chunk generator loss (if multiple accumulation steps)
                    if self.gradient_accumulation_steps > 1:
                        for i, loss_val in enumerate(generator_log_dict["generator_loss"].tolist()):
                            self.tb_writer.add_scalar(f"loss_per_chunk/generator_chunk_{i}", loss_val, self.step)
                    
                    # TensorBoard: log other useful metrics from generator_log_dict
                    for key, value in generator_log_dict.items():
                        if key not in ["generator_loss", "generator_grad_norm", "dmdtrain_gradient_norm"]:
                            if isinstance(value, torch.Tensor):
                                self.tb_writer.add_scalar(f"generator/{key}", value.mean().item(), self.step)
                            elif isinstance(value, (int, float)):
                                self.tb_writer.add_scalar(f"generator/{key}", value, self.step)

                critic_loss_mean = critic_log_dict["critic_loss"].mean().item()
                critic_grad_norm_mean = critic_log_dict["critic_grad_norm"].mean().item()
                
                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_loss_mean,
                        "critic_grad_norm": critic_grad_norm_mean
                    }
                )
                
                # TensorBoard: log critic metrics
                self.tb_writer.add_scalar("loss/critic", critic_loss_mean, self.step)
                self.tb_writer.add_scalar("grad_norm/critic", critic_grad_norm_mean, self.step)
                
                # TensorBoard: log per-chunk critic loss (if multiple accumulation steps)
                if self.gradient_accumulation_steps > 1:
                    for i, loss_val in enumerate(critic_log_dict["critic_loss"].tolist()):
                        self.tb_writer.add_scalar(f"loss_per_chunk/critic_chunk_{i}", loss_val, self.step)
                
                # TensorBoard: log other useful metrics from critic_log_dict
                for key, value in critic_log_dict.items():
                    if key not in ["critic_loss", "critic_grad_norm"]:
                        if isinstance(value, torch.Tensor):
                            self.tb_writer.add_scalar(f"critic/{key}", value.mean().item(), self.step)
                        elif isinstance(value, (int, float)):
                            self.tb_writer.add_scalar(f"critic/{key}", value, self.step)

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                
                # Print training progress to console (print key info every step)
                current_time = time.time()
                elapsed = current_time - train_start_time
                steps_done = self.step - start_step
                avg_time = elapsed / steps_done if steps_done > 0 else 0
                
                log_parts = [f"Step {self.step}"]
                if TRAIN_GENERATOR:
                    log_parts.append(f"G_loss={generator_loss_mean:.4f}")
                    log_parts.append(f"G_grad={generator_grad_norm_mean:.2f}")
                    if self.pd_enabled and "pd_loss" in generator_log_dict:
                        log_parts.append(f"PD_loss={pd_loss_mean:.4f}")
                log_parts.append(f"C_loss={critic_loss_mean:.4f}")
                log_parts.append(f"C_grad={critic_grad_norm_mean:.2f}")
                log_parts.append(f"{avg_time:.1f}s/step")
                log_parts.append(f"elapsed={datetime.timedelta(seconds=int(elapsed))}")
                
                print(" | ".join(log_parts))

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    iter_time = current_time - self.previous_time
                    # TensorBoard: log per-iteration time
                    self.tb_writer.add_scalar("time/iteration", iter_time, self.step)
                    # TensorBoard: log learning rates
                    self.tb_writer.add_scalar("lr/generator", self.generator_optimizer.param_groups[0]['lr'], self.step)
                    self.tb_writer.add_scalar("lr/critic", self.critic_optimizer.param_groups[0]['lr'], self.step)
                    
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iter_time}, step=self.step)
                    self.previous_time = current_time
