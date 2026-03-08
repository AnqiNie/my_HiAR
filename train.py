import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint. Use 'auto' to auto-detect latest checkpoint, "
                             "or specify the path to model.pt file")
    parser.add_argument("--min_exit_step", type=int, default=None,
                        help="Override min_exit_step in config. Controls the minimum exit step index "
                             "(i.e. randint low), e.g. 2 means skip step 0 and 1")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if args.resume:
        config.resume_from = args.resume

    if args.min_exit_step is not None:
        config.min_exit_step = args.min_exit_step

    trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
