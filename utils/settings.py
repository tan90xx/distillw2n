import os
import time
import wandb
import argparse
from typing import Dict, List, Tuple, Union
from utils.utils_env import AttrDict

def gen_wandb_name(args, config) -> str:
    """
    Generate a descriptive name for a Weights & Biases (wandb) run.

    Combines the model type, a dash-joined list of training instruments,
    and the current date into a single string identifier.

    Args:
        args: Parsed arguments namespace containing at least `model_type`.
        config: Configuration object/dict with a `training.instruments` field.

    Returns:
        str: Formatted run name in the form
            "<model_type>_[<instrument1>-<instrument2>-...]_<YYYY-MM-DD>".
    """
    dataset = os.path.basename(args.input_wavs_dir)
    model_name = os.path.splitext(os.path.basename(args.config))[0]
    time_str = time.strftime("%Y-%m-%d")
    name = f"{model_name}-{dataset}-{time_str}"
    return name
    

def wandb_init(
    args: argparse.Namespace, config: AttrDict, batch_size: int
) -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.

    Depending on the provided arguments, sets up wandb in one of three modes:
    - Offline mode when `args.wandb_offline` is True.
    - Disabled mode when no valid `wandb_key` is provided.
    - Online mode with authentication using `args.wandb_key`.

    Args:
        args (argparse.Namespace): Parsed arguments containing wandb options
            (`wandb_offline`, `wandb_key`, `device_ids`).
        config (Dict): Experiment configuration dictionary to log.
        batch_size (int): Training batch size to include in the run configuration.

    Returns:
        None
    """

    if args.wandb_offline:
        wandb.init(
            mode="offline",
            project=args.wandb_project,
            name=gen_wandb_name(args, config),
            config={
                "config": config,
                "args": args,
                "batch_size": batch_size,
            },
        )
    elif args.wandb_key is None or args.wandb_key.strip() == "":
        wandb.init(mode="disabled")
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project,
            name=gen_wandb_name(args, config),
            config={
                "config": config,
                "args": args,
                "batch_size": batch_size,
            },
        )