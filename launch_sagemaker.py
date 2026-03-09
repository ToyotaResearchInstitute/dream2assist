#!/usr/bin/env python3
"""
Launch a single training sweep on SageMaker for the Dream2Assist.

Requirements:
- TRI AWS account access (account ID: 401298207814)
- Access to TRI S3 buckets (s3://tri-hid-data-shared-autonomy/)
- Access to TRI ECR images
- hail_launch package from TRI's sdm_ws repository

For public standalone Box2D usage, this script is not needed.
See README.md for Box2D installation and training instructions.

---

This script defines a set of training runs and submits them to AWS SageMaker
using a predefined training image and configuration. Modify the `runs` list or
Sweep configuration to customize experiments.

Usage:
python launch_sagemaker.py

"""

import itertools
import math
import os
from datetime import datetime
from pathlib import Path

from hail_launch.launch_info import Run, RunConfigSagemaker, Sweep
from hail_launch.launch_sweep import launch_sagemaker_sweep
from hail_launch.util.sweep_util import generate_runs

# Find the path to parent directory containing .gitignore
current_dir = Path(".").resolve()
current_file = Path(__file__).resolve()
parent_with_gitignore = current_file
while parent_with_gitignore.parent != parent_with_gitignore:  # while not at root
    if (parent_with_gitignore / ".gitignore").exists():
        break
    parent_with_gitignore = parent_with_gitignore.parent
repo_root = parent_with_gitignore


num_scripts = 6
# cuda_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
cuda_ids = [0, 1, 2, 0, 1, 2]
assert len(cuda_ids) == num_scripts
wandb_run_group = "model_baseline-4.3-hidden-dim64_256-residual"
date_str = datetime.now().strftime("%Y%m%d-%H")


def main():
    # Start a single run
    runs = [
        Run('dream2assist_sagemaker', ['--configs', 'epic_ai', '--logdir', 'logs/epic_ai_leftrightfast_adorandomization_1']),
    ]

    sagemaker_run_config = RunConfigSagemaker(
        training_image='401298207814.dkr.ecr.us-east-1.amazonaws.com/sagemaker-training:sagemaker-carla-pytorch2.5.1-gpu-py311-cu124-ubuntu22.04',
        max_training_time=7 * 24,  # 7 days
    )
    sweep = Sweep(
        "dream2assist-test-sweep",
        runs,
        training_script="sdm_ws/src/dream2assist/dream2assist/dream2assist/train.py",
        training_cmd="python3 -u ",
        setup_script="src/dream2assist/setup_sdm.sh",
        sagemaker_run_config=sagemaker_run_config,

        # post_training_script="post_sdm.sh",
    )
    launch_sagemaker_sweep(sweep)


if __name__ == "__main__":
    main()
