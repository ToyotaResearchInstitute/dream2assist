"""
Download S3 artifacts for SageMaker training.

Requirements:
- TRI AWS account access (account ID: 401298207814)
- Access to TRI S3 buckets (s3://tri-hid-data-shared-autonomy/)
- AWS CLI configured with appropriate credentials

For public standalone Box2D usage, this script is not needed.
See README.md for Box2D installation and training instructions.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)


def load_yaml_file(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_trailing_slash(s: str) -> str:
    return s if s.endswith("/") else s + "/"


def list_of_dir_from_s3(s3_base_src: str) -> List:
    # Run the AWS CLI command
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"{ensure_trailing_slash(s3_base_src)}"],
            capture_output=True,  # capture stdout
            text=True,  # decode bytes to str
            check=True,
        )

        # Get stdout as a string
        output = result.stdout

        # Split output into lines
        lines = output.strip().split("\n")

        # Filter only dir lines (lines starting with "PRE")
        return [line.split()[1] for line in lines if line.strip().startswith("PRE")]
    except subprocess.CalledProcessError as e:
        logger.error(f"AWS CLI error (exit {e.returncode}): {e.stderr}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Error listing S3 directories from {s3_base_src}: {e}", exc_info=True
        )
        sys.exit(1)


def download_from_s3(base_s3_path: str, dir_name: str, destination: str):
    """Download from S3 to local path.

    Args:
        base_s3_path: S3 base path (e.g., 's3://bucket/path/to/models')
        dir_name: Directory name with trailing slash (e.g., 'model_name/')
        destination: Local destination base path (e.g., '/opt/ml/code/logs/')

    The directory will be downloaded to: {destination}/{dir_name}
    For example: '/opt/ml/code/logs/model_name/'
    """
    s3_uri = f"{ensure_trailing_slash(base_s3_path)}{dir_name}"
    dst_location = f"{ensure_trailing_slash(destination)}{dir_name}"

    if Path(dst_location).exists():
        logger.info(f"{s3_uri} has already been downloaded to {dst_location}")
        return

    s3_command = [
        "aws",
        "s3",
        "cp",
        "--only-show-errors",
        s3_uri,
        dst_location,
        "--recursive",
    ]

    try:
        subprocess.run(s3_command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded to {dst_location}")
        logger.info(f"Successfully downloaded {s3_uri}")
    except subprocess.CalledProcessError as e:
        logger.error(f"AWS CLI error (exit {e.returncode}): {e.stderr}", exc_info=True)
        print(f"Failed to download to {dst_location}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error downloading {s3_uri}: {e}", exc_info=True)
        print(f"Failed to download to {dst_location}")
        sys.exit(1)


def extract_run_args(runs_list: list) -> dict:
    """Extract logdir and configs from runs arguments list"""
    result = []

    for run in runs_list:
        run_argument = {"logdir": None, "configs": None}
        if isinstance(run, dict) and "arguments" in run:
            args = run["arguments"]
            for i, arg in enumerate(args):
                if arg == "--logdir":
                    run_argument["logdir"] = args[i + 1]
                elif arg == "--configs":
                    run_argument["configs"] = args[i + 1]
            run_name = run["name"]
            if not run_argument["logdir"]:
                logger.error(
                    f"logdir argument not found in sweep.yaml for run {run_name}"
                )
                sys.exit(1)
            if not run_argument["configs"]:
                logger.error(
                    f"configs argument not found in sweep.yaml for run {run_name}"
                )
                sys.exit(1)

        result.append(run_argument)

    return result


def main():
    """Download S3 artifacts for SageMaker training.

    This script downloads model checkpoints and logs from S3 based on the config.

    Path Handling:
    ---------------
    1. Model checkpoints are downloaded from S3 to /opt/ml/code/logs/
       Example: s3://.../epic_model/foo/ → /opt/ml/code/logs/foo/

    2. Config paths in configs.yaml should be specified relative to /opt/ml/code/
       Example: 'logs/foo' will resolve to /opt/ml/code/logs/foo/ in training code

    3. The script extracts base directory names from config paths for S3 matching
       Example: 'logs/epic_human_left/' → matches 'epic_human_left/' in S3

    4. Training code uses util.make_absolute_path() to construct full paths:
       On SageMaker: make_absolute_path('logs/foo') → /opt/ml/code/logs/foo
    """
    # Load sweep.yaml
    sweep_yaml_path = Path("/opt/ml/code/.hail_launch/sweep.yaml")
    if not sweep_yaml_path.exists():
        logger.error(f"{sweep_yaml_path} does not exist.")
        sys.exit(1)
    sweep_yaml_data = load_yaml_file(sweep_yaml_path)
    run_args = extract_run_args(sweep_yaml_data.get("runs", []))

    # S3 base paths
    s3_epic_model_base = "s3://tri-hid-data-shared-autonomy/sdm_demo/epic_model"
    s3_logs_base = "s3://tri-hid-data-shared-autonomy/sdm_demo/epic_logs"

    # Download destination - all models are downloaded here
    download_dir = Path("/opt/ml/code/logs/")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Load config.yaml to get human_ego_agent_paths
    config_yaml_path = (Path(__file__).parent / "dream2assist" / "dream2assist" / "configs.yaml").resolve()
    if not config_yaml_path.exists():
        logger.error(f"{config_yaml_path} does not exist.")
        sys.exit(1)
    config_yaml_data = load_yaml_file(config_yaml_path)

    for run_arg in run_args:
        logdir = run_arg["logdir"]
        config_key = run_arg["configs"]

        # Download human_ego_agent_paths
        # Note: Paths in configs.yaml should be relative to /opt/ml/code/ on SageMaker
        # Example: 'logs/my_model' will be downloaded from s3://.../my_model/ to /opt/ml/code/logs/my_model/
        human_ego_agent_paths = config_yaml_data[config_key].get(
            "human_ego_agent_paths", []
        )
        print(f"Human ego agent paths for config {config_key}: {human_ego_agent_paths}")

        # Extract just the directory names with trailing slashes for S3 comparison
        # This handles both relative paths (e.g., 'logs/foo') and absolute paths
        # by extracting the base directory name (e.g., 'foo') and adding trailing slash
        human_ego_dir_names = []
        for path in human_ego_agent_paths:
            # Handle both absolute and relative paths, and ensure trailing slash
            dir_name = Path(path).name
            if not dir_name.endswith('/'):
                dir_name += '/'
            human_ego_dir_names.append(dir_name)
        print(f"Extracted directory names for S3 matching: {human_ego_dir_names}")

        s3_epic_model_base_dir_list = list_of_dir_from_s3(s3_epic_model_base)
        for dir_name in s3_epic_model_base_dir_list:
            print(f"Checking directory {dir_name} in S3 for config {config_key}...")
            if dir_name in human_ego_dir_names:
                print(f"Found matching directory {dir_name} in S3 for config {config_key}, downloading...")
                download_from_s3(s3_epic_model_base, str(dir_name), str(download_dir))

        # Download logdir checkpoint if it exists
        logdir_name = Path(logdir).name
        s3_logs_base_dir_list = list_of_dir_from_s3(s3_logs_base)
        logs_dir_path = [
            dir_name for dir_name in s3_logs_base_dir_list if logdir_name in dir_name
        ]
        for dir_name in logs_dir_path:
            print(f"Found matching logdir {dir_name} in S3 for config {config_key}, downloading...")
            download_from_s3(s3_logs_base, dir_name, str(download_dir))


if __name__ == "__main__":
    main()
