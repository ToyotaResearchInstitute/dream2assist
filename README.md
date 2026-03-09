# dream2assist

A framework for training shared control policies where an AI agent learns to infer human intent and provide assistive interventions for autonomous driving scenarios.

## Quick Start (Box2D - Standalone Installation)

This is the recommended setup for public use. It requires only standard open-source dependencies and runs entirely on the Box2D physics simulator.

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:ToyotaResearchInstitute/dream2assist.git
   cd dream2assist
   ```

2. **Create a Python environment**

   Using micromamba (recommended):
   ```bash
   micromamba create -n dream2assist python=3.10.10
   micromamba activate dream2assist
   micromamba install swig  # Required for Box2D compilation
   ```

   Or using venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dream2assist**
   ```bash
   cd dream2assist
   pip install -e .
   pip install -r requirements.txt
   ```

4. **Install PyTorch**

   Install according to your hardware setup, as per the [pytorch.org](https://pytorch.org/) instructions.

   For CUDA (Linux/Windows with NVIDIA GPU):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CPU or Mac:
   ```bash
   pip install torch torchvision torchaudio
   ```

## Configs File Structure

The `configs.yaml` file is organized into logically grouped sections for clarity and consistency:

1. The `defaults` section that contains base parameters.

2. Sub-Configurations that inherit from `defaults` and override specific parameters. They are organized by platform / task:

**Box2D Configurations**:
- `box2d_racing_human_pass` - Human passing agent on racing track
- `box2d_town_human_left` - Left-biased human on town track
- `box2d_town_human_right` - Right-biased human on town track
- `box2d_racing_ai` - AI assistant on racing track
- `box2d_town_ai` - AI assistant on town track

**EPIC Configurations**:
- `epic_human_left` - Human agent that prefers passing on the left
- `epic_human_right` - Human agent that prefers passing on the right
- `epic_human_fast` - Human agent that prioritizes speed
- `epic_human_pass` - Human agent trained for general passing behavior
- `epic_human_cautious` - Cautious human agent that avoids collisions
- `epic_ai` - AI agent trained with EPIC dynamics and human models

**CARLA Configurations**:
- `carla_human_pass`, `carla_human_stay`
- `carla_ai` - AI agent in CARLA simulator

## Training Workflow (Box2D)

### Stage 1: Train Human Behaviors

First, train human agents with different passing preferences:

```bash
# Train left-passing human agent (town track)
python dream2assist/train.py --configs box2d_town_human_left --logdir logs/box2d_human_left_run1

# Train right-passing human agent (town track)
python dream2assist/train.py --configs box2d_town_human_right --logdir logs/box2d_human_right_run1
```

Or for the racing track:

```bash
# Train passing human agent (racing track)
python dream2assist/train.py --configs box2d_racing_human_pass --logdir logs/box2d_human_pass_run1
```

Training typically runs for 2M steps (configurable in `configs.yaml`).

### Stage 2: Train AI Agent with Human Models

After training human agents, train the AI agent that learns to infer human intent:

1. **Update the config** - Edit `dream2assist/configs.yaml` under the `box2d_town_ai` or `box2d_racing_ai` section:
   ```yaml
   box2d_town_ai:
     human_ego_agent_paths: [
       'logs/box2d_human_left_run1',
       'logs/box2d_human_right_run1'
     ]
     population_sample_bias: [0.7, 0.3]  # Sampling probability for each human model
     behavior_labels: ["left", "right"]  # Intent labels corresponding to human models
   ```

2. **Train the AI agent**:
   ```bash
   # For town track with left/right passing humans
   python dream2assist/train.py --configs box2d_town_ai --logdir logs/box2d_ai_run1

   # Or for racing track
   python dream2assist/train.py --configs box2d_racing_ai --logdir logs/box2d_ai_run1
   ```

The AI agent will learn to infer which passing behavior (left/right) the human prefers, provide minimal assistive interventions when needed, and respect human autonomy by acting only when necessary.

### Monitoring Training

**TensorBoard** (included):
```bash
tensorboard --logdir logs/
```

**Weights & Biases** (optional, requires account):
```bash
wandb login
# Training will automatically log to W&B
```

## Repository Structure

### Public Components (Box2D - Standalone)

These components work without any proprietary dependencies:

- **`dream2assist/`** - Core training code (Dreamer-based model-based RL)
- **`envs/box2d_*.py`** - Box2D racing environments
- **`assets/track*.csv`** - Track data for Box2D simulator
- **`configs.yaml`** - Configuration file (see `box2d_*` configs)
- **`requirements.txt`** - Public Python dependencies

### Optional Proprietary Components (TRI-Internal)

**These components require Toyota Research Institute internal access and are not needed for Box2D training:**

- **`envs/epic_*.py`** - EPIC Dynamics integration (requires `epic_workspace`)
- **`shared_decision_making/`** - ROS2-based shared decision making (requires TRI platforms)
- **`launch_sagemaker.py`** - AWS SageMaker launcher (requires TRI AWS access)
- **`download_s3_artifacts.py`** - S3 asset downloader (requires TRI AWS access)
- **`setup_sdm.sh`** - SDM setup script (requires TRI infrastructure)
- **`configs.yaml`** `epic_*` and `carla_*` configs - Proprietary environment configurations

## EPIC Dynamics Environment (Optional - Requires TRI Internal Access)

This is a more involved setup to allow training using UnifiedControl and UnifiedState via the dynamics models to interoperate with various TRI platforms (e.g. GRIP, Leia, etc.).

### Installation (EPIC Dynamics)

1. **Clone `sdm_ws`** (requires TRI internal access):
   ```bash
   git clone git@github.shared-services.aws.tri.global:tri-projects/sdm_ws.git
   ```

2. **Follow the `sdm_ws` setup** (instructions for the `sdm_demo` branch):
   ```bash
   cd ~/sdm_ws
   python -m venv .dream2assist-venv
   source .dream2assist-venv/bin/activate
   # Update the vcs and install sub-directory requirements
   git checkout sdm_demo
   vcs import src --input recipes/sdm_demo.yaml
   ./scripts/install_src_requirements.bash
   # Source ROS2 and colcon build
   source /opt/ros/humble/setup.bash
   python -m colcon build --symlink-install
   ```

3. **Install dream2assist**:
   ```bash
   cd ~/sdm_ws/src/dream2assist/dream2assist
   pip install -e .
   ```

---

## AWS SageMaker Training (Optional - Requires TRI Internal Access)

This section is for internal TRI use only and requires:
- Access to TRI S3 buckets (`s3://tri-hid-data-shared-autonomy/`)
- TRI SageMaker ECR images
- `hail_launch` package from TRI's `sdm_ws` repository

This script (`launch_sagemaker.py`) is used to launch training runs of Dream2Assist on AWS SageMaker. It defines a sweep configuration and submits one or more training jobs to SageMaker using the HAIL launcher framework.

### Requirements

1. **AWS credentials**: `aws configure` must be set with IAM permissions to run SageMaker training jobs.

2. **Docker image**: The ECR image URI used in the script must be accessible:
   ```
   401298207814.dkr.ecr.us-east-1.amazonaws.com/sagemaker-training:sagemaker-carla-pytorch2.5.1-gpu-py311-cu124-ubuntu22.04
   ```

3. **hail_launch library**:
   ```bash
   pip install sdm_ws/src/hail_launch/
   ```

4. **Clear build artifacts**: If you have colcon built locally, delete the `build/`, `install/`, and `logs/` directories and move your `.venv` directory outside of `sdm_ws`. These folders can cause upload issues and runtime errors.

    These folders can be quite large and cause the upload time to sagemaker to skyrocket. Having references to a local build in the sagemaker instance can cause colcon build to fail and can cause runtime errors. We recreate the venv and run colcon build on the sagemaker instance after we finish uploading.

5. **Run the SageMaker launch script from the correct directory**

    Navigate to the `sdm_ws` directory in your terminal and execute `launch_sagemaker.py` from there. The script captures your current working directory and uploads it to SageMaker for training. Running it from any other location will cause the training job to behave incorrectly.

    **Note:** MAKE SURE YOU HAVE DONE STEP 5 BEFORE RUNNING THE SCRIPT! Skipping step 5 has a likelihood of causing your run to fail.


### Notes

* Before running training for **Dream2Assist**, make sure you have the **latest assets** placed in:

  ```
  dream2assist/dream2assist/assets/
  ```

  Alternatively, you can download/update the assets from S3:

  ```
  s3://tri-hid-data-shared-autonomy/sdm_demo/epic_model/
  ```

* The **SDM_setup** script already handles copying the asset files into:

  ```
  /opt/ml/code/logs/
  ```

  However, you still need to **update your `configs.yaml` file** to point to **wherever your latest assets are located**.

### Configuring Model Paths for SageMaker

When training on SageMaker, model checkpoints are automatically downloaded from S3 during the setup phase. Understanding the path structure is crucial for correct configuration.

#### Path Structure

**SageMaker Directory Layout:**
```
/opt/ml/code/                          # Base directory
├── logs/                              # Download destination for models
│   ├── epic_human_left_carousel_186/  # Downloaded from S3
│   ├── epic_human_right_carousel_186/
│   └── epic_human_fast_carousel_2/
└── sdm_ws/...
```

**S3 Structure:**
```
s3://tri-hid-data-shared-autonomy/sdm_demo/
├── epic_model/                        # Model checkpoints
│   ├── epic_human_left_carousel_186/
│   ├── epic_human_right_carousel_186/
│   └── epic_human_fast_carousel_2/
└── epic_logs/                         # Training logs
```

#### Correct Configuration

In `configs.yaml`, specify paths **relative to `/opt/ml/code/`** using the format `'logs/model_name'`:

```yaml
epic_ai:
  human_ego_agent_paths: [
    'logs/epic_human_left_carousel_186',
    'logs/epic_human_right_carousel_186',
    'logs/epic_human_fast_carousel_2',
  ]
```

#### How It Works

1. **During setup** (`setup_sdm.sh` calls `download_s3_artifacts.py`):
   - Reads `human_ego_agent_paths` from your config
   - Extracts the base directory name (e.g., from `'logs/epic_human_left_carousel_186'` → `'epic_human_left_carousel_186'`)
   - Downloads matching directories from `s3://.../epic_model/` to `/opt/ml/code/logs/`

2. **During training** (`train.py`):
   - Reads path from config: `'logs/epic_human_left_carousel_186'`
   - Converts to absolute path: `/opt/ml/code/logs/epic_human_left_carousel_186`
   - Loads model from: `/opt/ml/code/logs/epic_human_left_carousel_186/latest_model.pt`

#### Adding a New Model

To add a new model from S3:

1. Ensure your model is uploaded to S3:
   ```bash
   aws s3 cp my_model/ s3://tri-hid-data-shared-autonomy/sdm_demo/epic_model/my_model/ --recursive
   ```

2. Add to `configs.yaml`:
   ```yaml
   human_ego_agent_paths: [
     'logs/my_model',  # Note: logs/ prefix required
   ]
   ```

3. Launch training - the model will be automatically downloaded during setup.
---

## Contributing

This repository is maintained by Toyota Research Institute. For questions or issues, please open a GitHub issue.

## License and Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0). See the [LICENSE](LICENSE) file for details.

### Citation

If you use this code in your research, please cite:

```bibtex
@article{dream2assist2024,
  title={Dream2Assist: Learning Intent-Aware Assistive Policies via Model-Based Reinforcement Learning},
  author={DeCastro, Jonathan and Silva, Andrew and Gopinath, Deepak and Sumner, Emily and Balch, Thomas Matrai and Dees, Laporsha and Rosman, Guy},
  journal={8th Annual Conference on Robot Learning},
  year={2024}
}
```
