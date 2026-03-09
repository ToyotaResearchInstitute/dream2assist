#!/bin/bash
# Setup script for Shared Decision Making (SDM) environment on SageMaker.
#
# Requirements:
# - TRI AWS SageMaker environment
# - Access to TRI S3 buckets (s3://tri-hid-data-shared-autonomy/)
# - ROS2 Humble installation
# - TRI proprietary packages (epic_workspace, hail_launch, etc.)
#
# For public standalone Box2D usage, this script is not needed.
# See README.md for Box2D installation and training instructions.

set -x
shopt -s extglob

rm -rf /opt/conda
source ~/.bashrc
# Install the lightweight tool `uv`
curl -LsSf https://astral.sh/uv/install.sh | sh
# Include the `uv` executable path in current environment.
source $HOME/.local/bin/env

mkdir sdm_ws
mv !(hail_launch*|sdm_ws*) sdm_ws

uv venv --python 3.10 .venv --clear
source .venv/bin/activate
uv pip install pip

pip install --upgrade-strategy only-if-needed -e hail_launch

echo "********************************************************************"
which python
echo "********************************************************************"

chmod 1777 /tmp
apt update
apt install -y sudo
sudo apt update && sudo apt install -y curl gnupg2 lsb-release locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe -y

export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-humble-ros-base
sudo apt install -y ros-dev-tools

echo "*****************************************************************"
source /opt/ros/humble/setup.bash
echo "*****************************************************************"

# This is stored in the docker image
# pip install /staging/carla-0.9.15-cp311-cp311-linux_x86_64.whl

sudo apt install -y swig

pip install pathspec

cd sdm_ws
export SDM_WS_DIR=/opt/ml/code/sdm_ws/

# Note: sdm_ws main repo has hail_launch as a dependency in requirements.txt as follows
# git+ssh://git@github.com/ToyotaResearchInstitute/hail_launch.git@main
# This needs to be removed when executing the training in sagemaker because sagemaker
# instance does not have ssh keys to be able to clone the repo.
HAIL_LAUNCH_REPO_REFERENCE="git+ssh://git@github.com/ToyotaResearchInstitute/hail_launch.git@main"
sed -i "\|$HAIL_LAUNCH_REPO_REFERENCE|c\\" requirements.txt

# View cuda configuration
pip show torch

pip install -r requirements.txt
pip install src/dream2assist/dream2assist/

# Set up sdm_ws and its dependencies.
pip install -r requirements.txt
./scripts/install_src_requirements.bash
# Set all jax packages to version 0.4.33 for compatibility with python 3.10
pip install -U "jax==0.4.33" "jaxlib==0.4.33" "jax-cuda12-plugin==0.4.33" "jax-cuda12-pjrt==0.4.33"
# Downgraded PyTorch to version 2.0 for compatibility with SageMaker image
pip install "torch==2.0" "torchvision"
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12
source /opt/ros/humble/setup.bash
python3 -m colcon build --symlink-install
source install/setup.bash
cd ..

pip install casadi


git config --global --add safe.directory /opt/ml/code

mkdir -p logs/assets

python -u sdm_ws/src/dream2assist/download_s3_artifacts.py

if [ $? -ne 0 ]; then
    echo "Error: download_s3_artifacts.py failed."
    exit 1
fi

cd /opt/ml/code

# Set appropriate cuda configuration
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip show torch

# Create a non-privilege user to run Carla
useradd -ms /bin/bash sdm_user
su - sdm_user 

