export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export EPIC_WS_DIR=$(dirname $(dirname $(dirname $(dirname $SCRIPT_DIR))))

domain_id=$1
cuda_visible_device=$2

if [ -z "$domain_id" ]
then
    domain_id=61
fi
if [ -z "$cuda_visible_device" ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$cuda_visible_device
fi

readonly steer_scale=450

source $EPIC_WS_DIR/install/setup.bash

export ROS_DOMAIN_ID=$domain_id
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/tri-runner-0-storage/motion-sim-0.23.5.2/bazel/experimental/carla_coordinator/runtime/files/fastrtps_profile_local_run.xml

ros2 run shared_decision_making ua_shared_controller \
    --steer_scale $steer_scale \
    --expected_time_step 0.02 \
    --phase_in_time 3 \
    --sdm_activation_delay 8 \
    --agent_path ~/sdm_demo/policies/epic_ai_leftright_carousel_straight_nointent_181/model_checkpoint_118099.pt \
    --dreamer_configs ~/epic_workspace/src/dream2assist/dream2assist/dream2assist/configs.yaml \
    --smoothen_actions \
    --smoothen_steps 5 \
    --visual_smoothen_steps 2 \
    --display_rollout_horizon 8 \
    --display_rollout_max_length 7 \
    --ai_intevention_intensity_threshold 1.0 \
    --trajectory_step_size 0.5 \
    --trajectory_euler_step_size 0.05 \
    --trajectory_lon_offset 3.0 \
    --trajectory_lat_offset 0.5 \
    --tf_skidpad_to_epa \
    --sdm_outer_activation_distance 50 \
    --sdm_inner_activation_distance 10 \
    --sdm_cone_activation_angle 70 \
    --disable_sdm_if_ado_behind \
    --enable_controls \
    --enable_haptics \
    --haptics_gain 0.8 \
    --throttle_brake_scalar 0.0 \
    --clip_ai_throttle \
    --speed_cap 15.0 \
    # --enable_trajectory \
    # --colorize_trajectory \
    # --always_active \
    # --enable_controls \
    # --enable_haptics \
    # --verbose \
    # --test_pass_through \
