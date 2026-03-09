from gymnasium.envs.registration import register

register(
    id="Box2dSharedRacing-v0",
    entry_point="envs.box2d_shared_racing:Box2dSharedRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

register(
    id="EpicDynamicsSharedRacing-v0",
    entry_point="envs.epic_dynamics_shared_racing:EpicDynamicsSharedRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

from envs.box2d_shared_racing_wrapper import Box2dSharedRacingWrapper
try:
    from envs.epic_dynamics_shared_racing_wrapper import EpicDynamicsSharedRacingWrapper
except ImportError as e:
    print(e)
    print("EpicDynamicsSharedRacingWrapper not available. Please install the required dependencies.")
