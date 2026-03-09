import argparse
from copy import deepcopy
import functools
import glob
import math
import os
import pathlib
import sys
import warnings

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml
import wandb

from gymnasium import spaces

sys.path.append(str(pathlib.Path(__file__).parent))

# import exploration as expl
from dreamer import Dreamer
from d2a import Dream2Assist
from envs import *
import tools as tools
from envs.render_utils import Render
import envs.wrappers as wrappers
from envs.human_wrappers import FixedLaggyHumanModel
from util import count_steps, make_absolute_path, ProcessEpisodeWrap

import torch
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


# Fallback implementations for hail_launch functions when not available
def _get_training_meta_fallback():
    """Fallback implementation when hail_launch is not available.

    Detects SageMaker environment using standard environment variables.
    """
    import os
    on_sagemaker = 'SM_TRAINING_ENV' in os.environ
    return {
        'on_sagemaker': on_sagemaker,
        'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', './output'),
        'training_job_name': os.environ.get('TRAINING_JOB_NAME', 'local-training'),
        'world_size': 1,
        'rank': 0,
        'local_rank': 0,
    }


def _init_wandb_fallback(**kwargs):
    """Fallback implementation using standard wandb API.

    Note: hail_launch's init_wandb may have additional features.
    This fallback provides basic wandb initialization for external usage.
    """
    # Remove output_dir if present - not a standard wandb.init parameter
    kwargs.pop('output_dir', None)
    wandb.init(**kwargs)


def _get_ddp_status_fallback(run_meta):
    """Fallback no-op for non-distributed training.

    When hail_launch is not available, assume single-process training.
    """
    pass


def _upload_checkpoints_to_s3_fallback(**kwargs):
    """Fallback no-op when not on SageMaker.

    Local training doesn't upload to S3.
    """
    pass


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def try_float_conversion(value):
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except:
        return value


def make_human_agents(config, on_sagemaker):
    """Create human agents based on the configuration.

    Parameters:
        config: The configuration.

    Returns:
        labeled_human_agents: A dictionary of human agents.
        human_config: The modified configuration for human agents.
    """
    env_class = config.task

    # TODO(jon) Here, we're making in-line changes to the AI config.  Rather, we should be using the human configs directly.
    human_config = deepcopy(config)
    human_config.egocentric_agent_names = ['human']
    human_config.human_ego_agent_paths = []
    human_config.frozen_agent_id = 0  # Ensure the weights of this agent are not updated.
    human_config.use_intent = False
    human_config.intent_head = 'binary'

    human_only_env = eval(env_class)(
        config=human_config,
        render_mode=None,
        dreamer_policy=None,
    )

    logdir = pathlib.Path(str(config.logdir) + "_tmp").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(human_config.traindir)
    tmp_logger = tools.Logger(logdir, config.action_repeat * step)
    obs_spaces = human_only_env.observation_spaces
    act_spaces = human_only_env.action_spaces
    labeled_human_agents = {}
    for path in getattr(config, 'human_ego_agent_paths', []):
        human_agents = Dream2Assist(obs_spaces, act_spaces, [human_config], tmp_logger, dataset=None).to(config.device)
        human_agents.requires_grad_(requires_grad=False)
        filename = os.path.join(str(make_absolute_path(path, on_sagemaker)), "latest_model.pt")
        label = None
        for i, l in human_only_env.get_label_dict().items():
            print(f"Checking if {l} is in {filename}")
            if l in str(filename):
                label = i
                break
        assert label is not None; f"No label found in checkpoint name. Expected {human_only_env.get_label_dict().values()} in the directory name."
        human_agents.load_state_dict(torch.load(filename))
        human_agents._should_pretrain._once = False
        if label in list(labeled_human_agents.keys()):
            raise RuntimeError("Duplicate label found in supplied human agent policy directories.")
        labeled_human_agents[label] = human_agents
    return labeled_human_agents, human_config


def make_env(config, logger, mode, train_eps, eval_eps, on_sagemaker):
    """Create an environment.

    Parameters:
        config: The configuration.
        logger: The logger.
        mode: The mode.
        train_eps: The training episodes.

    Returns:
        env: The environment.
    """
    # Load the external human agent policy.
    directory = config.traindir

    # Create temporary environment to extract observation/action space parameters
    # This avoids hardcoding domain-specific constants
    env_class = config.task
    config_temp = deepcopy(config)
    temp_env = eval(env_class)(
        config=config_temp,
        render_mode=None,
        dreamer_policy=None,
        human_agents=None,
    )

    # Extract ado spaces from the base environment
    # Wrapper classes expose base_env property; direct environments have spaces as attributes
    if hasattr(temp_env, 'base_env'):
        # Wrapper class - access the unwrapped base environment
        ado_observation_space = temp_env.base_env.ado_observation_space
        ado_action_space = temp_env.base_env.ado_action_space
    elif hasattr(temp_env, 'ado_observation_space'):
        # Direct environment class
        ado_observation_space = temp_env.ado_observation_space
        ado_action_space = temp_env.ado_action_space
    else:
        raise AttributeError(
            f"Environment class {env_class} does not expose ado_observation_space or base_env. "
            f"Cannot extract space parameters programmatically."
        )

    # Clean up temporary environment to avoid resource leaks
    temp_env.close()
    del temp_env

    # Configure num_actions from the extracted action space
    config = deepcopy(config)
    config.num_actions = ado_action_space.n if hasattr(ado_action_space, "n") else ado_action_space.shape[0]

    logdir = pathlib.Path(str(config.logdir) + "_tmp").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    tmp_logger = tools.Logger(logdir, config.action_repeat * step)

    # Load the ado-agent
    eval_policy = None
    if not getattr(config, 'use_pure_pursuit', False):
        config.use_intent = False
        agent = Dreamer(
            ado_observation_space,
            ado_action_space,
            config,
            tmp_logger,
            dataset=None,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        human_ego_agent_path = getattr(config, 'human_ego_agent_paths', None)
        if human_ego_agent_path:
            agent.load_state_dict(torch.load(str(make_absolute_path(human_ego_agent_path, on_sagemaker))))
        agent._should_pretrain._once = False
        eval_policy = functools.partial(agent, training=False)
    config.use_intent = True

    # Load the external human agent policies, and feed those into the active env.
    labeled_human_agents, human_config = make_human_agents(config, on_sagemaker) if getattr(config, 'use_human_inference_reward', False) else ({}, None)

    env_class = config.task
    env = eval(env_class)(
        config=config,
        render_mode=None if mode == "eval" else "no_render",
        dreamer_policy=eval_policy,
        human_agents=labeled_human_agents,
    )
    renderer = functools.partial(Render, env)
    if config.use_human_model:
        human_model_class = config.human_model_class
        env = eval(human_model_class)(env, config)

    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.AssistiveRewards(env, labeled_human_agents, config)
    if (mode == "train") or (mode == "eval"):
        callbacks = [
            functools.partial(
                ProcessEpisodeWrap.process_episode,
                config,
                logger,
                mode,
                train_eps,
                eval_eps,
                renderer,
            )
        ]
        env = wrappers.CollectDataset(env, mode, train_eps, callbacks=callbacks)
    env = wrappers.RewardObs(env)
    return env, human_config


def main(config):
    # Try to import hail_launch for SageMaker functionality
    # If not available (external/public usage), use fallback implementations
    try:
        from hail_launch.load_env import get_training_meta, init_wandb, get_ddp_status
        from hail_launch.util.utils import upload_checkpoints_to_s3
    except ImportError:
        # Use fallback implementations for external/local usage
        get_training_meta = _get_training_meta_fallback
        init_wandb = _init_wandb_fallback
        get_ddp_status = _get_ddp_status_fallback
        upload_checkpoints_to_s3 = _upload_checkpoints_to_s3_fallback

    run_meta = get_training_meta()
    on_sagemaker = run_meta["on_sagemaker"]
    if on_sagemaker:
        config.traindir = pathlib.Path(config.logdir).resolve() / "train_eps"
        config.evaldir = pathlib.Path(config.logdir).resolve() / "eval_eps"
        config.logdir = run_meta['output_dir']

    config.logdir = pathlib.Path(config.logdir).expanduser().resolve()

    # Helper function to resolve paths relative to train.py or as absolute paths
    def resolve_config_path(path_str):
        if not path_str:
            return None
        path = pathlib.Path(path_str)
        # If it's already absolute and exists, use it
        if path.is_absolute():
            return path.resolve(strict=True)
        # Try relative to current directory first
        if path.exists():
            return path.resolve(strict=True)
        # Try relative to train.py script location
        script_dir = pathlib.Path(__file__).parent
        script_relative = script_dir / path
        if script_relative.exists():
            return script_relative.resolve(strict=True)
        # Try relative to home directory (for paths like sdm_ws/...)
        home_relative = pathlib.Path.home() / path
        if home_relative.exists():
            return home_relative.resolve(strict=True)
        # If nothing works, try to resolve it anyway (will fail with good error)
        return path.resolve(strict=True)

    relative_track_csv = config.track_csv
    if relative_track_csv:
        config.track_csv = resolve_config_path(relative_track_csv)

    relative_track_map_process_csv = config.track_map_process_csv
    if relative_track_map_process_csv:
        config.track_map_process_csv = resolve_config_path(relative_track_map_process_csv)

    relative_ado_rails = getattr(config, 'ado_rails', None)
    if relative_ado_rails:
        config.ado_rails = resolve_config_path(relative_ado_rails)

    print("***************************************************************")
    print("Logdir: ", config.logdir)
    print("track csv path: ", config.track_csv)
    print("track_map_process_csv: ", config.track_map_process_csv)
    print("ado_rails: ", getattr(config, 'ado_rails', None))
    print("traindir: ", config.traindir)
    print("evaldir: ", config.evaldir)
    print("***************************************************************")

    config.traindir = config.traindir or config.logdir / "train_eps"
    config.evaldir = config.evaldir or config.logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    config_dict = dict(
            (attr_name, getattr(config, attr_name)) for attr_name in dir(config) if not attr_name.startswith("__")
        )

    get_ddp_status(run_meta)
    config_dict.update(run_meta)

    # start a new wandb run to track this script
    init_wandb(
        # uncomment when debugging to prevent spamming our wandb project
        # mode="disabled",
        name=config.logdir.name,
        # set the wandb project where this run will be logged
        project="dream2assist",
        # track hyperparameters and run metadata
        config=config_dict,
        output_dir=config.logdir,
    )

    print("Logdir", config.logdir)
    config.logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(config.logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    offline_eps = None
    if config.offlinedir:
        offline_eps = tools.load_episodes(config.offlinedir, limit=config.dataset_size, require_dir=True)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, on_sagemaker)
    train_envs_configs = [make("train") for _ in range(config.envs)]
    eval_envs_configs = [make("eval") for _ in range(config.envs)]
    train_envs, human_config = zip(*train_envs_configs)
    human_config = human_config[0] if config.use_human_inference_reward else None
    eval_envs, _ = zip(*eval_envs_configs)

    # Set random intents for the training environment.
    if config.intent_head == "binary":
        train_envs[0].set_inferred_intents(
            [np.random.randint(0, 2) for _ in range(len(config.egocentric_agent_names))]
        )
    elif config.intent_head == "onehot":
        train_envs[0].set_inferred_intents(
            [np.eye(len(config.behavior_labels))[np.random.randint(0, len(config.behavior_labels))] for _ in range(len(config.egocentric_agent_names))]
        )
    else:
        raise NotImplementedError(f"Unknown intent head type {config.intent_head}.")

    obs_spaces = train_envs[0].observation_spaces
    act_spaces = train_envs[0].action_spaces
    print(f" obs spaces {obs_spaces}")
    print(f" act spaces {act_spaces}")
    names = [f"{n}_agent" for n in config.egocentric_agent_names]
    random_actors = []
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        for acts in act_spaces:
            config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
            # config.num_actions = 6
            if hasattr(acts, "discrete"):
                random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(config.envs, 1))
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.Tensor(acts.low).repeat(config.envs, 1),
                        torch.Tensor(acts.high).repeat(config.envs, 1),
                    ),
                    1,
                )
            random_actors.append(random_actor)

        def random_agent(o, d, s, r):
            result = {"action": [], "logprob": []}
            for random_actor in random_actors:
                action = random_actor.sample()
                result["action"].append(action)
                result["logprob"].append(random_actor.log_prob(action))
            result = {k: list(zip(*v)) for k, v in result.items()}  # envs first

            return result, None

        random_intent = lambda s: [np.random.randint(0, 2) for _ in range(len(config.egocentric_agent_names))]

        #TODO (deepak.gopinath). Not sure if we have a human model wrapper, whether it should be active when collecting the dataset
        tools.simulate(random_agent, train_envs, prefill, get_intents=random_intent)
        logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    offline_dataset = None if offline_eps is None else make_dataset(offline_eps, config, rebalance_if_labeled=True)
    agents = Dream2Assist(
        obs_spaces,
        act_spaces,
        [config, human_config] if human_config is not None else [config] * len(obs_spaces),
        logger,
        train_dataset,
        offline_dataset,
        names,
    ).to(config.device)
    agents.requires_grad_(requires_grad=False)
    if (config.logdir / "latest_model.pt").exists():
        print(f"Loading model from {str(config.logdir / 'latest_model.pt')}.")
        agents.load_state_dict(torch.load(config.logdir / "latest_model.pt"))
        agents._should_pretrain._once = False

    state = None
    while agents._step < config.steps:
        logger.write()
        print("Start evaluation.")
        eval_policies = functools.partial(agents, training=False)
        # Sync the eval envs with the training envs.
        # TODO(jon) Re-implement this.
        # eval_envs[0].set_ado_slowdown_factor(train_envs[0].ado_slowdown_factor)
        eval_state, trajectory = tools.population_simulate(
            agents,
            eval_envs,
            episodes=config.eval_episode_num,
            config=config,
            training=False,
            on_sagemaker=on_sagemaker,
        )
        if config.video_pred_log:
            dataset = next(eval_dataset)
            video_pred_ai = agents._wm[0].video_pred(agents.data_agent(dataset, 0))
            video_pred_human = agents._wm[1].video_pred(agents.data_agent(dataset, 1))
            logger.video("eval_openl_ai", to_np(video_pred_ai))
            # pass array of dim (batch, time, channel, width, height)
            wandb.log({"eval_openl_ai": wandb.Video(np.moveaxis(to_np(video_pred_ai), -1, -3))})
            logger.video("eval_openl", to_np(video_pred_human))
            # pass array of dim (batch, time, channel, width, height)
            wandb.log({"eval_openl": wandb.Video(np.moveaxis(to_np(video_pred_human), -1, -3))})
        print("Start training.")
        state, _ = tools.population_simulate(
            agents,
            train_envs,
            steps=config.eval_every,
            state=state,
            config=config,
            training=True,
            on_sagemaker=on_sagemaker,
        )
        torch.save(agents.state_dict(), config.logdir / "latest_model.pt")
        model_checkpoint_filename = f"model_checkpoint_{agents._step:06d}.pt"
        torch.save(agents.state_dict(), config.logdir / model_checkpoint_filename)
        if run_meta['on_sagemaker']:
            upload_checkpoints_to_s3(checkpoint_dir=config.logdir, job_name=run_meta["training_job_name"], keep_last_n=3)

        # TODO: See if we need to save checkpoints of the ai and human agent separately.
        # torch.save(agents[0].state_dict(), config.logdir / "latest_model_ai.pt")
        # torch.save(agents[0].state_dict(), config.logdir / f"model_checkpoint_ai_{agents._step:06d}.pt")
        # torch.save(agents[1].state_dict(), config.logdir / "latest_model_human.pt")
        # torch.save(agents[1].state_dict(), config.logdir / f"model_checkpoint_human_{agents[1]._step:06d}.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        value = try_float_conversion(value)
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
