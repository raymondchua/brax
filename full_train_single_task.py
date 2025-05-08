# import functools
# import jax
# import os
#
# from datetime import datetime
# from jax import numpy as jp
# import matplotlib.pyplot as plt
#
# import brax
#
# import flax
# from brax import envs
# from brax.io import model
# from brax.io import json
# from brax.io import html
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.sac import train as sac
#
# env_name = "walker2d"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# backend = "positional"  # @param ['generalized', 'positional', 'spring']
#
# env = envs.get_environment(env_name=env_name, backend=backend)
# state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
#
# train_fn = {
#     "inverted_pendulum": functools.partial(
#         ppo.train,
#         num_timesteps=2_000_000,
#         num_evals=20,
#         reward_scaling=10,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=5,
#         num_minibatches=32,
#         num_updates_per_batch=4,
#         discounting=0.97,
#         learning_rate=3e-4,
#         entropy_cost=1e-2,
#         num_envs=2048,
#         batch_size=1024,
#         seed=1,
#     ),
#     "inverted_double_pendulum": functools.partial(
#         ppo.train,
#         num_timesteps=20_000_000,
#         num_evals=20,
#         reward_scaling=10,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=5,
#         num_minibatches=32,
#         num_updates_per_batch=4,
#         discounting=0.97,
#         learning_rate=3e-4,
#         entropy_cost=1e-2,
#         num_envs=2048,
#         batch_size=1024,
#         seed=1,
#     ),
#     "ant": functools.partial(
#         ppo.train,
#         num_timesteps=50_000_000,
#         num_evals=10,
#         reward_scaling=10,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=5,
#         num_minibatches=32,
#         num_updates_per_batch=4,
#         discounting=0.97,
#         learning_rate=3e-4,
#         entropy_cost=1e-2,
#         num_envs=4096,
#         batch_size=2048,
#         seed=1,
#     ),
#     "humanoid": functools.partial(
#         ppo.train,
#         num_timesteps=50_000_000,
#         num_evals=10,
#         reward_scaling=0.1,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=10,
#         num_minibatches=32,
#         num_updates_per_batch=8,
#         discounting=0.97,
#         learning_rate=3e-4,
#         entropy_cost=1e-3,
#         num_envs=2048,
#         batch_size=1024,
#         seed=1,
#     ),
#     "reacher": functools.partial(
#         ppo.train,
#         num_timesteps=50_000_000,
#         num_evals=20,
#         reward_scaling=5,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=4,
#         unroll_length=50,
#         num_minibatches=32,
#         num_updates_per_batch=8,
#         discounting=0.95,
#         learning_rate=3e-4,
#         entropy_cost=1e-3,
#         num_envs=2048,
#         batch_size=256,
#         max_devices_per_host=8,
#         seed=1,
#     ),
#     "humanoidstandup": functools.partial(
#         ppo.train,
#         num_timesteps=100_000_000,
#         num_evals=20,
#         reward_scaling=0.1,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=15,
#         num_minibatches=32,
#         num_updates_per_batch=8,
#         discounting=0.97,
#         learning_rate=6e-4,
#         entropy_cost=1e-2,
#         num_envs=2048,
#         batch_size=1024,
#         seed=1,
#     ),
#     "hopper": functools.partial(
#         sac.train,
#         num_timesteps=6_553_600,
#         num_evals=20,
#         reward_scaling=30,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         discounting=0.997,
#         learning_rate=6e-4,
#         num_envs=128,
#         batch_size=512,
#         grad_updates_per_step=64,
#         max_devices_per_host=1,
#         max_replay_size=1048576,
#         min_replay_size=8192,
#         seed=1,
#     ),
#     "walker2d": functools.partial(
#         sac.train,
#         num_timesteps=7_864_320,
#         num_evals=20,
#         reward_scaling=5,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         discounting=0.997,
#         learning_rate=6e-4,
#         num_envs=128,
#         batch_size=128,
#         grad_updates_per_step=32,
#         max_devices_per_host=1,
#         max_replay_size=1048576,
#         min_replay_size=8192,
#         seed=1,
#     ),
#     "halfcheetah": functools.partial(
#         ppo.train,
#         num_timesteps=50_000_000,
#         num_evals=20,
#         reward_scaling=1,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=20,
#         num_minibatches=32,
#         num_updates_per_batch=8,
#         discounting=0.95,
#         learning_rate=3e-4,
#         entropy_cost=0.001,
#         num_envs=2048,
#         batch_size=512,
#         seed=3,
#     ),
#     "pusher": functools.partial(
#         ppo.train,
#         num_timesteps=50_000_000,
#         num_evals=20,
#         reward_scaling=5,
#         episode_length=1000,
#         normalize_observations=True,
#         action_repeat=1,
#         unroll_length=30,
#         num_minibatches=16,
#         num_updates_per_batch=8,
#         discounting=0.95,
#         learning_rate=3e-4,
#         entropy_cost=1e-2,
#         num_envs=2048,
#         batch_size=512,
#         seed=3,
#     ),
# }[env_name]
#
#
# max_y = {
#     "ant": 8000,
#     "halfcheetah": 8000,
#     "hopper": 2500,
#     "humanoid": 13000,
#     "humanoidstandup": 75_000,
#     "reacher": 5,
#     "walker2d": 5000,
#     "pusher": 0,
# }[env_name]
# min_y = {"reacher": -100, "pusher": -150}.get(env_name, 0)
#
# xdata, ydata = [], []
# times = [datetime.now()]
#
#
# def progress(num_steps, metrics):
#     for key, value in metrics.items():
#         print(f"{key}: {value}")
#     times.append(datetime.now())
#     xdata.append(num_steps)
#     ydata.append(metrics["eval/episode_reward"])
#     plt.xlim([0, train_fn.keywords["num_timesteps"]])
#     plt.ylim([min_y, max_y])
#     plt.xlabel("# environment steps")
#     plt.ylabel("reward per episode")
#     plt.plot(xdata, ydata)
#     # plt.show()
#     # save the plot
#     plt.savefig(f"{env_name}_{backend}.png")
#
#
# make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
#
# print(f"time to jit: {times[1] - times[0]}")
# print(f"time to train: {times[-1] - times[1]}")

import functools
import jax
import os
import hydra
import wandb

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

import brax
import time
from omegaconf import OmegaConf

import flax
from brax import envs
from brax.io import model
from brax.training.agents.sac import train as sac

def progress(num_steps, metrics):
    print("Steps:", num_steps)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    # Optionally, you can add logic to save or visualize metrics here.
    wandb.log(metrics, step=num_steps)

def modify_env_properties(env, env_name: str, mass_factor=1.0, friction_factor=1.0):
    sys = env.unwrapped.sys
    config = sys.config
    # Update mass for all bodies
    for body in config.bodies:
        body.mass *= mass_factor

    # Update friction for all colliders
    for collider in config.colliders:
        collider.friction *= friction_factor

    # Reinitialize the environment with the modified config
    return envs.create(env_name=env_name, config=config)

def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    env_name = "ant"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = "positional"  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name, backend=backend)
    env = modify_env_properties(env, env_name=env_name, mass_factor=config["MASS_FACTOR"], friction_factor=config["FRICTION_FACTOR"])
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    # Train
    train_fn = {
        "ant": functools.partial(
            sac.train,
            num_timesteps=config["TIMESTEPS"],
            num_evals=config["NUM_EVALS"],
            reward_scaling=config["REWARD_SCALING"],
            episode_length=config["EPISODE_LENGTH"],
            normalize_observations=config["NORMALIZE_OBSERVATIONS"],
            action_repeat=config["ACTION_REPEAT"],
            discounting=config["DISCOUNTING"],
            learning_rate=config["LR"],
            num_envs=config["NUM_ENVS"],
            batch_size=config["BATCH_SIZE"],
            grad_updates_per_step=config["GRAD_UPDATES_PER_STEP"],
            max_devices_per_host=config["MAX_DEVICES_PER_HOST"],
            max_replay_size=config["MAX_REPLAY_SIZE"],
            min_replay_size=config["MIN_REPLAY_SIZE"],
            seed=config["SEED"],
        ),
    }[env_name]

    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)




    print(f"Took {time.time() - t0} seconds to complete.")


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
