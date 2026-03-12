# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RSL-RL agent and report success rate."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL and report success rate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1248, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=2000, help="Number of episodes to evaluate.")
parser.add_argument("--max_episode_steps", type=int, default=300, help="Safety cap on episode length (steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation (num_envs must be 1).")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=1_000_000, help="Interval between videos (unused in eval, kept for parity).")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible (single env).")
# Action noise is always enabled by default during evaluation; no CLI switch provided.
# checkpoint selection
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime
import json
import csv
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401


def _extract_asset_names_from_env(env) -> list[str]:
    """Extract asset base names from env config's MultiAssetSpawnerCfg.

    Assumes env.unwrapped.cfg.scene.object.spawn.assets_cfg exists and each has a usd_path like .../<name>/<name>.usd.
    """
    names: list[str] = []
    cfg = getattr(env, "unwrapped", env)
    scene = getattr(cfg, "cfg", None)
    if scene is None:
        scene = getattr(env, "cfg", None)
    if scene is None:
        return names
    scene_cfg = getattr(scene, "scene", None)
    if scene_cfg is None:
        return names
    object_cfg = getattr(scene_cfg, "object", None)
    spawn_cfg = getattr(object_cfg, "spawn", None) if object_cfg is not None else None
    assets_cfg = getattr(spawn_cfg, "assets_cfg", None) if spawn_cfg is not None else None
    if assets_cfg is None:
        return names
    for usd_cfg in assets_cfg:
        usd_path = getattr(usd_cfg, "usd_path", None)
        if isinstance(usd_path, str) and len(usd_path) > 0:
            base = os.path.basename(os.path.dirname(usd_path))
            names.append(base)
        else:
            names.append("unknown")
    return names


def _build_env_to_object_index(num_envs: int, num_assets: int) -> torch.Tensor:
    """Deterministic mapping from env_id to asset index when random_choice=False."""
    if num_assets <= 0:
        return torch.full((num_envs,), -1, dtype=torch.long)
    idx = torch.arange(num_envs, dtype=torch.long)
    return idx % num_assets


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate RSL-RL agent and report success rate over multiple episodes."""
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable observation noise for evaluation
    env_cfg.disable_obs_noise = True

    # Limitations for video recording in evaluation
    if args_cli.video and env_cfg.scene.num_envs != 1:
        print("[WARN] Video recording in eval supports only num_envs=1. Overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1

    # specify directory for loading experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # resolve resume path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,  # record first episode by default
            "video_length": min(args_cli.video_length, args_cli.max_episode_steps),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy_func = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy_obj = ppo_runner.alg.policy  # Get the actual policy object for act() method

    # evaluation loop (vectorized)
    num_envs = env.unwrapped.num_envs
    episodes_completed = 0
    total_successes = 0
    
    # Get initial stats from env - require them to be available
    if not hasattr(env.unwrapped, 'total_episodes') or not hasattr(env.unwrapped, 'total_successes'):
        raise AttributeError("Environment does not have total_episodes or total_successes. This eval script requires the NonPrehensileEnv with success tracking.")
    
    episodes_completed = env.unwrapped.total_episodes
    total_successes = env.unwrapped.total_successes
    

    # per-object accounting
    asset_names = _extract_asset_names_from_env(env)
    num_assets = len(asset_names)
    env_to_obj_idx = _build_env_to_object_index(num_envs, num_assets)
    # counters stored on CPU for printing simplicity
    obj_episodes = {name: 0 for name in asset_names}
    obj_successes = {name: 0 for name in asset_names}

    # reset environment
    obs, _ = env.get_observations()

    # timing
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None

    # Initialize progress bar
    pbar = tqdm(total=args_cli.num_episodes, desc="Evaluating", unit="episodes")
    pbar.set_postfix({
        "Success Rate": "0.00%",
        "Episodes": 0,
        "Successes": 0
    })

    # Run until desired number of episodes are completed
    step_count = 0
    while episodes_completed < args_cli.num_episodes and simulation_app.is_running():
        start_time = time.time()
        step_count += 1
        
        with torch.inference_mode():
            # Always use act() to sample actions with noise (like during training)
            actions = policy_obj.act(obs)
            obs, _, dones, _ = env.step(actions)

        # Use env's built-in success tracking - require it to be available
        if not hasattr(env.unwrapped, 'episode_success_buf'):
            raise AttributeError("Environment does not have episode_success_buf. This eval script requires the NonPrehensileEnv with success tracking.")

        # Use environment's episode ending signals directly
        # RslRlVecEnvWrapper already combines terminated | truncated into dones
        ended = dones.bool()
        

        if torch.any(ended):
            ended_ids = torch.where(ended)[0]
            # Use env's built-in statistics - require them to be available
            if not hasattr(env.unwrapped, 'total_episodes') or not hasattr(env.unwrapped, 'total_successes'):
                raise AttributeError("Environment does not have total_episodes or total_successes. This eval script requires the NonPrehensileEnv with success tracking.")
            
            episodes_completed = env.unwrapped.total_episodes
            total_successes = env.unwrapped.total_successes
            
            # per-object accumulation
            for env_id in ended_ids.tolist():
                if 0 <= num_assets and num_assets > 0:
                    obj_idx = int(env_to_obj_idx[env_id].item()) if env_to_obj_idx.numel() == num_envs else -1
                    if 0 <= obj_idx < num_assets:
                        obj_name = asset_names[obj_idx]
                        obj_episodes[obj_name] = obj_episodes.get(obj_name, 0) + 1
                        # Use env's success status before reset
                        if hasattr(env.unwrapped, '_episode_success_before_reset'):
                            env_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                        else:
                            # Fallback: use current episode_success_buf (may be reset)
                            env_success = bool(env.unwrapped.episode_success_buf[env_id].item())
                        
                        if env_success:
                            obj_successes[obj_name] = obj_successes.get(obj_name, 0) + 1
                        # Debug: print first few per-object stats
                        if episodes_completed <= 5:
                            print(f"[DEBUG] Env {env_id} (obj {obj_name}): success={env_success}")
            
            # Update progress bar
            current_success_rate = (total_successes / episodes_completed) * 100 if episodes_completed > 0 else 0.0
            pbar.update(ended_ids.numel())
            pbar.set_postfix({
                "Success Rate": f"{current_success_rate:.2f}%",
                "Episodes": episodes_completed,
                "Successes": total_successes
            })


        # optional real-time pacing
        if args_cli.real_time and (dt is not None):
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    success_rate = (total_successes / episodes_completed) if episodes_completed > 0 else 0.0

    # Close progress bar
    pbar.close()

    # persist results
    results_dir = log_dir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "eval_summary.json")
    per_object_path = os.path.join(results_dir, "eval_per_object.csv")

    results_payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "episodes": int(episodes_completed),
        "successes": int(total_successes),
        "success_rate": float(success_rate),
        "per_object": [
            {
                "name": name,
                "episodes": int(obj_episodes.get(name, 0)),
                "successes": int(obj_successes.get(name, 0)),
                "success_rate": (float(obj_successes.get(name, 0)) / float(obj_episodes.get(name, 0))) if obj_episodes.get(name, 0) > 0 else 0.0,
            }
            for name in sorted(obj_episodes.keys())
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    # write CSV for per-object breakdown
    with open(per_object_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["name", "episodes", "successes", "success_rate"])
        for name in sorted(obj_episodes.keys()):
            ep = int(obj_episodes.get(name, 0))
            sc = int(obj_successes.get(name, 0))
            rate = (sc / ep) if ep > 0 else 0.0
            writer.writerow([name, ep, sc, rate])

    # print summary
    print("\n========== Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"Episodes: {episodes_completed}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Saved: {summary_path}")
    if len(obj_episodes) > 0:
        print(f"Saved: {per_object_path}")
    if num_assets > 0 and len(obj_episodes) > 0:
        print("\nPer-object success rates:")
        for name in sorted(obj_episodes.keys()):
            ep = obj_episodes[name]
            sc = obj_successes.get(name, 0)
            rate = (sc / ep) * 100.0 if ep > 0 else 0.0
            print(f"  - {name}: {sc}/{ep} ({rate:.2f}%)")
    print("=======================================\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
