#!/usr/bin/env python3
"""Train Graph DiT RL Policy using RSL-RL framework.

This script integrates GraphDiTRLPolicy with RSL-RL's OnPolicyRunner,
allowing efficient multi-environment parallel training.

Usage:
    ./isaaclab.sh -p scripts/graph_dit_rl/train_rsl_rl.py \
        --task SO-ARM101-Reach-Cube-v0 \
        --pretrained_checkpoint ./logs/graph_dit/.../best_model.pt \
        --num_envs 64 \
        --max_iterations 1500
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

import argparse
import sys

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Graph DiT RL Policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to pre-trained Graph DiT checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

import os
from datetime import datetime

import gymnasium as gym
import torch

import SO_100.tasks  # noqa: F401  # Register environments
from SO_100.policies.graph_dit_rsl_rl_actor_critic import GraphDiTActorCritic  # noqa: F401
# Import ensures GraphDiTActorCritic is in Python namespace for RSL-RL to find


def main():
    """Main training function - uses standard RSL-RL flow with Hydra."""
    print("=" * 70)
    print("Graph DiT RL Fine-Tuning with RSL-RL (Standard Flow)")
    print("=" * 70)
    print(f"Note: This script uses the SAME interface as standard RSL-RL training")
    print(f"      (just like lift task). Hydra will auto-load the config.")
    print("=" * 70)

    # Set pretrained checkpoint via environment variable (if provided)
    # The config class will automatically read it
    if args_cli.pretrained_checkpoint:
        os.environ["GRAPH_DIT_PRETRAINED_CHECKPOINT"] = args_cli.pretrained_checkpoint
        print(f"\n[Train] Set pretrained checkpoint via env var: {args_cli.pretrained_checkpoint}")

    # Use the STANDARD RSL-RL training flow
    # Hydra will automatically load ReachCubeGraphDiTRLRunnerCfg from environment registration
    # This is the SAME way lift task works!
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    
    # Use Hydra to load config (standard RSL-RL way)
    # This will automatically load ReachCubeGraphDiTRLRunnerCfg from the environment registration
    # Note: Now we use SO-ARM101-Reach-Cube-v0 which directly uses Graph DiT config
    @hydra_task_config("SO-ARM101-Reach-Cube-v0", "rsl_rl_cfg_entry_point")
    def train_with_config(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        """Training function that gets config from Hydra."""
        
        # Override CLI arguments
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed
            env_cfg.seed = args_cli.seed

        # Create environment
        # Note: Now SO-ARM101-Reach-Cube-v0 directly uses Graph DiT config
        env = gym.make("SO-ARM101-Reach-Cube-v0", cfg=env_cfg, 
                      render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Create log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, timestamp)
        os.makedirs(log_dir, exist_ok=True)

        # Wrap environment for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Ensure SO_100 is available in namespace for RSL-RL's eval() calls
        # RSL-RL uses eval() internally to dynamically load custom ActorCritic classes.
        import SO_100.policies.graph_dit_rsl_rl_actor_critic  # noqa: F401
        import builtins
        import sys
        # Inject SO_100 into builtins so eval() can access it
        if not hasattr(builtins, "SO_100"):
            builtins.SO_100 = sys.modules["SO_100"]

        # Create RSL-RL runner (standard way)
        print(f"\n[Train] Creating RSL-RL runner...")
        print(f"[Train] Policy class: {agent_cfg.policy.class_name}")
        # Get device from config or use default
        device = args_cli.device if hasattr(args_cli, 'device') and args_cli.device is not None else agent_cfg.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

        # Print model info
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        
        total_params = sum(p.numel() for p in policy_nn.parameters())
        trainable_params = sum(p.numel() for p in policy_nn.parameters() if p.requires_grad)
        print(f"[Train] Total parameters: {total_params:,}")
        print(f"[Train] Trainable parameters: {trainable_params:,}")
        print(f"[Train] Frozen parameters: {total_params - trainable_params:,}")
        print(f"[Train] Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        # Dump configuration
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # Run training
        print(f"\n[Train] Starting training with {env_cfg.scene.num_envs} parallel environments...")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        # Close environment
        env.close()

    # Call the Hydra-decorated function
    # This will automatically load the config from environment registration
    train_with_config()


if __name__ == "__main__":
    main()
    simulation_app.close()
