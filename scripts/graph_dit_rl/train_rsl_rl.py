#!/usr/bin/env python3
"""Train Residual RL Policy using RSL-RL framework.

This script trains a Residual RL policy that uses:
- Frozen Graph-DiT for base action + scene understanding
- Trainable PPO for residual corrections

Usage:
    python scripts/graph_dit_rl/train_rsl_rl.py \
        --task SO-ARM101-Lift-Cube-ResidualRL-v0 \
        --pretrained_checkpoint ./logs/graph_dit/.../best_model.pt \
        --num_envs 256 \
        --max_iterations 300
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Residual RL Policy with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=256,
    help="Number of environments (default: 256, lower for diffusion).",
)
parser.add_argument(
    "--task",
    type=str,
    default="SO-ARM101-Lift-Cube-ResidualRL-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--pretrained_checkpoint",
    type=str,
    required=True,
    help="Path to pre-trained Graph DiT checkpoint.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=300, help="RL Policy training iterations."
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help="Save checkpoint every N iterations (overrides config).",
)
# Note: --distributed removed (not yet implemented)
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
import SO_101.tasks  # noqa: F401  # Register environments
import torch
# DEPRECATED: Old ResidualActorCritic has been removed
# from SO_101.policies.residual_rl_actor_critic import \
#     ResidualActorCritic  # noqa: F401


def main():
    """Main training function - uses standard RSL-RL flow with Hydra."""
    print("=" * 70)
    print("Residual RL Fine-Tuning with RSL-RL")
    print("=" * 70)
    print(f"Task: {args_cli.task}")
    print(f"Pretrained checkpoint: {args_cli.pretrained_checkpoint}")
    print(f"Num envs: {args_cli.num_envs} (lower for diffusion to save memory)")
    print("=" * 70)

    # ✅ CRITICAL: Verify checkpoint exists before proceeding
    if not os.path.exists(args_cli.pretrained_checkpoint):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args_cli.pretrained_checkpoint}\n"
            f"Please provide a valid path to a pre-trained Graph DiT checkpoint."
        )
    print(f"[Train] ✅ Verified checkpoint exists: {args_cli.pretrained_checkpoint}")

    # Set pretrained checkpoint via environment variable (as fallback)
    # Both GRAPH_DIT_PRETRAINED_CHECKPOINT and RESIDUAL_RL_PRETRAINED_CHECKPOINT
    if args_cli.pretrained_checkpoint:
        os.environ["GRAPH_DIT_PRETRAINED_CHECKPOINT"] = args_cli.pretrained_checkpoint
        os.environ["RESIDUAL_RL_PRETRAINED_CHECKPOINT"] = args_cli.pretrained_checkpoint
        print(f"[Train] Set checkpoint in environment variables (fallback)")

    # Use the STANDARD RSL-RL training flow
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    # Get task from CLI
    task_name = args_cli.task

    # Use Hydra to load config based on task
    @hydra_task_config(task_name, "rsl_rl_cfg_entry_point")
    def train_with_config(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        """Training function that gets config from Hydra."""

        # ✅ CRITICAL: Directly inject checkpoint into policy config
        # This is more reliable than environment variables (Hydra may load config before env vars are set)
        if hasattr(agent_cfg, "policy") and agent_cfg.policy is not None:
            # Try to inject into residual_rl_cfg
            if hasattr(agent_cfg.policy, "residual_rl_cfg"):
                residual_cfg = agent_cfg.policy.residual_rl_cfg
                if isinstance(residual_cfg, dict):
                    residual_cfg["pretrained_checkpoint"] = args_cli.pretrained_checkpoint
                else:
                    # It's a config object
                    residual_cfg.pretrained_checkpoint = args_cli.pretrained_checkpoint
                print(f"[Train] ✅ Injected checkpoint into policy.residual_rl_cfg")
            elif hasattr(agent_cfg.policy, "graph_dit_cfg"):
                # Fallback: try graph_dit_cfg
                graph_cfg = agent_cfg.policy.graph_dit_cfg
                if isinstance(graph_cfg, dict):
                    graph_cfg["pretrained_checkpoint"] = args_cli.pretrained_checkpoint
                else:
                    graph_cfg.pretrained_checkpoint = args_cli.pretrained_checkpoint
                print(f"[Train] ✅ Injected checkpoint into policy.graph_dit_cfg")
            else:
                print(
                    f"[Train] ⚠️  Warning: Could not find residual_rl_cfg or graph_dit_cfg in policy config."
                )
                print(f"[Train] ⚠️  Will rely on environment variable: RESIDUAL_RL_PRETRAINED_CHECKPOINT")

        # Override CLI arguments
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
            agent_cfg.num_envs = args_cli.num_envs
            print(f"[Train] ✅ Updated agent_cfg.num_envs to {args_cli.num_envs}")
            # CRITICAL: Also update policy config's num_envs for phase detection!
            # The policy config's num_envs was set in __post_init__ with the default value (1)
            # We need to update it to match the actual num_envs from CLI
            if hasattr(agent_cfg, 'policy') and agent_cfg.policy is not None:
                # Update ResidualActorCriticCfg's num_envs
                if hasattr(agent_cfg.policy, 'num_envs'):
                    agent_cfg.policy.num_envs = args_cli.num_envs
                    print(f"[Train] ✅ Updated policy.num_envs to {args_cli.num_envs}")
                else:
                    # If num_envs doesn't exist, add it
                    agent_cfg.policy.num_envs = args_cli.num_envs
                    print(f"[Train] ✅ Set policy.num_envs to {args_cli.num_envs}")
                # Also check residual_rl_cfg (though it shouldn't have num_envs)
                if hasattr(agent_cfg.policy, 'residual_rl_cfg'):
                    # Remove num_envs from residual_rl_cfg if it exists (it shouldn't)
                    if hasattr(agent_cfg.policy.residual_rl_cfg, 'num_envs'):
                        delattr(agent_cfg.policy.residual_rl_cfg, 'num_envs')
                        print(f"[Train] ✅ Removed num_envs from residual_rl_cfg")
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.save_interval is not None:
            agent_cfg.save_interval = args_cli.save_interval
            print(f"[Train] Set save_interval: {args_cli.save_interval}")

        # ✅ Complete seed setup (torch, numpy, cuda)
        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed
            env_cfg.seed = args_cli.seed
            torch.manual_seed(args_cli.seed)
            import numpy as np

            np.random.seed(args_cli.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args_cli.seed)
            print(f"[Train] Set random seed: {args_cli.seed} (torch, numpy, cuda)")

        # Create environment
        env = gym.make(
            task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
        )

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

        # Ensure SO_101 is available in namespace for RSL-RL's eval() calls
        # RSL-RL uses eval() internally to dynamically load custom ActorCritic classes.
        import builtins
        import sys

        # DEPRECATED: Old ResidualActorCritic has been removed
        # import SO_101.policies.residual_rl_actor_critic  # noqa: F401

        # Inject SO_101 into builtins so eval() can access it
        if not hasattr(builtins, "SO_101"):
            builtins.SO_101 = sys.modules["SO_101"]

        # Create RSL-RL runner (standard way)
        print(f"\n[Train] Creating RSL-RL runner...")
        print(f"[Train] Policy class: {agent_cfg.policy.class_name}")

        # Safe device selection with fallback
        device = getattr(args_cli, "device", None) or agent_cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Train] Using device: {device}")

        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=device
        )

        # Resume from checkpoint if specified
        if args_cli.resume:
            if not os.path.exists(args_cli.resume):
                raise FileNotFoundError(f"Resume checkpoint not found: {args_cli.resume}")
            print(f"[Train] Resuming from checkpoint: {args_cli.resume}")
            runner.load(args_cli.resume)

        # Print model info
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        total_params = sum(p.numel() for p in policy_nn.parameters())
        trainable_params = sum(
            p.numel() for p in policy_nn.parameters() if p.requires_grad
        )
        print(f"[Train] Total parameters: {total_params:,}")
        print(f"[Train] Trainable parameters: {trainable_params:,}")
        print(f"[Train] Frozen parameters: {total_params - trainable_params:,}")
        print(f"[Train] Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        # Dump configuration
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # ✅ Print training configuration summary
        print(f"\n[Train] === Training Configuration Summary ===")
        print(f"  Task: {task_name}")
        print(f"  Num envs: {env_cfg.scene.num_envs}")
        print(f"  Max iterations: {agent_cfg.max_iterations}")
        print(f"  Save interval: {getattr(agent_cfg, 'save_interval', 'N/A')}")
        print(f"  Device: {device}")
        print(f"  Seed: {getattr(agent_cfg, 'seed', 'N/A')}")
        print(f"  Pretrained checkpoint: {args_cli.pretrained_checkpoint}")
        if args_cli.resume:
            print(f"  Resume from: {args_cli.resume}")
        print(f"  Log directory: {log_dir}")
        print(f"=" * 70)

        # Run training
        print(
            f"\n[Train] Starting training with {env_cfg.scene.num_envs} parallel environments..."
        )
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
        )

        # Close environment
        env.close()

    # Call the Hydra-decorated function
    # This will automatically load the config from environment registration
    train_with_config()


if __name__ == "__main__":
    main()
    simulation_app.close()
