#!/usr/bin/env python3
"""
Analyze mapping between action[5]/action[11] (gripper from teleop) and joint_pos[5] (gripper from obs).

Recording: teleop sends action to env.step(); recorder stores that action.
Training: replaces action with joint_pos[t+5] from obs.
Play: model outputs joint_pos (from joint_pos[t+5] training target).

This script compares:
- action[5], action[11] (raw from HDF5 = what teleop sent)
- right_joint_pos[5], left_joint_pos[5] (from obs = sim state)
- joint_pos[t+5] (what train uses as target)

To find: joint[5] -> action[5] mapping for play (if env expects action format).
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze action[5] vs joint_pos[5] mapping")
    parser.add_argument("--dataset", type=str, default="./datasets/dual_cube_stack_annotated_dataset.hdf5")
    parser.add_argument("--max_demos", type=int, default=50, help="Max demos to analyze")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        print("Need h5py: pip install h5py")
        return

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        demo_keys = demo_keys[: args.max_demos]

        all_action_r5 = []
        all_action_l5 = []
        all_joint_r5 = []
        all_joint_l5 = []
        all_joint_r5_t5 = []  # joint_pos[t+5] (train target)
        all_joint_l5_t5 = []
        all_action_r5_vs_joint_r5 = []  # (action[t][5], joint[t][5])
        all_action_r5_vs_joint_r5_t5 = []  # (action[t][5], joint[t+5][5])

        for dk in demo_keys:
            demo = f[f"data/{dk}"]
            obs = demo.get("obs", demo.get("observations", None))
            if obs is None:
                continue
            actions = np.array(demo["actions"])
            right_jp = np.array(obs["right_joint_pos"])
            left_jp = np.array(obs["left_joint_pos"])
            T = len(actions)

            # action[5]=right gripper, action[11]=left gripper
            # right_joint_pos[5]=right gripper, left_joint_pos[5]=left gripper
            for t in range(T):
                all_action_r5.append(actions[t, 5])
                all_action_l5.append(actions[t, 11])
                all_joint_r5.append(right_jp[t, 5])
                all_joint_l5.append(left_jp[t, 5])
                t5 = min(t + 5, T - 1)
                all_joint_r5_t5.append(right_jp[t5, 5])
                all_joint_l5_t5.append(left_jp[t5, 5])
                all_action_r5_vs_joint_r5.append((actions[t, 5], right_jp[t, 5]))
                all_action_r5_vs_joint_r5_t5.append((actions[t, 5], right_jp[t5, 5]))

        all_action_r5 = np.array(all_action_r5)
        all_action_l5 = np.array(all_action_l5)
        all_joint_r5 = np.array(all_joint_r5)
        all_joint_l5 = np.array(all_joint_l5)
        all_joint_r5_t5 = np.array(all_joint_r5_t5)
        all_joint_l5_t5 = np.array(all_joint_l5_t5)

        print("=" * 70)
        print("action[5] vs joint_pos[5] 映射统计 (Dual Cube Stack)")
        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Demos: {len(demo_keys)}")
        print(f"Total steps: {len(all_action_r5)}")
        print()

        print("--- action[5] (right gripper, 录制时 teleop 发送的) ---")
        print(f"  min={all_action_r5.min():.4f}, max={all_action_r5.max():.4f}")
        print(f"  mean={all_action_r5.mean():.4f}, std={all_action_r5.std():.4f}")
        print()

        print("--- action[11] (left gripper) ---")
        print(f"  min={all_action_l5.min():.4f}, max={all_action_l5.max():.4f}")
        print(f"  mean={all_action_l5.mean():.4f}, std={all_action_l5.std():.4f}")
        print()

        print("--- right_joint_pos[5] (obs 中的关节状态) ---")
        print(f"  min={all_joint_r5.min():.4f}, max={all_joint_r5.max():.4f}")
        print(f"  mean={all_joint_r5.mean():.4f}, std={all_joint_r5.std():.4f}")
        print()

        print("--- left_joint_pos[5] ---")
        print(f"  min={all_joint_l5.min():.4f}, max={all_joint_l5.max():.4f}")
        print(f"  mean={all_joint_l5.mean():.4f}, std={all_joint_l5.std():.4f}")
        print()

        print("--- joint_pos[t+5] (训练时用的 target, 即 model 学的是这个) ---")
        print("  Right gripper:")
        print(f"    min={all_joint_r5_t5.min():.4f}, max={all_joint_r5_t5.max():.4f}")
        print(f"    mean={all_joint_r5_t5.mean():.4f}, std={all_joint_r5_t5.std():.4f}")
        print("  Left gripper:")
        print(f"    min={all_joint_l5_t5.min():.4f}, max={all_joint_l5_t5.max():.4f}")
        print(f"    mean={all_joint_l5_t5.mean():.4f}, std={all_joint_l5_t5.std():.4f}")
        print()

        # Correlation: action[t][5] vs joint[t][5]
        corr_same = np.corrcoef(all_action_r5, all_joint_r5)[0, 1]
        corr_t5 = np.corrcoef(all_action_r5, all_joint_r5_t5)[0, 1]
        print("--- 相关性 ---")
        print(f"  action[t][5] vs joint[t][5]:     corr = {corr_same:.4f}")
        print(f"  action[t][5] vs joint[t+5][5]:  corr = {corr_t5:.4f}")
        print()

        # Difference stats
        diff_same = all_action_r5 - all_joint_r5
        diff_t5 = all_action_r5 - all_joint_r5_t5
        print("--- action[5] - joint_pos[5] 差值 ---")
        print(f"  action[t] - joint[t]:     mean={diff_same.mean():.4f}, std={diff_same.std():.4f}, max_abs={np.abs(diff_same).max():.4f}")
        print(f"  action[t] - joint[t+5]:   mean={diff_t5.mean():.4f}, std={diff_t5.std():.4f}, max_abs={np.abs(diff_t5).max():.4f}")
        print()

        # Percentiles for threshold
        print("--- joint_pos[5] 分位数 (用于 threshold 参考) ---")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            v = np.percentile(all_joint_r5_t5, p)
            print(f"  p{p:2d}: {v:.4f}")
        print()

        # Linear fit: joint[5] -> action[5] (用 joint 预测 action，供 play 时 model 输出 joint 后映射)
        from numpy.linalg import lstsq
        # 分别拟合
        ones_r = np.ones_like(all_joint_r5_t5)
        X_r = np.column_stack([all_joint_r5_t5, ones_r])
        coeff_r, _, _, _ = lstsq(X_r, all_action_r5, rcond=None)
        ones_l = np.ones_like(all_joint_l5_t5)
        X_l = np.column_stack([all_joint_l5_t5, ones_l])
        coeff_l, _, _, _ = lstsq(X_l, all_action_l5, rcond=None)

        # 合并左右手数据，拟合统一系数（左右手硬件相同，映射应一致）
        all_joint = np.concatenate([all_joint_r5_t5, all_joint_l5_t5])
        all_action = np.concatenate([all_action_r5, all_action_l5])
        ones_comb = np.ones_like(all_joint)
        X_comb = np.column_stack([all_joint, ones_comb])
        coeff_comb, _, _, _ = lstsq(X_comb, all_action, rcond=None)

        print("--- 线性映射 joint[5] -> action[5] (最小二乘拟合) ---")
        print("  分别拟合 (左右手数据分布不同导致系数不同):")
        print("    Right: action = {:.4f} * joint + {:.4f}".format(coeff_r[0], coeff_r[1]))
        print("    Left:  action = {:.4f} * joint + {:.4f}".format(coeff_l[0], coeff_l[1]))
        print("  统一拟合 (左右手用相同映射，更合理):")
        print("    action = {:.4f} * joint + {:.4f}".format(coeff_comb[0], coeff_comb[1]))
        pred_comb_r = coeff_comb[0] * all_joint_r5_t5 + coeff_comb[1]
        pred_comb_l = coeff_comb[0] * all_joint_l5_t5 + coeff_comb[1]
        mae_comb_r = np.abs(pred_comb_r - all_action_r5).mean()
        mae_comb_l = np.abs(pred_comb_l - all_action_l5).mean()
        print(f"  统一映射 MAE: Right={mae_comb_r:.4f}, Left={mae_comb_l:.4f}")
        print()

        # Threshold for binary: joint_pos 分位数
        # 闭合时 joint 更负，张开时 joint 更大
        closed_vals = all_joint_r5_t5[all_action_r5 < -0.3]  # action 负 = 闭合?
        open_vals = all_joint_r5_t5[all_action_r5 > 0.3]  # action 正 = 张开?
        if len(closed_vals) > 10 and len(open_vals) > 10:
            print("--- 按 action 正负分组的 joint 分布 (右臂) ---")
            print(f"  action<0 (闭合): joint mean={closed_vals.mean():.4f}, std={closed_vals.std():.4f}")
            print(f"  action>0 (张开): joint mean={open_vals.mean():.4f}, std={open_vals.std():.4f}")
            mid = (closed_vals.mean() + open_vals.mean()) / 2
            print(f"  建议 threshold (joint): {mid:.4f}")
        print()

        print("=" * 70)
        print("结论与建议")
        print("=" * 70)
        print("""
1. action[5] 与 joint_pos[5] 尺度不同！录制时 teleop 发 action，obs 里是 joint。
2. 训练时 target=joint_pos[t+5]，model 输出的是 joint 空间。
3. Play 时：若 env 期望 joint position，直接传 model 输出即可。
   若 env 期望 action 格式，用上面的线性映射: action = a*joint + b
4. 若用 binary threshold：用上面建议的 joint threshold 分界开/关。
""")


if __name__ == "__main__":
    main()
