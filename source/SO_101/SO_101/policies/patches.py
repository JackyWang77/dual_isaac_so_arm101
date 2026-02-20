# ============================================================
# PATCH 1: graph_dit_policy.py — GraphDiTPolicyCfg 新增字段
# 在 exec_horizon 字段后面加这两行:
# ============================================================

#   arm_action_dim: int | None = None
#   """每条臂的 action 维度 (dual arm 时 = action_dim // 2 = 6)。
#   None 表示单臂模式，不启用 DualArmUnetPolicy。"""
#
#   cross_arm_heads: int = 4
#   """CrossArmAttention 的 attention heads 数量。"""


# ============================================================
# PATCH 2: graph_unet_policy.py — 文件开头 import 处新增
# ============================================================

# from SO_101.policies.dual_arm_unet_policy import DualArmUnetPolicy
# __all__ = [..., "DualArmUnetPolicy"]


# ============================================================
# PATCH 3: train.py — train_graph_unet_policy() 里
# 找到 "PolicyClass = GraphUnetPolicy if ..." 这行，替换成:
# ============================================================

TRAIN_POLICY_CLASS_PATCH = """
# 自动检测是否需要 DualArmUnetPolicy
is_dual_arm_unet = (
    policy_type == "unet"          # graph_unet 也可以改成 dual_arm，按需
    and action_dim == 12
    and node_configs is not None   # 有 4-node config 才算 dual arm
)

if is_dual_arm_unet:
    from SO_101.policies.dual_arm_unet_policy import DualArmUnetPolicy
    PolicyClass = DualArmUnetPolicy
    # 补充 cfg 字段
    cfg.arm_action_dim = action_dim // 2   # 6
    cfg.cross_arm_heads = 4
    print(f"[Train] 🤖 DualArmUnetPolicy 已启用 (arm_dim={cfg.arm_action_dim})")
elif policy_type == "graph_unet":
    PolicyClass = GraphUnetPolicy
else:
    PolicyClass = UnetPolicy
"""


# ============================================================
# PATCH 4: play.py — play_graph_unet_policy() 里
# 找到 "PolicyClass = GraphUnetPolicy if ..." 这行，替换成:
# ============================================================

PLAY_POLICY_CLASS_PATCH = """
# 从 checkpoint cfg 自动判断是否 dual arm
cfg = checkpoint.get("cfg", None)
arm_action_dim = getattr(cfg, "arm_action_dim", None) if cfg else None

if arm_action_dim is not None:
    from SO_101.policies.dual_arm_unet_policy import DualArmUnetPolicy
    PolicyClass = DualArmUnetPolicy
    print(f"[Play] 🤖 DualArmUnetPolicy 检测到 (arm_dim={arm_action_dim})")
elif policy_type == "graph_unet":
    PolicyClass = GraphUnetPolicy
else:
    PolicyClass = UnetPolicy

policy = PolicyClass.load(checkpoint_path, device=device)
"""


# ============================================================
# 训练命令示例 (dual arm pick & place)
# ============================================================
EXAMPLE_TRAIN_CMD = """
./isaaclab.sh -p scripts/graph_unet/train.py \\
    --task SO-ARM101-Pick-Place-DualArm-IK-Abs-v0 \\
    --dataset ./datasets/pick_place.hdf5 \\
    --policy_type unet \\
    --obs_keys '["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel",
                  "left_ee_position","left_ee_orientation",
                  "right_ee_position","right_ee_orientation",
                  "cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori"]' \\
    --node_configs '[
        {"name":"left_ee",  "type":0,"pos_key":"left_ee_position", "ori_key":"left_ee_orientation"},
        {"name":"right_ee", "type":0,"pos_key":"right_ee_position","ori_key":"right_ee_orientation"},
        {"name":"cube_1",   "type":1,"pos_key":"cube_1_pos",       "ori_key":"cube_1_ori"},
        {"name":"cube_2",   "type":1,"pos_key":"cube_2_pos",       "ori_key":"cube_2_ori"}
    ]' \\
    --obs_dim 64 --action_dim 12 \\
    --epochs 300 --batch_size 8 --lr 1e-4 \\
    --pred_horizon 16 --exec_horizon 8
"""
