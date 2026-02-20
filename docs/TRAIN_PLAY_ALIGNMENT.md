# Train vs Play 输入输出对齐检查

## 1. Observation (obs) - 64 dim

### Train (HDF5 + obs_keys)
- **顺序**: left_joint_pos(6), left_joint_vel(6), right_joint_pos(6), right_joint_vel(6),
  left_ee_position(3), left_ee_orientation(4), right_ee_position(3), right_ee_orientation(4),
  cube_1_pos(3), cube_1_ori(4), cube_2_pos(3), cube_2_ori(4), last_action_all(12)
- **归一化**: (obs - obs_mean) / obs_std
- **来源**: HDF5 obs 按 obs_keys 顺序拼接

### Play (Env policy obs)
- **顺序**: 与 cube_stack_env_cfg PolicyCfg 一致，同上
- **归一化**: (obs - obs_mean) / obs_std，使用 checkpoint 中 obs_stats
- **来源**: env.reset() / env.step() 返回 obs["policy"]

**结论**: ✅ 一致，obs 顺序一致，归一化一致

---

## 2. Action - 12 dim

### Train (action target)
- **顺序**: left_joint_pos(6) + right_joint_pos(6) = [left_6, right_6]
- **来源**: 用 joint_pos[t+5] 替代原始 actions，来自 obs 的 left/right_joint_pos
- **归一化**: (action - action_mean) / action_std

### Play (policy output)
- **顺序**: 模型输出 [left_6, right_6]（与 train 一致）
- **归一化**: 先归一化输入，输出后反归一化

### Env (期望的 action)
- **顺序**: [right_6, left_6]（ActionsCfg: right_arm_action, right_gripper, left_arm_action, left_gripper）

**结论**: ⚠️ **顺序不一致** - 需在 env.step 前重排: policy [left_6, right_6] → env [right_6, left_6]

**修复**: play.py 中已添加 `action_for_env = torch.cat([actions[:, 6:12], actions[:, 0:6]], dim=1)`

---

## 3. Node features (4-node: left_ee, right_ee, cube_1, cube_2)

### Train
- **顺序**: node_configs 顺序，left_ee_position(3)+ori(4), right_ee..., cube_1..., cube_2...
- **归一化**: type 0 (ee) 用 ee_mean/std, type 1 (object) 用 object_mean/std
- **来源**: obs_key_offsets 从 HDF5 动态计算

### Play
- **顺序**: STACK_OBS_OFFSETS 硬编码，与 obs 布局一致
- **归一化**: 同上，用 checkpoint 的 node_stats
- **来源**: obs_tensor 按 indices 提取

**obs 布局 (64 dim)**: 0-5 left_jp, 6-11 left_jv, 12-17 right_jp, 18-23 right_jv,
24-26 left_ee_pos, 27-30 left_ee_ori, 31-33 right_ee_pos, 34-37 right_ee_ori,
38-40 cube_1_pos, 41-44 cube_1_ori, 45-47 cube_2_pos, 48-51 cube_2_ori, 52-63 last_action

**结论**: ✅ 一致，STACK_OBS_OFFSETS 与 obs 布局匹配

---

## 4. Joint states (position only)

### Train
- **顺序**: left_joint_pos(6) + right_joint_pos(6) = 12 dim
- **归一化**: (joint - joint_mean) / joint_std

### Play
- **顺序**: _extract_joint_states_from_obs 取 obs[:, 0:6] + obs[:, 12:18] = 12 dim
- **归一化**: 同上

**结论**: ✅ 一致

---

## 5. Action history

### Train
- **顺序**: 与 action 一致 [left_6, right_6]
- **归一化**: 与 action 一致
- **padding**: 不足时用 actions[0] 重复

### Play
- **顺序**: 与 policy output 一致 [left_6, right_6]（buffer 存的是 policy 输出）
- **归一化**: 存 normalized actions
- **padding**: 初始为 zeros

**结论**: ✅ 一致（buffer 存的是 policy 输出，未重排前的 order）

---

## 6. last_action_all (obs 中)

- **Train**: HDF5 中来自录制时 env 的 last_action，应为 [right_6, left_6]
- **Play**: env 的 obs["policy"] 中 last_action_all，应为 [right_6, left_6]
- **注意**: 模型不直接使用 last_action_all（用 node_histories + action_history）

---

## 总结

| 项目 | 状态 | 备注 |
|------|------|------|
| obs 顺序 | ✅ | 64 dim 一致 |
| obs 归一化 | ✅ | 用 checkpoint stats |
| action 顺序 | ✅ 已修复 | 需 env 前重排 left↔right |
| action 归一化 | ✅ | 用 checkpoint stats |
| node 提取 | ✅ | STACK_OBS_OFFSETS 正确 |
| node 归一化 | ✅ | ee/object 分 type |
| joint states | ✅ | 12 dim left+right |
| action history | ✅ | 存 policy output 顺序 |
