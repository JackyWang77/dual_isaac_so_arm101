# Train vs Play 输入对齐检查

## 1. Obs 向量顺序（已对齐）

- **Train**：`stack/train_graph_unet.sh` 里  
  `OBS_KEYS='["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel","left_ee_position","left_ee_orientation","right_ee_position","right_ee_orientation","cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori","last_action_all"]'`  
  Dataset 按此顺序拼成 `[T, 64]`，并据此算 `obs_key_offsets` / `obs_key_dims`，写入 checkpoint。

- **Play**：`play.py` 里当 `obs["policy"]` 为 dict 时，用  
  `_dual_arm_policy_order = ["left_joint_pos", "left_joint_vel", "right_joint_pos", "right_joint_vel", "left_ee_position", "left_ee_orientation", "right_ee_position", "right_ee_orientation", "cube_1_pos", "cube_1_ori", "cube_2_pos", "cube_2_ori", "last_action_all"]`  
 按此顺序从 dict 里取 `keys_order = [k for k in _dual_arm_policy_order if k in obs_val]` 再 concat。

两处顺序一致，且 play 显式按该顺序拼接，不依赖 env 的 dict 迭代顺序。只要 env 提供的 key 齐全，obs 维度和各 key 的索引与 train 一致。

**需确认**：Play 用的 env（`SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0`）的 policy obs 是否包含以上全部 13 个 key。若少一个（例如没有 `last_action_all`），concat 后维度会小于 64，而 checkpoint 的 `obs_key_offsets` 是按 64 维算的，会导致后续所有索引错位。

---

## 2. 若 env 返回的是已拼接的 Tensor（未走 dict 分支）

当 `obs["policy"]` 是 **Tensor**（例如某配置下 `concatenate_terms=True`）时，play 会直接用该 tensor，**不再**按 `_dual_arm_policy_order` 重排。此时 obs 顺序完全由 Isaac Lab 的 observation 拼接顺序决定。若该顺序与 train 的 OBS_KEYS 不一致，就会出现 train/play 不对齐。

当前 Stack Play 用的是 `DualCubeStackJointStatesMimicEnvCfg_PLAY`，其中 `observations.policy.concatenate_terms = False`，因此是 dict 分支，顺序由 play 显式控制，与 train 一致。

---

## 3. Node / history 与归一化

- **Node_configs**：train 和 play 都用同一套 `node_configs`（left_ee, right_ee, cube_1, cube_2），play 从 policy.cfg 读，checkpoint 里保存的与 train 一致。
- **Node 特征**：play 用 `_extract_node_features_multi(obs_tensor)`，按 `_obs_offsets`（来自 checkpoint）和 `node_configs` 的 pos_key/ori_key 切片，与 train 里用 `obs_key_offsets` 建 node history 的方式一致。
- **History 长度与 padding**：train 对早期步用 `actions[0]` 和首帧 obs 做 padding；play 在 reset 后用当前帧 prefill。逻辑一致。
- **归一化**：obs、node（ee/object mean&std）、joint、action 的 mean/std 都从 checkpoint 读，在 play 里按相同顺序应用，与 train 一致。

只要 obs 的 64 维顺序和 key 齐全性没问题，这些部分就是对上的。

---

## 4. 可能的不对齐来源（建议排查）

| 项 | 说明 |
|----|------|
| **Env 少 key** | 若 env 的 policy obs 缺少上述任一 key，`keys_order` 会变短，concat 后不是 64 维，`obs_key_offsets` 全部错位。可在 play 里打印 `obs_tensor.shape[1]` 和 `keys_order`，确认是否为 64 且 13 个 key 都在。 |
| **Dataset 录制时 obs 顺序不同** | 若 HDF5 是用另一套 obs 顺序或另一 env 录的，而 train 用的 `obs_keys` 与录制时不一致，则 `obs_key_offsets` 与真实数据不对应；用该 checkpoint 在 play 里也会错。需确认录制脚本写入的 obs 键与 train OBS_KEYS 一致且顺序一致。 |
| **Policy 用的是 obs_structure 而非 offsets** | 若 policy 内部用 `obs_structure`（start,end）做切片，需保证与 checkpoint 的 `obs_key_offsets`/dims 一致；play 侧只用 checkpoint 的 offsets，若训练时用了别的 structure 会不一致。当前 train 从 dataset 的 obs_key_offsets 建 obs_structure 并传给 policy，play 用同一 checkpoint 的 offsets，理论一致。 |

---

## 5. 建议的快速验证

在 play 里（在构建 `obs_tensor` 之后、第一次用 `_obs_offsets` 之前）加一次校验：

- 若 checkpoint 有 `obs_key_offsets`：根据各 key 的 offset+dim 算得总维数，应等于 `obs_tensor.shape[1]`（64）。
- 打印前 2 个 step 的 `obs_tensor.shape` 和（若为 dict 分支）`len(keys_order)`、是否包含 `last_action_all`。

若 64 维且 13 个 key 都在，则 train 与 play 的 obs 输入是对齐的；若 policy 仍表现异常，再查 node 归一化、subtask、或策略本身。
