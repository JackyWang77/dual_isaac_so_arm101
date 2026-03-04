# Disentangled 里的 Graph 流程

Disentangled 指 **DualArmDisentangledPolicy**（继承 DisentangledGraphUnetPolicy）：用「解耦」的图编码（位置/旋转分开做 attention）+ 双 U-Net 出左右臂动作。

---

## 1. 整体数据流（inference / predict）

```
play 传入: obs, node_histories [B,N,H,7], node_types, joint_states_history, subtask_condition
                    ↓
① _build_node_histories  →  nh [B, N, H, 7], nt [N]
                    ↓
② _compute_disentangled_edges(nh)  →  dist_embed, sim_embed  （边特征：距离 + 四元数相似度）
                    ↓
③ _encode_disentangled_graph(nh, dist_embed, sim_embed)  →  node_features, graph_z [B, z_dim]
                    ↓
④ z = concat(graph_z_proj(graph_z), raw_proj(nh[:,:,-1,:]))  [+ subtask]
                    ↓
⑤ 双 U-Net + 可选 CrossArmAttn  →  velocity  →  flow matching 积分  →  action [B, pred_horizon, 12]
```

- **nh**：node history，形状 `[B, N, H, 7]`，N=4（left_ee, right_ee, cube_1, cube_2），H=history_length，7=pos(3)+quat(4)。
- **graph_z**：图编码得到的全局 latent，再和「当前帧 raw」拼成 **z**，作为 U-Net 的 condition。
- **action_history** 在 forward/predict 里**没有参与** graph 或 condition，只作为接口参数（disentangled 的 U-Net 条件只有 z）。

---

## 2. ① _build_node_histories

- 若传入 **node_histories**（play 里用）：直接返回 `(node_histories, node_types)`，不再从 obs 里抽。
- 否则：用 **ee_node_history** + **object_node_history** 拼成 2 节点，或从 **obs** 里 `_extract_node_features(obs)` 得到 ee/object 单帧再 stack 成 history。

Stack 4-node 时，play 一定传 **node_histories**，所以 graph 的输入就是 play 里按 `node_configs` 顺序拼好的 `[B, 4, H, 7]`（left_ee, right_ee, cube_1, cube_2）。

---

## 3. ② _compute_disentangled_edges（v1 边）

- 输入：**nh** `[B, N, H, 7]`。
- 对所有节点对 (i,j)：
  - **距离**：`||pos_i - pos_j||` → 1 维，再经 `dist_edge_embedding` → dist_embed。
  - **相似度**：`|quat_i · quat_j|`（四元数点乘绝对值）→ 1 维，再经 `sim_edge_embedding` → sim_embed。
- 输出：
  - N=2：`dist_embed` / `sim_embed` 为 `[B, H, edge_dim]`。
  - N>2：为 `[B, N, N, H, edge_dim]`（all-pairs，对称）。

边只用于后面的 **attention bias**，不参与 value 的 gate/scale。

---

## 4. ③ _encode_disentangled_graph

- **输入**：`nh` [B, N, H, 7]，`dist_embed`，`sim_embed`。
- **步骤**：
  1. **初始嵌入**：`node_histories` 展平后过 `node_embedding`，再加 `node_type_embedding` → `node_features` [B, N, H, hidden_dim]。
  2. **多层 DisentangledGraphAttention**（通常 1 层）：
     - 把 **nh** 拆成 pos(3) / rot(4)，分别过 `pos_embed` / `rot_embed`。
     - **Pos 流**：只用 pos 的 Q/K/V，attention 上加 **dist_bias**（由 dist_embed 得到）。
     - **Rot 流**：只用 rot 的 Q/K/V，attention 上加 **sim_bias**（由 sim_embed 得到）。
     - 两流各自做 multi-head attention，输出合并再 `out_proj`，**加残差**到上一层的 node_features。
  3. 每层后接 **FFN + 残差**。
  4. **Pool**：`_pool_node_latent(node_features)`：先按时间聚合（若有 `node_temporal_aggregator`），再按 `graph_pool_mode`（mean 或 concat）聚成 `[B, z_dim]` → **graph_z**。

- **输出**：`(node_features, graph_z)`；若 `return_attention=True` 还返回最后一层 attention 权重。

---

## 5. ④ 得到 z，喂给 U-Net

- `raw_latest = nh[:, :, -1, :].reshape(B, -1)`：当前帧 4 节点共 N*7 维。
- `raw_feat = raw_proj(raw_latest)`，`graph_z_comp = graph_z_proj(graph_z)`，二者 concat 成 **z** [B, z_dim]（再可选加 subtask）。
- **DualArmDisentangledPolicy** 里左右 U-Net 共用同一个 z（`z_left = z_right = z_base`），各自再和 timestep embedding、noisy action 一起进 encode → 可选 CrossArmAttn → decode，得到 velocity，再 flow matching 积分得到 action。

---

## 6. 小结（和 RawOnly / 普通 GraphUnet 的差别）

| 项目 | Disentangled |
|------|----------------|
| 边 | v1：仅距离 + 四元数相似度，分开 embed，只做 attention bias |
| 图注意力 | 位置/旋转解耦：pos 头只看 pos+dist_bias，rot 头只看 rot+sim_bias |
| 输入 | node_histories [B,N,H,7]（play 传）；obs 仅当未传 node_histories 时用于 build |
| 条件 z | graph_z（pool 图编码）+ raw_proj(当前帧) + 可选 subtask |
| action_history | 不参与 graph，不参与 U-Net 条件 |

所以 **disentangled 的 graph 流程**就是：**node_histories → v1 边(距离+相似度) → 解耦图 attention(pos/rot 分头 + 边 bias) → pool 得 graph_z → 与 raw 拼成 z → 双 U-Net → action**。

---

## 7. Bias 怎么算、加在哪、和 attention 的关系

### 7.1 边的形状（4 节点、H=4 条历史）

- **dist_embed / sim_embed** 形状：`[B, N, N, H, edge_dim]`，例如 `[B, 4, 4, 4, 32]`。
- 含义：**只对「同一时刻」的节点对**算边。
  - `flat = node_histories.reshape(B*H, N, 7)`：按 (batch, time) 展平，每个 (b,t) 有 N 个节点。
  - 对每个 (i,j) 且 i<j，算的是 **node_i 在 t 时刻** 与 **node_j 在 t 时刻** 的 1 维距离、1 维四元数相似度。
  - 所以 `dist_embed[:, i, j, t, :]` = 在时刻 t、节点 i 与 j 之间的距离嵌入（1 维 → MLP → edge_dim 维）；sim 同理。
- 因此：**没有**「node_i 在 t1」与「node_j 在 t2」的边，只有「同一 t」的 (i,j) 边。

### 7.2 Bias 从边到 attention 的映射

- **dist**（1 维/边）→ `dist_edge_embedding` → `[B,N,N,H, edge_dim]`  
  → 在 `_build_disentangled_bias` 里对每个 (i,j) 取 `dist_edge_embed[:, i, j]` 即 `[B, H, edge_dim]`  
  → `dist_to_bias`（Linear(edge_dim, pos_heads)）→ `[B, H, pos_heads]`  
  → 填到 **pos_bias** 的 `[B, pos_heads, seq, seq]` 的 **(ni, nj)** 位置。
- **sim**（1 维/边）→ `sim_edge_embedding` → 同上形状  
  → `sim_to_bias`(Linear(edge_dim, rot_heads)) → 填到 **rot_bias** 的 **(ni, nj)** 位置。

其中 **序列下标** 是：`ni = i*H + t`，`nj = j*H + t`（即 node_i 在时刻 t 的 token 下标）。  
所以 **只有「同一时刻 t」的 (node_i, node_j) 对** 会得到非零的 dist/sim bias；  
**(node_i, t1) 与 (node_j, t2)** 在 t1≠t2 时 **没有** 这两项，只有下面的 temporal bias。

### 7.3 加到什么上

- **pos_bias** 加到 **pos 流的 attention logits** 上：`scores_pos += pos_bias`，形状都是 `[B, pos_heads, seq_len, seq_len]`。
- **rot_bias** 加到 **rot 流的 attention logits** 上：`scores_rot += rot_bias`，形状 `[B, rot_heads, seq_len, seq_len]`。
- 另外还有 **temporal_bias**：按 `t_query - t_key` 查表 `temporal_bias_embedding`，得到 `[seq, seq, num_heads]`，拆成 pos_heads/rot_heads 两半，分别加到 pos_bias 和 rot_bias 上。所以 **跨时刻** 的 attention 只有这个时间差 bias，没有 dist/sim。

### 7.4 Node attention 是「全序列」还是「每步内」？

- **序列排布**：`seq_len = N*H`，按 **先 node 再 time** 展平：  
  index = `node_id * H + t` → 0..H-1=node0 的 t=0..H-1，H..2H-1=node1 的 t=0..H-1，…
- **Attention 范围**：对 **整个 N*H 序列** 做 self-attention。  
  所以 **node1 在 step1** 会和 **node2 在 step1**、**node1 在 step2**、**node2 在 step3** 等所有位置都做 attention，不是「只在同一 history 步内」。
- **Bias 的加法**：
  - **同一时刻** 的 (i,t)↔(j,t)：加 **dist_bias**（pos 流）和 **sim_bias**（rot 流）。
  - **任意 (t1, t2)**：都加 **temporal_bias**(t1-t2)。

总结：**边 = 同一 t 的节点对，每个 (i,j,t) 一条 dist 一条 sim；bias = 这些边嵌入后加到同 t 的 (ni,nj) 的 attention logits 上；attention 本身是整条 N*H 序列，所以跨 t、跨 node 都会 attend，只是不同 t 的 pair 没有 dist/sim，只有时间差 bias。**

---

## 8. 从信息角度看：Graph 提供什么，和 Raw 的区别，注入方式

### 8.1 两路各提供什么

| 分支 | 输入 | 输出 | 信息含义 |
|------|------|------|----------|
| **Raw** | `nh[:, :, -1, :]`，即**当前帧** N 个节点的 pos+quat，展平为 [B, N×7] | `raw_feat = raw_proj(raw_latest)`，[B, z_dim/2] | **当前瞬时状态**：此刻 4 个节点各自在哪、朝哪，无历史、无显式关系。 |
| **Graph** | 整段 **node_histories** [B, N, H, 7] + 同序边 (dist/sim) | `graph_z = _encode_disentangled_graph(...)` 再 `graph_z_proj` → [B, z_dim/2] | **上下文与关系**：在 N×H 上做 attention（带同 t 的 dist/sim bias + 跨 t 的 temporal bias），再 pool。编码的是「谁和谁在何时距离/朝向如何、随时间怎么变」的压缩表示。 |

- **Raw**：只看「现在这一帧」的 N 个 7 维向量，没有时间、没有节点间关系，等价于一张**静态快照**。
- **Graph**：看**整段历史 H 步 × N 个节点**，通过 attention 和边 bias 把**时序（怎么动）**和**空间关系（谁离谁多远、朝向多像）**都压进一个向量。

### 8.2 合并后注入 UNet 的方式

- `z_base = concat(graph_z_comp, raw_feat)`，即 **z = [graph 那半 | raw 那半]**，维度 [B, z_dim]。可选再加 subtask。
- 在 **SplitConditionalUnet1D** 里，z 作为 **global_cond**：每个 block 里 `cond = concat(diffusion_step_encoder(t), global_cond)`，再和该层特征一起参与计算（concat 进 condition 维度）。**不是** FiLM 的 scale/shift，而是「条件向量里同时含有时间步 + graph 信息 + raw 信息」。

所以从信息流看：**UNet 每一层看到的 condition 里都同时包含「当前时刻的扩散步」和「场景的上下文（graph）+ 当前快照（raw）」**。

### 8.3 Graph 相对 Raw 多出来的信息（总结）

- **时序**：过去 H 步里各节点位置/朝向怎么变（attention 可跨 t，temporal bias 区分时间差）。
- **关系**：同一时刻谁和谁距离多少、朝向多像（dist/sim 只加在同 t 的节点对上）。
- **聚合方式**：不是「把历史拼成更长向量」，而是「attention + pool」得到的**压缩表示**，更偏「当前决策相关的摘要」，而不是逐帧罗列。

因此：**Raw = 当前状态；Graph = 在历史与关系上做的一次「上下文编码」；两者拼成 z 再注入 UNet，相当于让策略同时知道「现在长什么样」和「之前怎么动、谁和谁啥关系」**。
