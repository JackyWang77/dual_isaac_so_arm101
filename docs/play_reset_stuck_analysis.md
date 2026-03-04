# Reset 后仍然卡住：可能原因与排查

## 1. 已做的修复（当前逻辑）

- **last_action_all**：reset 后下一步把该 env 的 `last_action_all` 置 0，避免上一 episode 的极端动作继续影响。
- **Node/joint/action history**：reset 时用**当前 obs**（认为是 post-reset）做 pre-fill，不再用零。
- **Action buffer**：reset 时 clear，下一步会触发 replan。
- **EMA**：reset 后下一步用 `ema_needs_init`，不再和 0 做平滑。

所以从「我们这边」看：buffer、reset 后预填、初始位置（pre-fill 用的 obs）都是一致的。若仍卡住，更可能是下面几类原因。

---

## 2. 可能原因一：env 返回的 obs 不是「真正 reset 后」的

**逻辑**：`obs, rewards, terminated, truncated, info = env.step(...)` 当 `done=True` 时，**有的实现会返回「终止瞬间」的 obs，而不是 reset 后的新 episode 首帧**。若 Isaac Lab 也是这样，我们就会：

- 用「卡住那一帧」的 obs 去 pre-fill node/joint/action history；
- 下一步 policy 看到的仍是「卡住状态 + last_action_all=0」；
- 模型可能继续输出「维持/小幅修正」的动，看起来就像 reset 后还卡着。

**和旧版的差别**：旧版 (d974c4c) **没有**用当前 obs 做 pre-fill，reset 后 history 是 **zero**。所以：

- 旧版：reset 后 = obs 可能是错的（终端态），但 history 全是 0；
- 现在：reset 后 = 若 obs 是错的，我们还会用这份错的 obs 去 pre-fill，把「卡住状态」写进 history。

也就是说：**若 env 在 done 时返回的是终端 obs 而不是 reset obs，当前版本反而会「放大」问题**（把错误状态写满 history），而旧版因为 history 是零，没有把终端态写进去。

**建议排查**：

- 在 done 分支里、pre-fill 之前，打印 reset 后**关节位置**（例如 `left_joint_pos`, `right_joint_pos` 从当前 `obs` 解析）。
- 若 reset 后应为 (0,0,0,0,0,0.4) 每臂，但打印出来是别的（例如卡住时的角度），就可以确认 **obs 在 done 时不是 post-reset**。
- 若确认如此，有两种改法：
  - 若 Isaac 提供「取当前 obs」的 API，在 done 分支里先触发 reset 再取一次 obs 用于 pre-fill；或
  - 在 done 分支**不做 pre-fill**，只 clear buffer、设 `_env_just_reset`，下一步用「全零 history」跑一步（和旧版一致），再依赖后续 step 自然把 history 填满。

---

## 3. 可能原因二：`obs` 是共享 buffer，被下一步覆盖

**逻辑**：若 `obs` 是 env 内部 buffer 的引用，`env.step()` 返回后，下一轮循环开头可能**再次**用同一个 buffer 被更新成「下一步」的 obs。那样在 done 分支里拿到的 `obs` 理论上仍是「刚 reset 完」的那一帧；但若存在多 env 或异步，需要确认：**对刚 reset 的 env，我们读到的 obs 是否一定来自该 env 的 reset 后状态**。

**建议排查**：在 done 分支里对刚 reset 的 env 立刻从 `obs` 解析并打印 joint pos；下一步（下一次循环开头）再打印一次同一 env 的 joint pos。若两次完全一致且都像 reset（0,0,0,0,0,0.4），则排除「被覆盖」问题。

---

## 4. 可能原因三：last_action_all 的写入时机/对象

**逻辑**：我们在「构建 `obs_tensor` 之后」对 `obs_tensor[env_id, la_off:la_off+la_dim] = 0` 做清零，**没有**改 env 内部 buffer。所以：

- 若 env 在 **reset 时** 会清空自己的 `last_action_all`，那我们只是「再清一次」，没问题；
- 若 env **从不**清空 `last_action_all`，那我们只在「下一步」用 `obs_tensor` 时把它置 0，**policy 看到的是 0**，也对。

唯一要确认的是：**清零的是「当前这一步」要喂给 policy 的 `obs_tensor`**，且 `_env_just_reset[i]` 只在 reset 后下一步为 True。当前实现是这样，所以这一块逻辑没问题，除非有别的代码路径在 reset 后仍用「未清零」的 obs。

---

## 5. 可能原因四：旧版为什么「没发现」或「不明显」

- **旧版**：reset 后 node/joint history **全零**，action buffer clear，**没有** last_action_all 显式清零，**没有**用当前 obs 做 pre-fill。
- 若 env 在 done 时返回的**已经是 post-reset obs**：
  - 旧版下一步：obs = reset 态，history = 0，last_action_all = 上一 episode 的极端动作。模型可能更多依赖 obs（reset 态），输出一个「从 reset 开始」的动，所以有时能跑开；
  - 若 env 在 done 时返回的是**终端 obs**：
  - 旧版下一步：obs = 卡住态，history = 0，last_action_all = 极端动作。模型看到「卡住 + 极端 last action」，也可能继续卡；但 history 是 0，至少没有把「卡住态」复制 4 份写进 history。
- **当前版**：若 env 返回的是**终端 obs**，我们用这份 obs 预填了 history，policy 会看到「4 帧都是卡住态」，更容易一直输出维持卡住的策略，所以「reset 后还卡」会更明显。

因此：**不是旧版「没问题」，而是旧版在「env 返回终端 obs」的情况下，没有用错误状态填满 history，问题可能被部分掩盖或表现不同。**

---

## 6. 建议的验证步骤（按顺序）

1. **确认 post-reset obs**  
   在 done 分支里，对刚 reset 的 env，从当前 `obs` 解析并打印：
   - `left_joint_pos`, `right_joint_pos`  
   期望：每臂 (0,0,0,0,0,0.4) 左右。若明显是「卡住」时的角度，说明 **done 时返回的不是 reset 后 obs**。

2. **若 step 1 成立（obs 不对）**  
   - 方案 A：在 done 分支**不做** pre-fill，只 clear buffer + 设 `_env_just_reset`，让下一步用「全零 history + 当前 obs」跑一步（和旧版一致），看是否还卡；
   - 方案 B：查 Isaac Lab 文档/源码，确认 `step()` 在 done 时返回的到底是 terminal 还是 post-reset obs；若有 API 能显式取「当前帧 obs」，在 reset 后调一次再 pre-fill。

3. **若 step 1 不成立（obs 正确）**  
   则问题更可能是：**模型在「正确 reset 状态」下仍然输出会再次卡住的动**（策略或数据问题），而不是 buffer/reset/初始位置逻辑错误。可再查：
   - 同一 checkpoint 在「一直不 reset、单 episode」下是否也会卡在同一姿态；
   - 换 checkpoint / 更多数据再训，看是否仍复现。

---

## 7. 代码里的假设（需要验证）

当前实现**假定**：`env.step()` 在 `done=True` 时返回的 `obs` 是 **reset 后的新 episode 首帧**。

- [play.py 1505](scripts/graph_unet/play.py)：注释「Isaac Lab done 后 obs 为 reset 态」
- [play.py 1568–1570](scripts/graph_unet/play.py)：注释「`obs` is the post-reset observation from env.step() auto-reset」
- [train_graph_rl.py 487](scripts/graph_dit_rl/train_graph_rl.py)、[play_graph_rl.py 310](scripts/graph_dit_rl/play_graph_rl.py)：同样假定 next_obs 对 done env 是 reset 后 obs

**已有 RESET DEBUG**（episode_count < 15）：打印 `reset_cube1`、`reset_cube2`（来自 `_obs_to_tensor(obs)`），用于确认场景是否重置。  
**尚未打印**：reset 后**关节位置**（`left_joint_pos` / `right_joint_pos` 或 `joint_pos`）。若 env 在 done 时返回的是终端帧，机器人关节在 obs 里可能仍是「卡住」时的角度，而 cube 可能已被 reset，单看 cube 会误以为整帧都是 reset 态。因此**最直接的验证**是：在 RESET DEBUG 里同时打印该 env 的关节位置，期望每臂约 `(0,0,0,0,0,0.4)`；若明显是其它角度，即可认定 done 时返回的不是 reset 后 obs。

---

## 8. 小结

- **Buffer、reset 后 clear、初始位置（pre-fill 用的内容）**：当前实现是一致的；若 obs 正确，这套逻辑不会导致「reset 后还卡」。
- **最可疑**：**done 时拿到的 `obs` 是否真是「reset 后」的**；若不是，当前 pre-fill 会把错误状态写进 history，反而比旧版更容易卡。
- **建议**：在 RESET DEBUG 中增加**关节位置**打印（用 `_obs_offsets` 取 `left_joint_pos`/`right_joint_pos` 或 `joint_pos`），跑几轮看 reset 后是否为初始姿态；再根据结果决定是改 pre-fill 策略、查 env 文档/源码，或从策略/数据侧排查。
