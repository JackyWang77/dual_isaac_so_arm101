#!/bin/bash
# 直接运行命令 - 一键启动
# play_graph_dit_gripper.sh 内部会调用 python scripts/graph_dit/play.py
# play.py 会通过 AppLauncher 自动启动 Isaac Sim
./play_graph_dit_gripper.sh \
    --task SO-ARM101-Lift-Cube-Play-v0 \
    --checkpoint ./logs/graph_dit/lift_joint_flow_matching/2026-01-27_18-18-08/best_model.pt \
    --gripper-model ./logs/gripper_model/gripper_model.pt \
    --num-envs 2 \
    --num-episodes 10
