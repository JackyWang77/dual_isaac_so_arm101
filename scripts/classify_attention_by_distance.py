#!/usr/bin/env python3
"""Filter successful episodes and classify records by distance into 4 phases.

Usage:
    python scripts/classify_attention_by_distance.py attention_records.json -o classified.json
    python scripts/classify_attention_by_distance.py attention_records.json -o classified.json --eps_xy 0.06
"""
import argparse
import json
import math

# 与 cube_stack_env_cfg / play 一致
PICK_TARGET_XY = (0.1623, -0.023)
PICK_TARGET_Z = 0.006
PICK_EPS_XY = 0.0603
PICK_EPS_Z = 0.002
STACK_EXPECTED_HEIGHT = 0.018
STACK_EPS_Z = 0.001
STACK_EPS_XY = 0.0073


def classify_phase(obs: dict) -> str | None:
    """根据 obs 中的位置判断 phase。返回 left_pick | release | right_pick | right_stack | None"""
    c1 = obs.get("cube_1_pos")
    c2 = obs.get("cube_2_pos")
    right_ee = obs.get("right_ee_position")
    if not c1 or not c2:
        return None
    c1, c2 = [x if len(x) >= 3 else x + [0] * (3 - len(x)) for x in [c1, c2]]
    target_xy = PICK_TARGET_XY
    at_xy_1 = math.hypot(c1[0] - target_xy[0], c1[1] - target_xy[1]) < PICK_EPS_XY
    at_z_1 = abs(c1[2] - PICK_TARGET_Z) < PICK_EPS_Z
    at_xy_2 = math.hypot(c2[0] - target_xy[0], c2[1] - target_xy[1]) < PICK_EPS_XY
    at_z_2 = abs(c2[2] - PICK_TARGET_Z) < PICK_EPS_Z
    pick_cube = (at_xy_1 and at_z_1) or (at_xy_2 and at_z_2)
    z_ok_a = abs((c1[2] - c2[2]) - STACK_EXPECTED_HEIGHT) < STACK_EPS_Z
    xy_ok_a = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < STACK_EPS_XY
    z_ok_b = abs((c2[2] - c1[2]) - STACK_EXPECTED_HEIGHT) < STACK_EPS_Z
    xy_ok_b = math.hypot(c2[0] - c1[0], c2[1] - c1[1]) < STACK_EPS_XY
    stack_cube = (z_ok_a and xy_ok_a) or (z_ok_b and xy_ok_b)

    if not pick_cube:
        return "left_pick"
    if stack_cube:
        return "right_stack"
    # pick_cube & ~stack_cube
    if right_ee and len(right_ee) >= 3:
        right_at_c2_xy = math.hypot(right_ee[0] - c2[0], right_ee[1] - c2[1]) < PICK_EPS_XY
        right_at_c2_z = abs(right_ee[2] - c2[2]) < PICK_EPS_Z
        if right_at_c2_xy and right_at_c2_z:
            return "right_pick"
    return "release"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="attention_records.json (raw, with obs)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON (classified, success only)")
    parser.add_argument("--eps_xy", type=float, default=None, help="Override PICK_EPS_XY")
    parser.add_argument("--eps_z", type=float, default=None, help="Override PICK_EPS_Z")
    args = parser.parse_args()

    global PICK_EPS_XY, PICK_EPS_Z
    if args.eps_xy is not None:
        PICK_EPS_XY = args.eps_xy
    if args.eps_z is not None:
        PICK_EPS_Z = args.eps_z

    with open(args.input) as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = [records]

    success_records = [r for r in records if r.get("success", False)]
    # Group by (env_id, episode_id)
    from collections import defaultdict
    by_ep = defaultdict(list)
    for r in success_records:
        key = (r["env_id"], r["episode_id"])
        by_ep[key].append(r)
    for key in by_ep:
        by_ep[key].sort(key=lambda x: x["step"])

    # For each episode, pick first record per phase
    classified = []
    for (env_id, episode_id), recs in sorted(by_ep.items()):
        phases = {}
        for r in recs:
            phase = classify_phase(r.get("obs", {}))
            if phase and phase not in phases:
                phases[phase] = r["attention"]
        if phases:
            classified.append({
                "env_id": env_id,
                "episode_id": episode_id,
                "phases": phases,
                "node_names": recs[0].get("node_names", ["left_ee", "right_ee", "cube_1", "cube_2"]),
                "n_records": len(recs),
            })

    out = args.output or args.input.replace(".json", "_classified.json")
    with open(out, "w") as f:
        json.dump(classified, f, indent=2)
    print(f"Filtered {len(success_records)} success records -> {len(classified)} episodes, saved to {out}")


if __name__ == "__main__":
    main()
