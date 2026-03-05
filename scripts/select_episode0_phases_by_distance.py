#!/usr/bin/env python3
"""Select 4 phase frames in episode 0 by distance for attention plotting.

Phase 1: Left EE closest to cube_1 (left hand about to pick).
Phase 2: Left EE starts moving away from cube_1 (release).
Phase 3: Right EE closest to cube_2 (right hand about to pick cube_2).
Phase 4: Last frame(s) of episode (stack done).

Usage:
    python scripts/select_episode0_phases_by_distance.py --obs_records attention_output/obs_records.json
    python scripts/select_episode0_phases_by_distance.py --obs_records attention_output/obs_records.json --episode_id 0 --output attention_output/ep0_phase_indices.json
"""
import argparse
import json
import math


def dist(a, b):
    """Euclidean distance between two [x,y,z] lists."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    parser = argparse.ArgumentParser(description="Select 4 phase record indices in episode 0 by distance.")
    parser.add_argument("--obs_records", required=True, help="obs_records.json from run_attention_pipeline.sh")
    parser.add_argument("--episode_id", type=int, default=0, help="Episode to analyze (default 0)")
    parser.add_argument("--env_id", type=int, default=0, help="Env to use (default 0)")
    parser.add_argument("--release_threshold", type=float, default=0.02, help="Min distance increase to count as release (m)")
    parser.add_argument("--right_pick_exclude_last_ratio", type=float, default=0.25, help="Exclude last N%% of episode when finding right_ee–cube_2 min (so Phase 3 = pick, not stack)")
    parser.add_argument("--output", default=None, help="Write selected indices to this JSON (default: print only)")
    args = parser.parse_args()

    with open(args.obs_records) as f:
        records = json.load(f)

    # Episode 0: keep global indices
    indices_ep = [i for i, r in enumerate(records) if r.get("env_id") == args.env_id and r.get("episode_id") == args.episode_id]
    if not indices_ep:
        print(f"No records for env_id={args.env_id} episode_id={args.episode_id}")
        return

    # Compute distances for each record in episode
    steps = []
    for idx in indices_ep:
        r = records[idx]
        obs = r.get("obs", {})
        le = obs.get("left_ee_position")
        re = obs.get("right_ee_position")
        c1 = obs.get("cube_1_pos")
        c2 = obs.get("cube_2_pos")
        if not all([le, re, c1, c2]):
            continue
        d_l1 = dist(le, c1)
        d_l2 = dist(le, c2)
        d_r1 = dist(re, c1)
        d_r2 = dist(re, c2)
        steps.append({
            "global_index": idx,
            "step": r.get("step"),
            "d_left_ee_cube1": d_l1,
            "d_left_ee_cube2": d_l2,
            "d_right_ee_cube1": d_r1,
            "d_right_ee_cube2": d_r2,
        })

    if len(steps) < 4:
        print(f"Episode has only {len(steps)} records, need at least 4")
        return

    # Phase 1: first time left_ee and cube_1 are closest (argmin d_l1)
    i_min_l1 = min(range(len(steps)), key=lambda i: steps[i]["d_left_ee_cube1"])
    phase1 = steps[i_min_l1]

    # Phase 2: after phase1, first frame where left_ee moves away from cube_1 (release)
    min_d_l1 = phase1["d_left_ee_cube1"]
    phase2 = None
    for i in range(i_min_l1 + 1, len(steps)):
        if steps[i]["d_left_ee_cube1"] >= min_d_l1 + args.release_threshold:
            phase2 = steps[i]
            break
    if phase2 is None:
        phase2 = steps[min(i_min_l1 + 1, len(steps) - 1)]

    # Phase 3: right_ee and cube_2 closest *before* stack — exclude last N% of episode so we get "right pick" not "stacked"
    n_exclude = max(1, int(len(steps) * args.right_pick_exclude_last_ratio))
    search_range = range(0, len(steps) - n_exclude)
    if not search_range:
        search_range = range(len(steps))
    i_min_r2 = min(search_range, key=lambda i: steps[i]["d_right_ee_cube2"])
    phase3 = steps[i_min_r2]

    # Phase 4: last frame of episode (stack done)
    phase4 = steps[-1]

    result = {
        "episode_id": args.episode_id,
        "env_id": args.env_id,
        "num_records_in_episode": len(steps),
        "phases": {
            "1_left_pick_cube1": {"description": "left EE closest to cube_1 (before pick)", "global_index": phase1["global_index"], "step": phase1["step"], "d_left_ee_cube1": phase1["d_left_ee_cube1"]},
            "2_left_release": {"description": "left EE moved away from cube_1 (release)", "global_index": phase2["global_index"], "step": phase2["step"], "d_left_ee_cube1": phase2["d_left_ee_cube1"]},
            "3_right_pick_cube2": {"description": "right EE closest to cube_2 (before pick)", "global_index": phase3["global_index"], "step": phase3["step"], "d_right_ee_cube2": phase3["d_right_ee_cube2"]},
            "4_stack_done": {"description": "last frame of episode (stack done)", "global_index": phase4["global_index"], "step": phase4["step"]},
        },
        "global_indices_for_extract": [phase1["global_index"], phase2["global_index"], phase3["global_index"], phase4["global_index"]],
        "all_global_indices": [s["global_index"] for s in steps],
    }

    print("Episode {} — 4 phases by distance".format(args.episode_id))
    print("  Phase 1 (left pick cube_1):  global_index={}  step={}  d(left_ee,cube_1)={:.4f}m".format(
        result["phases"]["1_left_pick_cube1"]["global_index"], result["phases"]["1_left_pick_cube1"]["step"], result["phases"]["1_left_pick_cube1"]["d_left_ee_cube1"]))
    print("  Phase 2 (left release):      global_index={}  step={}  d(left_ee,cube_1)={:.4f}m".format(
        result["phases"]["2_left_release"]["global_index"], result["phases"]["2_left_release"]["step"], result["phases"]["2_left_release"]["d_left_ee_cube1"]))
    print("  Phase 3 (right pick cube_2): global_index={}  step={}  d(right_ee,cube_2)={:.4f}m".format(
        result["phases"]["3_right_pick_cube2"]["global_index"], result["phases"]["3_right_pick_cube2"]["step"], result["phases"]["3_right_pick_cube2"]["d_right_ee_cube2"]))
    print("  Phase 4 (stack done):        global_index={}  step={}".format(
        result["phases"]["4_stack_done"]["global_index"], result["phases"]["4_stack_done"]["step"]))
    print("  global_indices_for_extract:", result["global_indices_for_extract"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
