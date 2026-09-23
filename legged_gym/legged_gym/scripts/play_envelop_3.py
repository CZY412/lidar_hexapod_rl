# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.

"""Interactive playback for 78-D network-conditioned modal ``el4090_envelop_3``."""

import play_envelop_2 as envelop_2_play

import torch


TASK_NAME = "el4090_envelop_3"
EXPECTED_OBS_DIM = 78


def _print_state(env, label, robot_index=0):
    label = {
        "前腿 mammal 测试包络": "最小包络",
        "O 循环开始：全昆虫型": "O 循环开始：最大包络",
        "O 循环中点：全哺乳型": "O 循环中点：最小包络",
        "O 循环完成：全昆虫型": "O 循环完成：最大包络",
    }.get(label, label)
    condition = env._get_structure_condition()[robot_index]
    names = list(env.condition_names)
    geometry = "  ".join(f"{name}={condition[names.index(name)].item():.3f}" for name in names)
    ranges = env.haa_swing_ranges[robot_index].detach().cpu()
    range_text = "  ".join(
        f"{env.dof_names[index]}=[{ranges[leg, 0]:.3f}, {ranges[leg, 1]:.3f}]"
        for leg, index in enumerate(env.haa_indices.tolist())
    )
    priors = torch.stack(
        [
            env._get_morphology_prior("morphology_front_prior")[robot_index],
            env._get_morphology_prior("morphology_middle_prior")[robot_index],
            env._get_morphology_prior("morphology_back_prior")[robot_index],
        ]
    ).detach().cpu()
    prior_text = "  ".join(
        f"{name}={value:.3f}"
        for name, value in zip(("front", "middle", "back"), priors.tolist())
    )
    print(
        f"\n[{label}]\n  allowed envelope: {geometry}\n"
        f"  network HAA ranges: {range_text}\n  morphology priors: {prior_text}"
    )


def _apply_envelope(env, condition, *, reset_transition=True):
    """Update network inputs and restart the modal-pose transition if requested."""
    updated = env.set_envelope_condition(condition)
    if reset_transition:
        env.embedded_state_transition_time[:] = 0.0
    return updated


def _minimum_envelope_condition(env):
    """Smallest five-parameter envelope inside the network training bounds."""
    state = env.envelope_state
    condition = (0.5 * (state.low + state.high)).unsqueeze(0).repeat(env.num_envs, 1)
    names = list(state.condition_names)
    for name in ("front_width", "middle_width", "back_width", "forward_limit"):
        index = names.index(name)
        condition[:, index] = state.low[index]
    backward_index = names.index("backward_limit")
    condition[:, backward_index] = state.high[backward_index]
    return condition


def _print_keyboard_table(follow_camera):
    print("\n按键：W/S 前后，A/D 横移，Q/E 转向，X 停止，"
          "M 最大包络，R 随机训练包络，F 最小包络，"
          "O 最大/最小包络循环，ESC 退出。")
    if follow_camera:
        print("C：切换相机视角。")


def main() -> None:
    # envelop_2 already drives the same five external geometry controls.
    envelop_2_play.TASK_NAME = TASK_NAME
    envelop_2_play.EXPECTED_OBS_DIM = EXPECTED_OBS_DIM
    envelop_2_play.EXPORT_POLICY = True
    envelop_2_play._print_envelope_state = _print_state
    envelop_2_play._apply_envelope = _apply_envelope
    envelop_2_play._front_mammal_condition = _minimum_envelope_condition
    envelop_2_play._all_insect_condition = envelop_2_play._maximum_envelope_condition
    envelop_2_play._all_mammal_condition = _minimum_envelope_condition
    envelop_2_play._print_keyboard_table = _print_keyboard_table
    args = envelop_2_play.get_args(
        [{"name": "--max_steps", "type": int, "help": "Stop after this many simulation steps."}]
    )
    envelop_2_play.FOLLOW_CAMERA = not args.headless
    envelop_2_play.REALTIME_MODE = not args.headless
    envelop_2_play.play(args)


if __name__ == "__main__":
    main()
