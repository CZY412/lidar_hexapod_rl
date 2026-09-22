# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.

"""Interactive playback for the API-derived ``el4090_envelop_3`` task."""

import play_envelop_2 as envelop_2_play


TASK_NAME = "el4090_envelop_3"


def _print_state(env, label, robot_index=0):
    condition = env._get_structure_condition()[robot_index]
    names = list(env.condition_names)
    geometry = "  ".join(f"{name}={condition[names.index(name)].item():.3f}" for name in names)
    priors = env.api_morphology_priors[robot_index].detach().cpu()
    ranges = env.haa_swing_ranges[robot_index].detach().cpu()
    prior_text = "  ".join(
        f"{name}={value:.3f}" for name, value in zip(("front", "middle", "back"), priors.tolist())
    )
    range_text = "  ".join(
        f"{env.dof_names[index]}=[{ranges[leg, 0]:.3f}, {ranges[leg, 1]:.3f}]"
        for leg, index in enumerate(env.haa_indices.tolist())
    )
    print(f"\n[{label}]\n  allowed envelope: {geometry}\n  derived priors: {prior_text}\n  API HAA ranges: {range_text}")


def main() -> None:
    # envelop_2 already drives the same five external geometry controls.
    envelop_2_play.TASK_NAME = TASK_NAME
    envelop_2_play.EXPORT_POLICY = True
    envelop_2_play._print_envelope_state = _print_state
    args = envelop_2_play.get_args(
        [{"name": "--max_steps", "type": int, "help": "Stop after this many simulation steps."}]
    )
    envelop_2_play.FOLLOW_CAMERA = not args.headless
    envelop_2_play.REALTIME_MODE = not args.headless
    envelop_2_play.play(args)


if __name__ == "__main__":
    main()
