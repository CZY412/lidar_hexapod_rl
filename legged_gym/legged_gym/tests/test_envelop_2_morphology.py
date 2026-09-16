"""Test ``el4090_envelop_2`` with a fixed spider, mammal, or mixed morphology.

Examples:
    python legged_gym/legged_gym/tests/test_envelop_2_morphology.py --headless --morphology spider
    python legged_gym/legged_gym/tests/test_envelop_2_morphology.py --headless --morphology mammal
    python legged_gym/legged_gym/tests/test_envelop_2_morphology.py --headless --morphology mixed

``mixed`` makes front and middle legs mammal-like and leaves the rear legs in
the spider configuration.  The selected prior is reapplied before every step,
so periodic command/envelope resampling cannot change the test morphology.
"""

import isaacgym  # noqa: F401 -- must be imported before torch-backed modules.
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- registers environments.
from legged_gym.utils import get_args, task_registry


TASK_NAME = "el4090_envelop_2"
MORPHOLOGY_PRIOR_NAMES = (
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
)
MORPHOLOGY_PRIORS = {
    "spider": (0.0, 0.0, 0.0),
    "mammal": (1.0, 1.0, 1.0),
    "mixed": (1.0, 1.0, 0.0),
}


def set_fixed_morphology(env, morphology):
    """Set morphology priors directly; geometry fields retain their midpoints."""
    condition = env.envelope_state.get().clone()
    prior_indices = [env.condition_names.index(name) for name in MORPHOLOGY_PRIOR_NAMES]
    condition[:, prior_indices] = condition.new_tensor(MORPHOLOGY_PRIORS[morphology])
    # Do not derive priors from envelope geometry: the CLI flag is authoritative.
    env.set_envelope_condition(condition, derive_priors=False)


def print_joint_targets(env):
    target = env.embedded_state_default_dof_pos[0]
    print("Morphology targets [rad]:")
    for leg in ("RF", "RM", "RB", "LF", "LM", "LB"):
        values = []
        for joint in ("HAA", "HFE", "KFE"):
            index = env.dof_names.index("{}_{}".format(leg, joint))
            values.append("{}={:+.4f}".format(joint, target[index].item()))
        print("  {}: {}".format(leg, ", ".join(values)))


def test_env(args):
    if args.task != TASK_NAME:
        raise ValueError("This test only supports --task {}".format(TASK_NAME))

    env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    set_fixed_morphology(env, args.morphology)
    env.reset()
    # reset_idx uses the fixed envelope condition, therefore the first state
    # already has the selected joint target rather than transitioning from it.
    set_fixed_morphology(env, args.morphology)
    print("Testing {} morphology".format(args.morphology))
    print_joint_targets(env)

    for _ in range(int(10 * env.max_episode_length)):
        set_fixed_morphology(env, args.morphology)
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
    print("Done")


if __name__ == "__main__":
    args = get_args(
        [
            {
                "name": "--morphology",
                "type": str,
                "default": "spider",
                "help": "Initial morphology: spider, mammal, or mixed (front/middle mammal).",
            }
        ]
    )
    if args.morphology not in MORPHOLOGY_PRIORS:
        raise ValueError("--morphology must be one of: {}".format(", ".join(MORPHOLOGY_PRIORS)))
    test_env(args)
