"""Offline supervised-learning pipeline for the EA2 envelope task.

The pipeline mirrors the online PPO setup but replaces the RL objective with
imitation of the geometric oracle:

    (range-image sequence, ego motion)  ->  5-D normalised envelope params

Because the EA2 state transition is independent of the action (the motion is
scripted along a planned path), data can be collected with a *zero-action*
rollout and there is no distribution shift between collection and deployment.
That makes plain supervised training strictly more informative than sampling the
same signal through PPO, and it takes seconds instead of hours.

Layout
------
``sl/``          library code (config, dataset, model, train, evaluate, export)
``sl/scripts/``  thin CLI entry points that call into the library
``sl/logs/``     artifacts: ``data/`` (collected maps) + ``runs/<name>/``

Note that ``sl/train.py`` (library) and ``sl/scripts/train.py`` (CLI) share a
name by design: the scripts mirror the library's module layout, and the CLI is
always invoked by its full dotted path.

Quick start
-----------
::

    # one command for everything
    python -m legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.pipeline \\
        --run-name baseline

    # or stage by stage
    python -m ...sl.scripts.collect --seeds 1 --num-envs 96
    python -m ...sl.scripts.train   --run-name baseline --epochs 50 --patience 10
    python -m ...sl.scripts.eval    --ckpt sl/logs/runs/baseline/model.pt
    python -m ...sl.scripts.export  --ckpt sl/logs/runs/baseline/model.pt \\
                                    --run-name baseline
    # then, with the exported checkpoint now in logs/el4090_ea2/<run>/model_0.pt:
    python -m legged_gym.scripts.play_ea2  --task=el4090_ea2 \\
        --load_run baseline --checkpoint 0

Isaac Gym constraint
--------------------
Only **one** environment may exist per process: constructing a second
``EL_4090_EA2`` segfaults.  ``collect`` and ``eval`` therefore run one map seed
per process, and ``pipeline`` spawns a subprocess per stage.

Test environment variables
--------------------------
The integration assertions under ``tests/ea2/`` are skipped unless the relevant
variable is set, so the suite stays runnable without a GPU or collected data:

=========================  ==================================================
``EA2_SL_DATA_DIR``        directory of ``map_seed*.pt`` (G0)
``EA2_SL_METRICS``         ``model_metrics.json`` written by train (G2)
``EA2_SL_CKPT``            ``model.pt`` SL checkpoint (G4)
``EA2_SL_EVAL``            ``closed_loop*.json`` written by eval (G3)
``EA2_SL_PPO_DIR``         directory of ``ppo_*.json`` (G5)
=========================  ==================================================

Example::

    export EA2_SL_DATA_DIR=legged_gym/envs/el_4090/envelope_adaptive_2/sl/logs/data
    export EA2_SL_METRICS=.../sl/logs/runs/baseline/model_metrics.json
    export EA2_SL_CKPT=.../sl/logs/runs/baseline/model.pt
    export EA2_SL_EVAL=.../sl/logs/runs/baseline/closed_loop.json
    export EA2_SL_PPO_DIR=.../sl/logs/runs/baseline
    python -m pytest legged_gym/tests/ea2 -q
"""
