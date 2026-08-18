"""The twelve NEC encoder-ablation arms must differ ONLY in the encoder.

3 games x 4 encoders (`nature`, `dinov2_finetune`, `clip_finetune`,
`mae_finetune`).  Hydra experiment configs do not compose with each other, so
every shared setting is repeated verbatim in all twelve files and nothing but a
test keeps them in step.

``tests/test_nec_mae_finetune.py::test_the_mae_arm_holds_every_learning_knob_identical``
already pinned the algorithm-side knobs for the three ``_mae`` files.  This
widens that to all twelve arms and adds the two trainer-side knobs that were
NOT previously guarded and had in fact drifted:

``num_envs``
    was 16 on the three `nature` arms against 8 on the nine PVM arms.  It is
    **learning-relevant for episodic control**, not a pure resource knob: NEC
    writes to the DND only at EPISODE END, so frames sitting in a trailing
    partial episode when the run stops are never written at all, and the loss is
    ~``num_envs x (mean episode length / 2)`` — absolute, so it bites hardest at
    a short budget.  The MFEC side measured the same effect from the other end
    (`configs/experiment/mfec/rp_gray.yaml`): 16 envs kept 39% fewer unique
    states than 4 at equal frames, because the envs share one memory and a low
    epsilon and largely retread one trajectory.
``num_eval_episodes``
    was 10 on the `nature` arms against 5 on the PVM arms, which gave the
    baseline's ``eval/return_mean`` a different standard error from the arms it
    is plotted against.
"""
from __future__ import annotations

import itertools

import pytest

GAMES = ("mspacman", "qbert", "frostbite")
ENCODERS = ("", "_dinov2", "_clip", "_mae")

#: Everything that moves the learning curve.  A knob added to the ablation
#: belongs here, not in a comment.
ALGORITHM_KNOBS = (
    "num_updates", "eps_start", "eps_end", "annealing_frames",
    "init_random_frames", "eval_eps", "kernel_delta", "gamma", "n_step",
    "embedding_dim", "k", "dnd_capacity", "lr", "rmsprop_alpha", "rmsprop_eps",
    "max_grad_norm", "batch_size", "dnd_lr", "dnd_key_lr", "dnd_value_lr",
    "frames_per_batch",
)

TRAINER_KNOBS = (
    "total_frames", "seed", "num_envs", "log_every_n_steps",
    "eval_every_n_steps", "num_eval_episodes",
)


def _cfg(game: str, encoder: str):
    from tests.conftest import load_experiment_cfg

    return load_experiment_cfg(f"nec/{game}{encoder}", ["logger=[]"])


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("encoder", [e for e in ENCODERS if e])
def test_every_arm_matches_its_nature_baseline(game, encoder):
    arm = _cfg(game, encoder)
    nature = _cfg(game, "")

    for key in ALGORITHM_KNOBS:
        assert arm.algorithm[key] == nature.algorithm[key], (
            f"nec/{game}{encoder} differs from nec/{game} on algorithm.{key}: "
            f"{arm.algorithm[key]} vs {nature.algorithm[key]}. Tuning one arm "
            f"voids the encoder comparison — land the change in all twelve."
        )
    for key in TRAINER_KNOBS:
        assert arm.trainer[key] == nature.trainer[key], (
            f"nec/{game}{encoder} differs from nec/{game} on trainer.{key}: "
            f"{arm.trainer[key]} vs {nature.trainer[key]}."
        )

    # Same env pair, so the encoder really is the only variable.
    assert arm.environment.name == nature.environment.name
    assert arm.eval_environment.name == nature.eval_environment.name


@pytest.mark.parametrize("knob", TRAINER_KNOBS)
def test_trainer_knobs_are_identical_across_games_too(knob):
    """A knob that moved with the GAME would make a cross-game read something
    other than a game comparison — the same rule the MFEC ablation follows."""
    values = {
        f"{g}{e}": _cfg(g, e).trainer[knob]
        for g, e in itertools.product(GAMES, ENCODERS)
    }
    assert len(set(values.values())) == 1, f"trainer.{knob} varies: {values}"


def test_the_exploration_schedule_fits_inside_the_probe_budget():
    """The bug this pins: `annealing_frames` and `init_random_frames` were sized
    for a 1_000_000-step run and survived the cut to the 100k probe budget, so
    the anneal spanned half the run and ~26.6% of every collected frame was a
    uniform random action.

    Both are AGENT-STEP counts, exactly like `trainer.total_frames`, so they are
    only meaningful relative to it and must be rescaled with it.
    """
    for game, encoder in itertools.product(GAMES, ENCODERS):
        cfg = _cfg(game, encoder)
        alg, trainer = cfg.algorithm, cfg.trainer
        name = f"nec/{game}{encoder}"

        # torchrl rounds init_random_frames UP to a whole number of collector
        # batches and warns when it has to (collectors/_single.py).
        assert alg.init_random_frames % alg.frames_per_batch == 0, (
            f"{name}: init_random_frames must be a multiple of frames_per_batch"
        )
        assert alg.frames_per_batch % trainer.num_envs == 0, name

        # The anneal has to finish well inside the run, or eps_end is never
        # reached and the configured floor is fiction.
        assert alg.annealing_frames <= trainer.total_frames / 5, (
            f"{name}: annealing_frames={alg.annealing_frames} against "
            f"total_frames={trainer.total_frames} — epsilon reaches its floor "
            f"too late for the run to show it."
        )
        assert alg.init_random_frames <= trainer.total_frames / 10, (
            f"{name}: init_random_frames={alg.init_random_frames} spends "
            f"{100 * alg.init_random_frames / trainer.total_frames:.0f}% of the "
            f"budget on a uniform random policy."
        )
        # The warm-up must not outlast the anneal: its whole purpose is to keep
        # the optimistic-init argmax off an empty DND until the memory exists.
        assert alg.init_random_frames < alg.annealing_frames, name
