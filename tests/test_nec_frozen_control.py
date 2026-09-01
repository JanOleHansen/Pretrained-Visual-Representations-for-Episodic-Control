"""The twelve frozen-NEC arms — the RQ4 control.

RQ4 asks whether finetuning a pretrained encoder beats keeping its features
fixed.  The reported grids cannot answer it: MFEC is frozen in all 90 of its
runs and NEC is finetuned in all 60 of its own, so `algorithm` and `training
regime` move together.  `configs/experiment/nec/{game}_{enc}_frozen.yaml` is the
missing cell, and its whole value rests on being its finetuned counterpart with
ONE knob changed.  That is what this file pins.

Two things are deliberately NOT tested here because they are already covered:

* the freeze MECHANICS (backbone params get ``requires_grad=False``, the head
  stays trainable, ``param_groups`` drops the empty backbone group) live in
  ``tests/test_nec_{resnet,dinov2,clip,mae}_finetune.py``, which have the stub
  backbones needed to build a real network without a checkpoint download;
* the shared learning knobs are pinned across every arm, frozen ones included,
  by ``tests/test_nec_ablation_parity.py``.

What is left, and what this file asserts, is that the twelve configs select the
right group, flip exactly one knob, and do not collide with the finetuned arm's
run directory or W&B group.
"""
from __future__ import annotations

import itertools

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

GAMES = ("mspacman", "qbert", "frostbite")
ENCODERS = ("resnet", "dinov2", "clip", "mae")

NETWORK_CLASS = {
    "resnet": "src.networks.ResNetEmbedding",
    "dinov2": "src.networks.DINOv2Embedding",
    "clip": "src.networks.CLIPEmbedding",
    "mae": "src.networks.MAEEmbedding",
}

ARMS = list(itertools.product(GAMES, ENCODERS))


def _cfg(name: str):
    from tests.conftest import load_experiment_cfg

    return load_experiment_cfg(f"nec/{name}", ["logger=[]"])


@pytest.mark.parametrize("game,enc", ARMS)
def test_the_frozen_arm_freezes_the_backbone(game, enc):
    cfg = _cfg(f"{game}_{enc}_frozen")

    assert cfg.algorithm.embedding_network.freeze_backbone is True, (
        "the RQ4 control exists only to hold the pretrained trunk fixed"
    )
    assert cfg.algorithm.embedding_network._target_ == NETWORK_CLASS[enc]


@pytest.mark.parametrize("game,enc", ARMS)
def test_the_only_functional_difference_is_freeze_backbone(game, enc):
    """The RQ4 contrast is only clean if nothing else moved.

    A second difference here — a resolution, a checkpoint, a pooling mode —
    would make `frozen vs finetuned` a comparison of two things at once, which
    is exactly the confound the control was built to remove.
    """
    frozen = _cfg(f"{game}_{enc}_frozen")
    tuned = _cfg(f"{game}_{enc}")

    a = OmegaConf.to_container(frozen.algorithm.embedding_network, resolve=True)
    b = OmegaConf.to_container(tuned.algorithm.embedding_network, resolve=True)

    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"freeze_backbone"}, (
        f"nec/{game}_{enc}_frozen differs from nec/{game}_{enc} on "
        f"{sorted(differing)}; only freeze_backbone may differ."
    )

    # Same task, same budget -> the encoder's trainability is the only variable.
    assert frozen.environment.name == tuned.environment.name
    assert frozen.eval_environment.name == tuned.eval_environment.name
    assert frozen.trainer.total_frames == tuned.trainer.total_frames
    assert frozen.trainer.num_envs == tuned.trainer.num_envs
    assert frozen.trainer.num_eval_episodes == tuned.trainer.num_eval_episodes


@pytest.mark.parametrize("game,enc", ARMS)
def test_the_run_name_does_not_collide_with_the_finetuned_arm(game, enc):
    """`run.name` is the run DIRECTORY and `run.group` is the W&B group that a
    multi-seed sweep is averaged over.  Reusing the finetuned arm's `encoder`
    token would overwrite its checkpoints and silently average two different
    arms into one mean +/- SEM curve."""
    frozen = _cfg(f"{game}_{enc}_frozen")
    tuned = _cfg(f"{game}_{enc}")

    assert frozen.run.encoder == f"{enc}_frozen"
    assert frozen.run.name == f"nec_{game}_{enc}_frozen_seed42"
    assert frozen.run.group == f"nec_{game}_{enc}_frozen"

    assert frozen.run.name != tuned.run.name
    assert frozen.run.group != tuned.run.group
    # ...and not against the NatureEmbedding baseline either.
    assert frozen.run.name != _cfg(game).run.name


def test_every_frozen_run_name_is_unique_across_the_grid():
    """36 finetuned/frozen/baseline arms, 36 distinct directories."""
    names = {}
    for game in GAMES:
        for suffix in ("", *(f"_{e}" for e in ENCODERS),
                       *(f"_{e}_frozen" for e in ENCODERS)):
            cfg = _cfg(f"{game}{suffix}")
            assert cfg.run.name not in names, (
                f"nec/{game}{suffix} and nec/{names[cfg.run.name]} both write "
                f"{cfg.run.name}"
            )
            names[cfg.run.name] = f"{game}{suffix}"
    assert len(names) == 27


@pytest.mark.parametrize("game,enc", ARMS)
def test_the_frozen_arm_instantiates(game, enc):
    """The embedding network is a `_partial_` factory, so this builds the
    algorithm without downloading any checkpoint."""
    from src.algorithms.nec import NECAlgorithm

    alg = instantiate(_cfg(f"{game}_{enc}_frozen").algorithm, device=None)
    assert isinstance(alg, NECAlgorithm)
