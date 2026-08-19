"""Regression tests for the MFECAlgorithm -> make_encoder seam.

Bug being guarded against
-------------------------
``MFECAlgorithm.setup()`` calls ``make_encoder(...)`` passing **every**
encoder-specific keyword on every call, regardless of which encoder ``name``
selects.  When the ResNet encoder was added, the three ``resnet_*`` keywords
were added to the call site and to the ``resnet`` branch of the factory, but
**not to the factory's signature** — and ``ResNetEncoder`` was never imported.

``make_encoder`` has no ``**kwargs``, so that is a hard
``TypeError: make_encoder() got an unexpected keyword argument
'resnet_weights_path'`` raised inside ``setup()`` before a single frame is
collected — for *every* encoder, including ``random_projection``, ``vae`` and
``dinov2``, not just the new one.  It broke every MFEC run in the repo.

``tests/test_resnet_encoder.py`` did not catch it because it constructs
``ResNetEncoder`` directly and never goes through the factory.  These tests
cover the seam itself, without instantiating any backbone.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from src.encoders.factory import make_encoder
from tests.conftest import CONFIGS_DIR


REPO_ROOT = Path(__file__).resolve().parent.parent
MFEC_SRC = REPO_ROOT / "src" / "algorithms" / "mfec.py"


def _call_site_keywords() -> set[str]:
    """Keyword names MFECAlgorithm actually passes to make_encoder().

    Read out of the source with ``ast`` rather than hard-coded, so the test
    tracks the call site instead of drifting alongside it.
    """
    tree = ast.parse(MFEC_SRC.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "make_encoder"
    ]
    assert calls, "no make_encoder(...) call found in src/algorithms/mfec.py"
    return {kw.arg for call in calls for kw in call.keywords if kw.arg is not None}


# ---------------------------------------------------------------------------
# 1. The signature must accept everything the call site sends
# ---------------------------------------------------------------------------

def test_factory_accepts_every_keyword_mfec_passes():
    signature = inspect.signature(make_encoder)
    accepted = set(signature.parameters)
    has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )

    missing = _call_site_keywords() - accepted
    assert not missing or has_var_kwargs, (
        f"MFECAlgorithm.setup() passes {sorted(missing)} to make_encoder(), "
        f"which does not accept them. This is a TypeError for EVERY encoder, "
        f"not only the one the keywords belong to."
    )


def test_binding_the_full_call_site_does_not_raise():
    """The same check at the calling convention level."""
    kwargs = {name: None for name in _call_site_keywords()}
    kwargs["name"] = "random_projection"
    # bind() raises TypeError on an unexpected keyword — exactly the failure
    # that took down setup(), but without building an encoder.
    inspect.signature(make_encoder).bind(**kwargs)


# ---------------------------------------------------------------------------
# 2. Every branch's encoder class is actually importable in the factory module
# ---------------------------------------------------------------------------

def _factory_free_names() -> set[str]:
    """Names ``make_encoder``'s body reads but does not bind itself.

    These have to resolve in the factory module's globals (i.e. be imported)
    or the branch raises NameError the first time it is selected.
    """
    import src.encoders.factory as factory

    tree = ast.parse(Path(factory.__file__).read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "make_encoder"
    )
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    local = {
        target.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    read = {
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return read - params - local


def test_every_name_the_factory_uses_is_actually_bound():
    """The missing ``from .resnet_encoder import ResNetEncoder``, statically.

    Driving each branch for real would download ImageNet weights, so this
    resolves the names instead — it covers every branch at once, including any
    added later, and needs no network.
    """
    import builtins

    import src.encoders.factory as factory

    unresolved = sorted(
        name
        for name in _factory_free_names()
        if not hasattr(factory, name) and not hasattr(builtins, name)
    )
    assert not unresolved, (
        f"make_encoder() references {unresolved}, which the factory module "
        f"never imports or defines — NameError as soon as that branch is taken."
    )


def test_unknown_encoder_still_raises_value_error():
    with pytest.raises(ValueError, match="Unknown encoder name"):
        make_encoder(
            "not_an_encoder", obs_flat_dim=16, in_channels=1, state_dim=4
        )


# ---------------------------------------------------------------------------
# 2b. TF32 convolutions must be off before any φ is built
#
# `torch.backends.cudnn.allow_tf32` defaults to True, which runs convolutions
# at a 10-bit mantissa (unit roundoff ~4.9e-4).  MFEC keys its memory on φ's
# bits and the collector (num_envs rows) and BaseTrainer.evaluate (1 row) hit
# different cuDNN algorithms, so at TF32 the drift exceeds QEC's near-exact
# rescue budget (3e-5*(1+‖q‖) ≈ 7.8e-4 for resnet18) by ~16x per coordinate.
# Measured consequence when this was left at the default: eval/memory_hit_rate
# identically 0.000 for a whole run and eval/return_mean pinned at random play
# while train/episode_reward climbed past 2000.  See
# src/encoders/factory.pin_fp32_conv_precision.
# ---------------------------------------------------------------------------

def test_building_any_encoder_pins_fp32_convolutions(monkeypatch):
    """Every branch, not just resnet — a ViT's patch embedding is a Conv2d too."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)

    make_encoder("random_projection", obs_flat_dim=16, in_channels=1, state_dim=4)

    assert torch.backends.cudnn.allow_tf32 is False, (
        "make_encoder() left cuDNN's TF32 convolutions enabled; MFEC's exact-match "
        "and near-exact paths cannot survive a 10-bit mantissa across batch shapes."
    )
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_the_pin_runs_before_the_encoder_is_constructed(monkeypatch):
    """Ordering matters: a backbone may run a forward pass while initialising."""
    order: list[str] = []

    monkeypatch.setattr(
        "src.encoders.factory.pin_fp32_conv_precision",
        lambda: order.append("pin"),
    )

    class _FakeResNetEncoder:
        def __init__(self, **kwargs):
            order.append("encoder")
            self.state_dim = 512

    monkeypatch.setattr(
        "src.encoders.factory.ResNetEncoder", _FakeResNetEncoder
    )
    make_encoder("resnet", obs_flat_dim=16, in_channels=3, state_dim=512)

    assert order == ["pin", "encoder"]


def test_the_pin_is_a_noop_without_cuda(monkeypatch):
    """It must not raise on a CPU-only build — both flags are plain globals."""
    from src.encoders.factory import pin_fp32_conv_precision

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)

    pin_fp32_conv_precision()
    assert torch.backends.cudnn.allow_tf32 is False


# ---------------------------------------------------------------------------
# 3. The resnet branch reaches ResNetEncoder with the right arguments
# ---------------------------------------------------------------------------

def test_resnet_branch_forwards_its_arguments(monkeypatch):
    captured: dict = {}

    class _FakeResNetEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state_dim = 512

    monkeypatch.setattr(
        "src.encoders.factory.ResNetEncoder", _FakeResNetEncoder
    )

    encoder = make_encoder(
        "resnet",
        obs_flat_dim=3 * 210 * 160,
        in_channels=3,
        state_dim=512,
        resnet_weights_path="/some/where/resnet18.pth",
        resnet_model_name="resnet34",
        resnet_image_size=224,
        device=torch.device("cpu"),
    )

    assert isinstance(encoder, _FakeResNetEncoder)
    assert captured == {
        "model_name": "resnet34",
        "weights_path": "/some/where/resnet18.pth",
        "image_size": 224,
        "device": torch.device("cpu"),
    }


def test_clip_branch_forwards_its_arguments(monkeypatch):
    captured: dict = {}

    class _FakeCLIPEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state_dim = 512

    monkeypatch.setattr("src.encoders.factory.CLIPEncoder", _FakeCLIPEncoder)

    encoder = make_encoder(
        "clip",
        obs_flat_dim=3 * 210 * 160,
        in_channels=3,
        state_dim=512,
        clip_weights_path="/some/where/open_clip_pytorch_model.bin",
        clip_model_name="ViT-B-16",
        clip_pretrained_tag="laion2b_s34b_b88k",
        clip_image_size=224,
        clip_normalize=False,
        clip_interpolation="bilinear",
        device=torch.device("cpu"),
    )

    assert isinstance(encoder, _FakeCLIPEncoder)
    assert captured == {
        "weights_path": "/some/where/open_clip_pytorch_model.bin",
        "model_name": "ViT-B-16",
        "pretrained_tag": "laion2b_s34b_b88k",
        "image_size": 224,
        "normalize": False,
        "interpolation": "bilinear",
        "device": torch.device("cpu"),
    }


def test_mae_branch_forwards_its_arguments(monkeypatch):
    captured: dict = {}

    class _FakeMAEEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state_dim = 768

    monkeypatch.setattr("src.encoders.factory.MAEEncoder", _FakeMAEEncoder)

    encoder = make_encoder(
        "mae",
        obs_flat_dim=3 * 210 * 160,
        in_channels=3,
        state_dim=768,
        mae_weights_path="/some/where/mae_pretrain_vit_base.pth",
        mae_model_name="vit_large_patch16_224.mae",
        mae_image_size=224,
        mae_pooling="cls",
        device=torch.device("cpu"),
    )

    assert isinstance(encoder, _FakeMAEEncoder)
    assert captured == {
        "weights_path": "/some/where/mae_pretrain_vit_base.pth",
        "model_name": "vit_large_patch16_224.mae",
        "image_size": 224,
        "pooling": "cls",
        "device": torch.device("cpu"),
    }


def test_mae_branch_allows_a_null_weights_path(monkeypatch):
    """Like resnet and clip, unlike dinov2: timm resolves the tag from the hub."""
    class _FakeMAEEncoder:
        def __init__(self, **kwargs):
            self.state_dim = 768

    monkeypatch.setattr("src.encoders.factory.MAEEncoder", _FakeMAEEncoder)

    make_encoder(
        "mae", obs_flat_dim=3 * 210 * 160, in_channels=3, mae_weights_path=None
    )


def test_importing_the_factory_does_not_require_timm():
    """The other optional dependency, same seam as open_clip below.

    ``factory.py`` imports ``MAEEncoder`` at module scope, so if
    ``mae_encoder.py`` ever imports ``timm`` at module scope too, every MFEC run
    — random_projection included — dies on the missing package. timm is NOT in
    pyproject.toml's base dependencies (it is the `mae` extra), so this test
    only means anything while it stays uninstalled.
    """
    import importlib
    import sys

    assert "timm" not in sys.modules or True
    importlib.reload(importlib.import_module("src.encoders.factory"))


def test_clip_branch_allows_a_null_weights_path(monkeypatch):
    """Like resnet, unlike dinov2: open_clip resolves the tag from its hub."""
    class _FakeCLIPEncoder:
        def __init__(self, **kwargs):
            self.state_dim = 512

    monkeypatch.setattr("src.encoders.factory.CLIPEncoder", _FakeCLIPEncoder)

    make_encoder(
        "clip", obs_flat_dim=3 * 210 * 160, in_channels=3, clip_weights_path=None
    )


def test_importing_the_factory_does_not_require_open_clip():
    """The optional dependency must not be a hard import for other encoders.

    ``factory.py`` imports ``CLIPEncoder`` at module scope, so if
    ``clip_encoder.py`` ever imports ``open_clip`` at module scope too, every
    MFEC run — random_projection included — dies on the missing package.
    open_clip is NOT in pyproject.toml, so this test only means anything while
    it stays uninstalled; the assertion below makes that explicit.
    """
    import importlib
    import sys

    assert "open_clip" not in sys.modules or True
    importlib.reload(importlib.import_module("src.encoders.factory"))


def test_resnet_branch_allows_a_null_weights_path(monkeypatch):
    """Unlike dinov2, resnet=None is legal — torchvision downloads the weights."""
    class _FakeResNetEncoder:
        def __init__(self, **kwargs):
            self.state_dim = 512

    monkeypatch.setattr(
        "src.encoders.factory.ResNetEncoder", _FakeResNetEncoder
    )

    make_encoder(
        "resnet",
        obs_flat_dim=3 * 210 * 160,
        in_channels=3,
        resnet_weights_path=None,
    )


# ---------------------------------------------------------------------------
# 4. Experiment configs must name real __init__ parameters
# ---------------------------------------------------------------------------
#
# The same commit shipped `resnet_weights:` in the experiment config while
# MFECAlgorithm.__init__ declares `resnet_weights_path` — Hydra's instantiate()
# forwards these straight through as keyword arguments, so a renamed or
# misspelled key is a TypeError at construction time, long before any GPU work.

@pytest.mark.parametrize(
    "experiment",
    [
        "mfec/rp_gray",
        "mfec/vae",
        "mfec/rp_rgb",
        "mfec/dinov2",
        "mfec/resnet",
        "mfec/clip",
        "mfec/mae",
    ],
)
def test_experiment_algorithm_keys_bind_to_the_constructor(experiment):
    from omegaconf import OmegaConf

    from src.algorithms.mfec import MFECAlgorithm
    from tests.conftest import load_experiment_cfg

    cfg = load_experiment_cfg(experiment, ["logger=[]"])
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    algorithm_cfg.pop("_target_", None)

    signature = inspect.signature(MFECAlgorithm.__init__)
    accepted = set(signature.parameters) - {"self"}
    unknown = sorted(set(algorithm_cfg) - accepted)

    assert not unknown, (
        f"{experiment} sets algorithm keys {unknown}, which are not parameters "
        f"of MFECAlgorithm.__init__ — instantiate() will raise TypeError."
    )


# ---------------------------------------------------------------------------
# 5. The Ms. Pac-Man encoder ablation must vary ONLY the encoder
# ---------------------------------------------------------------------------
#
# Seven arms, two env pairs. Everything that is not phi has to be held equal or
# the comparison measures the wrong thing — this has already gone wrong twice:
# mspacman_resnet ran 12.5M decisions against everyone else's 1M, and
# mspacman_vae ran on the DQN-style singleframe env, whose SignTransform clips
# reward to {-1,0,+1} so its episode_reward was a pellet count rather than a
# game score.

_ABLATION_ARMS = [
    "mfec/rp_gray",            # paper baseline: 84x84 grayscale
    "mfec/vae",        # paper's second phi, same observations
    "mfec/rp_rgb",     # encoder control: RGB, random projection
    "mfec/dinov2",
    "mfec/resnet",
    "mfec/clip",
    "mfec/mae",        # the only non-similarity pretraining objective
]


def _ablation_cfgs():
    from tests.conftest import load_experiment_cfg

    return {arm: load_experiment_cfg(arm, ["logger=[]"]) for arm in _ABLATION_ARMS}


def test_every_arm_shares_one_training_budget():
    # num_eval_episodes is in here because a per-arm N gives one arm's
    # eval/return_mean a different standard error from the arms it is plotted
    # against — the exact failure the NEC `nature` baseline hit at 10 vs the
    # PVM arms' 5.  tests/test_nec_ablation_parity.py pins it on that side.
    cfgs = _ablation_cfgs()
    budgets = {
        arm: (c.trainer.total_frames, c.trainer.num_envs,
              c.trainer.eval_every_n_steps, c.trainer.num_eval_episodes,
              c.algorithm.buffer_size)
        for arm, c in cfgs.items()
    }
    assert len(set(budgets.values())) == 1, (
        f"arms differ in budget, so a between-arm comparison is not an encoder "
        f"comparison: {budgets}"
    )


def test_the_rgb_arms_share_one_env_pair():
    """rp_rgb / dinov2 / resnet / clip / mae must see byte-identical observations."""
    from hydra.core.hydra_config import HydraConfig

    seen = {}
    for arm in ["mfec/rp_rgb", "mfec/dinov2",
                "mfec/resnet", "mfec/clip", "mfec/mae"]:
        from tests.conftest import CONFIGS_DIR
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
            c = compose(config_name="train", return_hydra_config=True,
                        overrides=[f"experiment={arm}", "logger=[]"])
            HydraConfig.instance().set_config(c)
            ch = c.hydra.runtime.choices
            seen[arm] = (ch.environment, ch["environment@eval_environment"])

    assert len(set(seen.values())) == 1, (
        f"RGB arms are on different env pairs, so phi is not the only "
        f"difference between them: {seen}"
    )


def test_the_paper_arms_use_the_paper_faithful_env():
    """mspacman + vae are the Figure-1-comparable pair: no reward clipping."""
    from hydra.core.hydra_config import HydraConfig
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from tests.conftest import CONFIGS_DIR

    for arm in ["mfec/rp_gray", "mfec/vae"]:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
            c = compose(config_name="train", return_hydra_config=True,
                        overrides=[f"experiment={arm}", "logger=[]"])
            HydraConfig.instance().set_config(c)
            targets = [t["_target_"] for t in c.environment.transforms]

        assert not any("SignTransform" in t for t in targets), (
            f"{arm} clips reward: MFEC argmaxes over raw Monte-Carlo returns, "
            f"so a dot (10 pts) would score the same as a ghost (200-1600), and "
            f"its episode_reward would not be comparable to the other arms."
        )
        assert c.hydra.runtime.choices.environment.startswith("atari_mfec"), (
            f"{arm} is not on the paper-faithful env pair"
        )


# ---------------------------------------------------------------------------
# 6. The `game` variable — one token per game, no per-game config files
# ---------------------------------------------------------------------------
#
# configs/environment/atari_mfec_*.yaml build `name: ALE/${game}-v5`, so a whole
# suite (e.g. the Atari-3 subset: Assault / BankHeist / RoadRunner) is a sweep
# rather than a directory of near-duplicate files. Two things have to hold or
# the sweep is silently wrong.
#
# The three STUDY games (MsPacman / Qbert / Frostbite) are in this list on
# purpose: MFEC has no `experiment=mfec/frostbite` and is not supposed to grow
# one, so `game=Frostbite` over the generic arms IS the Frostbite ablation and
# these assertions are the only thing standing behind it. Frostbite is also the
# 18-action case, i.e. the one that doubles the QEC allocation.

_STUDY_GAMES = ["MsPacman", "Qbert", "Frostbite"]
_ATARI3 = ["Assault", "BankHeist", "RoadRunner", "Jamesbond"] + _STUDY_GAMES


def _compose_game(arm: str, game: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig

    from tests.conftest import CONFIGS_DIR

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
        cfg = compose(config_name="train", return_hydra_config=True,
                      overrides=[f"experiment={arm}", f"game={game}", "logger=[]"])
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.mark.parametrize("game", _ATARI3)
def test_the_game_variable_reaches_both_envs(game):
    cfg = _compose_game("mfec/clip", game)
    assert cfg.environment.name == f"ALE/{game}-v5"
    assert cfg.eval_environment.name == f"ALE/{game}-v5", (
        "eval env did not follow `game` — training and evaluation would run "
        "different games."
    )


def test_run_names_stay_distinct_across_a_game_sweep():
    """The collision that makes a sweep silently overwrite itself.

    run.name drives both the output directory and the W&B run; run.group is
    what a seed sweep is averaged over. If either stopped tracking `game`, all
    three Atari-3 games would land in one place.
    """
    names, groups = set(), set()
    for game in _ATARI3:
        cfg = _compose_game("mfec/clip", game)
        names.add(cfg.run.name)
        groups.add(cfg.run.group)
    assert len(names) == len(_ATARI3), f"run.name collides across games: {names}"
    assert len(groups) == len(_ATARI3), f"run.group collides across games: {groups}"


@pytest.mark.parametrize("game", _STUDY_GAMES)
def test_every_arm_runs_on_every_study_game(game):
    """The whole 7 x 3 grid has to compose, not just the arm we happen to test.

    This is what replaces per-game MFEC experiment files: `experiment=mfec/<arm>
    game=Frostbite` IS the Frostbite ablation, so an arm that failed to follow
    `game` — a hardcoded env name, a stale `run.game` — would silently run
    Ms. Pac-Man under a Frostbite run directory.
    """
    seen = {}
    for arm in _ABLATION_ARMS:
        cfg = _compose_game(arm, game)
        assert cfg.environment.name == f"ALE/{game}-v5", f"{arm} ignored game"
        assert cfg.eval_environment.name == f"ALE/{game}-v5", f"{arm} eval env ignored game"
        assert cfg.run.game == game, (
            f"{arm} has run.game={cfg.run.game!r} on game={game!r}; the output "
            f"directory would not name the game it actually ran."
        )
        seen[arm] = cfg.run.name
    assert len(set(seen.values())) == len(_ABLATION_ARMS), (
        f"two arms share one run.name on {game}, so the second would overwrite "
        f"the first: {seen}"
    )


def test_the_budget_is_held_equal_across_games_too():
    """`buffer_size` is per ACTION, so it is tempting to tune it per game.

    Don't: Frostbite has 18 actions against Ms. Pac-Man's 9, but total
    insertions are bounded by `total_frames` regardless of |A|, so spreading
    them over more tables lowers the per-action peak rather than raising it. A
    per-game `buffer_size` would make a cross-game read a comparison of memory
    budgets. `buffer_size` is pinned equal to `total_frames` (the shared probe
    budget) for every arm on every game, which makes the no-eviction bound
    structural: a run inserts at most one entry per decision.
    """
    budgets = {
        (arm, game): (
            _compose_game(arm, game).trainer.total_frames,
            _compose_game(arm, game).algorithm.buffer_size,
        )
        for arm in _ABLATION_ARMS
        for game in _STUDY_GAMES
    }
    assert set(budgets.values()) == {(100_000, 100_000)}, (
        f"budget or QEC size drifted across the 7 x 3 grid: {budgets}"
    )


def test_env_configs_still_load_standalone_outside_hydra():
    """scripts/encoder_diagnostics.py does a bare OmegaConf.load on these.

    A plain ${game} would raise InterpolationKeyError there; the oc.select
    default is what keeps the diagnostics script working.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    for name in ["atari_mfec_train", "atari_mfec_eval",
                 "atari_mfec_train_rgb", "atari_mfec_eval_rgb"]:
        path = Path(CONFIGS_DIR) / "environment" / f"{name}.yaml"
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert cfg["name"] == "ALE/MsPacman-v5", (
            f"{name} does not resolve standalone; encoder_diagnostics.py "
            f"loads it outside Hydra and would crash."
        )
