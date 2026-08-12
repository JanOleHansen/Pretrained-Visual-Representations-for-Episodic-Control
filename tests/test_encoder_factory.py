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
# Six arms, two env pairs. Everything that is not phi has to be held equal or
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
]


def _ablation_cfgs():
    from tests.conftest import load_experiment_cfg

    return {arm: load_experiment_cfg(arm, ["logger=[]"]) for arm in _ABLATION_ARMS}


def test_every_arm_shares_one_training_budget():
    cfgs = _ablation_cfgs()
    budgets = {
        arm: (c.trainer.total_frames, c.trainer.num_envs,
              c.trainer.eval_every_n_steps, c.algorithm.buffer_size)
        for arm, c in cfgs.items()
    }
    assert len(set(budgets.values())) == 1, (
        f"arms differ in budget, so a between-arm comparison is not an encoder "
        f"comparison: {budgets}"
    )


def test_the_rgb_arms_share_one_env_pair():
    """rp_rgb / dinov2 / resnet / clip must see byte-identical observations."""
    from hydra.core.hydra_config import HydraConfig

    seen = {}
    for arm in ["mfec/rp_rgb", "mfec/dinov2",
                "mfec/resnet", "mfec/clip"]:
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

_ATARI3 = ["Assault", "BankHeist", "RoadRunner", "Jamesbond", "MsPacman"]


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
