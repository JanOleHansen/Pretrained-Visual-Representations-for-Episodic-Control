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
    ["mfec/mspacman", "mfec/mspacman_dinov2", "mfec/mspacman_resnet"],
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
