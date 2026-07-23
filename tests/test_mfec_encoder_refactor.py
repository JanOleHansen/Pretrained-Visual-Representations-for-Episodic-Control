"""Smoke test for the encoder-abstraction refactor in src/algorithms/mfec.py.

Before the refactor, QECPolicy hardcoded a random-projection matrix built
inline in MFECAlgorithm.setup(). The refactor factors this out into a
pluggable Encoder (src/encoders/), selected via encoder_name and constructed
by make_encoder(). This test verifies the refactor is *transparent*: with
encoder_name="random_projection", MFEC's observable behaviour (embeddings,
forward() Q-values, checkpoint round-trip, QEC exact-hit determinism,
deepcopy sharing, and step() metrics) is unchanged from the pre-refactor
hardcoded-projection version.

Uses a lightweight duck-typed mock env (no real Atari/gym dependency) since
MFECAlgorithm.setup() only reads observation_spec / action_spec / batch_size
from the proof env.
"""
from __future__ import annotations

import copy

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite

from src.algorithms.mfec import MFECAlgorithm
from src.encoders.random_projectins import RandomProjectionEncoder


OBS_SHAPE = (4, 84, 84)   # Atari-like stacked grayscale frames
NUM_ACTIONS = 4
STATE_DIM = 8


# ---------------------------------------------------------------------------
# Mock env + algorithm builder
# ---------------------------------------------------------------------------

class _MockAtariEnv:
    """Duck-typed stand-in for a TorchRL EnvBase.

    MFECAlgorithm.setup() only reads observation_spec, action_spec, and
    batch_size off the proof env -- it never resets or steps it -- so a
    minimal object exposing just those three attributes is sufficient.
    """

    def __init__(self, obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS):
        self.observation_spec = Composite(
            pixels=Bounded(low=0, high=255, shape=obs_shape, dtype=torch.uint8)
        )
        self.action_spec = Categorical(n=num_actions)
        self.batch_size = torch.Size([])


def _make_setup_algorithm(
    *,
    seed: int | None = 0,
    state_dim: int = STATE_DIM,
    k: int = 1,
    buffer_size: int = 200,
    num_actions: int = NUM_ACTIONS,
    obs_shape: tuple = OBS_SHAPE,
    **extra_kwargs,
) -> MFECAlgorithm:
    """Build + setup() a fresh MFECAlgorithm wired for random_projection."""
    alg = MFECAlgorithm(
        device=None,
        encoder_name="random_projection",
        seed=seed,
        obs_key="pixels",
        buffer_size=buffer_size,
        k=k,
        state_dim=state_dim,
        gamma=0.9,
        frames_per_batch=10,
        key_scale=1e5,
        **extra_kwargs,
    )
    alg.setup(lambda: _MockAtariEnv(obs_shape=obs_shape, num_actions=num_actions))
    return alg


# ---------------------------------------------------------------------------
# 1. setup() wiring
# ---------------------------------------------------------------------------

def test_setup_wires_random_projection_encoder():
    alg = _make_setup_algorithm(seed=0, state_dim=STATE_DIM)

    assert isinstance(alg.encoder, RandomProjectionEncoder)
    assert alg.encoder.state_dim == STATE_DIM
    assert alg.qec_policy.encoder is alg.encoder


# ---------------------------------------------------------------------------
# 2. embed() shape + device
# ---------------------------------------------------------------------------

def test_embed_shape_and_device():
    alg = _make_setup_algorithm(seed=0)
    E, T = 2, 3
    obs = torch.randint(0, 256, (E, T, *OBS_SHAPE), dtype=torch.uint8)

    out = alg.qec_policy.embed(obs)

    assert out.shape == (E * T, STATE_DIM)
    assert out.device == obs.device


# ---------------------------------------------------------------------------
# 3. Determinism: embed() and _make_keys() agree on repeated calls
# ---------------------------------------------------------------------------

def test_embed_and_keys_are_deterministic():
    alg = _make_setup_algorithm(seed=0)
    obs = torch.randint(0, 256, (5, *OBS_SHAPE), dtype=torch.uint8)

    state1 = alg.qec_policy.embed(obs)
    state2 = alg.qec_policy.embed(obs)
    assert torch.equal(state1, state2), (
        "embed() returned different values for identical input -- the QEC "
        "exact-hit hash path relies on bit-identical embeddings."
    )

    keys1 = alg.qec._make_keys(state1)
    keys2 = alg.qec._make_keys(state2)
    assert keys1 == keys2, (
        "_make_keys() produced different bytes for identical embeddings."
    )


# ---------------------------------------------------------------------------
# 4. forward(): no inf leaks through, on empty and partially-populated QEC
# ---------------------------------------------------------------------------

def test_forward_replaces_inf_when_qec_empty():
    alg = _make_setup_algorithm(seed=0)
    obs = torch.randint(0, 256, (2, 3, *OBS_SHAPE), dtype=torch.uint8)

    q = alg.qec_policy.forward(obs)

    assert q.shape == (2, 3, NUM_ACTIONS)
    assert not torch.isinf(q).any()
    assert torch.all(q == 1e9), "empty QEC must be optimistic everywhere (1e9 sentinel)"


def test_forward_replaces_inf_when_partially_populated():
    alg = _make_setup_algorithm(seed=0, k=1)
    obs = torch.randint(0, 256, (6, *OBS_SHAPE), dtype=torch.uint8)
    states = alg.qec_policy.embed(obs)

    # Populate action 0 only; actions 1..N-1 stay empty.
    alg.qec.add_batch(0, states[:3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    q = alg.qec_policy.forward(obs)

    assert q.shape == (6, NUM_ACTIONS)
    assert not torch.isinf(q).any()
    # Untouched actions still saturate at the isinf -> 1e9 replacement.
    assert torch.all(q[:, 1:] == 1e9)


# ---------------------------------------------------------------------------
# 5. __deepcopy__ shares .qec and .encoder by reference
# ---------------------------------------------------------------------------

def test_deepcopy_shares_qec_and_encoder_by_reference():
    alg = _make_setup_algorithm(seed=0)

    copied = copy.deepcopy(alg.qec_policy)

    assert copied is not alg.qec_policy
    assert copied.qec is alg.qec_policy.qec
    assert copied.encoder is alg.qec_policy.encoder
    assert id(copied.qec) == id(alg.qec_policy.qec)
    assert id(copied.encoder) == id(alg.qec_policy.encoder)


# ---------------------------------------------------------------------------
# 6. Checkpoint round-trip restores the encoder and QEC memory
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_restores_encoder_and_qec():
    src = _make_setup_algorithm(seed=0)
    obs = torch.randint(0, 256, (4, *OBS_SHAPE), dtype=torch.uint8)
    states = src.qec_policy.embed(obs)
    src.qec.add_batch(0, states[:2], torch.tensor([1.0, 2.0], dtype=torch.float64))

    before = src.qec_policy.embed(obs)

    state = src._get_training_state()
    assert "projection" not in state.extra, (
        "TrainingState.extra must not reference the old hardcoded-projection key"
    )
    assert "encoder_state" in state.extra

    # Target built with a different seed so its projection differs pre-load.
    dst = _make_setup_algorithm(seed=999)
    before_load = dst.qec_policy.embed(obs)
    assert not torch.equal(before_load, before), (
        "sanity check: different seeds must produce different projections"
    )

    dst._load_training_state(state)

    after = dst.qec_policy.embed(obs)
    assert torch.equal(after, before), (
        "embeddings must match pre- and post-round-trip once the encoder "
        "state has been restored"
    )

    for a in range(NUM_ACTIONS):
        assert dst.qec._sizes[a] == src.qec._sizes[a]


# ---------------------------------------------------------------------------
# 7. One integration step() call
# ---------------------------------------------------------------------------

def test_step_runs_and_returns_expected_metric_keys():
    alg = _make_setup_algorithm(seed=0, k=1)
    E, T = 1, 4

    obs = torch.randint(0, 256, (E, T, *OBS_SHAPE), dtype=torch.uint8)
    rewards = torch.ones(E, T)
    dones = torch.zeros(E, T, dtype=torch.bool)
    dones[0, 1] = True   # episode 1: t=0..1
    dones[0, 3] = True   # episode 2: t=2..3
    actions = torch.randint(0, NUM_ACTIONS, (E, T))

    batch = TensorDict(
        {
            "pixels": obs,
            "action": actions,
            "next": TensorDict(
                {
                    "reward": rewards.unsqueeze(-1).float(),
                    "done": dones.unsqueeze(-1),
                },
                batch_size=[E, T],
            ),
        },
        batch_size=[E, T],
    )

    metrics = alg.step(batch)

    assert set(metrics) == {"train/epsilon", "train/qec_size", "train/exact_hit_rate"}
    assert alg._collected_frames == E * T
