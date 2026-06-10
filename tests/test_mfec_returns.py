"""Unit tests for MFEC per-env return computation.

Verifies that:
  1. Discounted returns are computed per env with no cross-env contamination.
  2. Partial episodes at batch boundaries are buffered and not used until done.
  3. For E=1, the new per-env path produces returns identical to the old
     single-lfilter path on a contiguous batch (regression guard).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter
from tensordict import TensorDict

from src.algorithms.mfec import MFECAlgorithm, QEC, QECPolicy


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

class _MockGreedy:
    """Minimal stand-in for EGreedyModule (no-op, fixed eps)."""
    eps = 0.5
    def step(self, n: int) -> None:
        pass


def _make_algorithm(
    *,
    gamma: float = 0.9,
    E: int = 2,
    num_actions: int = 3,
    state_dim: int = 4,
    obs_flat_dim: int = 4,
    buffer_size: int = 200,
    k: int = 1,
) -> MFECAlgorithm:
    """Build a MFECAlgorithm with internals wired directly (no real TorchRL env)."""
    alg = MFECAlgorithm(
        device=None,
        obs_key="obs",
        buffer_size=buffer_size,
        k=k,
        state_dim=state_dim,
        gamma=gamma,
        frames_per_batch=10,
        key_scale=1e5,
    )
    dev = torch.device("cpu")
    alg._buffer_device    = dev
    alg._num_actions      = num_actions
    alg._num_envs         = E
    alg._carry            = [None] * E
    alg._collected_frames = 0

    # Identity projection: first state_dim pixels are the state.
    proj = np.zeros((obs_flat_dim, state_dim), dtype=np.float32)
    for i in range(min(obs_flat_dim, state_dim)):
        proj[i, i] = 1.0
    alg.projection = proj

    alg.qec        = QEC(num_actions, buffer_size, k, dev, key_scale=1e5)
    alg.qec_policy = QECPolicy(alg.qec, alg.projection, num_actions)
    alg.greedy_module = _MockGreedy()
    return alg


def _make_batch(
    obs:     torch.Tensor,   # (E, T, obs_flat_dim) or (T, obs_flat_dim)
    rewards: torch.Tensor,   # (E, T) or (T,) float
    dones:   torch.Tensor,   # (E, T) or (T,) bool
    actions: torch.Tensor,   # (E, T) or (T,) long
) -> TensorDict:
    """Wrap raw tensors into the TensorDict shape that SyncDataCollector produces."""
    bs = obs.shape[:-1]  # (E, T) or (T,)
    return TensorDict(
        {
            "obs":    obs,
            "action": actions,
            "next": TensorDict(
                {
                    "reward": rewards.unsqueeze(-1).float(),   # (..., 1)
                    "done":   dones.unsqueeze(-1),             # (..., 1) bool
                },
                batch_size=list(bs),
            ),
        },
        batch_size=list(bs),
    )


def _disc_return(rewards: list[float], gamma: float) -> list[float]:
    """Reference: discounted return via scipy lfilter (same as the algorithm uses)."""
    r = np.array(rewards, dtype=np.float64)
    return list(lfilter([1.0], [1.0, -gamma], r[::-1])[::-1])


def _capture_G(alg: MFECAlgorithm) -> list[float]:
    """Monkey-patch add_batch to collect all G values passed to QEC."""
    captured: list[float] = []
    orig = alg.qec.add_batch

    def _cap(action, states, values):
        captured.extend(values.cpu().tolist())
        orig(action, states, values)

    alg.qec.add_batch = _cap  # type: ignore[method-assign]
    return captured


# ---------------------------------------------------------------------------
# Test 1 — no cross-env contamination
# ---------------------------------------------------------------------------

def test_per_env_returns_no_cross_env_contamination():
    """G_t is computed from each env's own rewards; other envs are never mixed in."""
    gamma = 0.9
    E, T  = 2, 4

    # env 0: episodes [0..1] (done at t=1) and [2..3] (done at t=3)
    # env 1: episode  [0..3] (done at t=3)
    #
    # Cross-env contamination would make env-0's G[2] depend on env-1's r[0..1],
    # which is wrong.  We detect this by checking G values match hand-computation.

    obs = torch.arange(E * T * 4, dtype=torch.float32).reshape(E, T, 4)
    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],   # env 0
        [5.0, 6.0, 7.0, 8.0],   # env 1
    ])
    dones = torch.zeros(E, T, dtype=torch.bool)
    dones[0, 1] = True
    dones[0, 3] = True
    dones[1, 3] = True
    actions = torch.zeros(E, T, dtype=torch.long)   # all action 0

    alg = _make_algorithm(gamma=gamma, E=E)
    captured = _capture_G(alg)

    alg.step(_make_batch(obs, rewards, dones, actions))

    # Hand-computed expected returns (per env, per episode):
    #   env 0, ep 1: r=[1,2]         → G = [1+0.9*2, 2]      = [2.8, 2.0]
    #   env 0, ep 2: r=[3,4]         → G = [3+0.9*4, 4]      = [6.6, 4.0]
    #   env 1, ep 1: r=[5,6,7,8]     → G = lfilter(...)
    G_e0_ep1 = _disc_return([1.0, 2.0], gamma)
    G_e0_ep2 = _disc_return([3.0, 4.0], gamma)
    G_e1_ep1 = _disc_return([5.0, 6.0, 7.0, 8.0], gamma)
    expected = sorted(G_e0_ep1 + G_e0_ep2 + G_e1_ep1)

    assert len(captured) == len(expected), (
        f"Expected {len(expected)} transitions in QEC, got {len(captured)}"
    )
    np.testing.assert_allclose(
        sorted(captured), expected, rtol=1e-6,
        err_msg="G values deviate from hand-computed per-env returns",
    )

    # If cross-env contamination occurred, env-0's ep-2 G[0] would be:
    #   G = 3 + 0.9*(4 + 0.9*5 + ...) — much larger than 6.6
    # Check that 6.6 is present and nothing larger dominates env-0's portion.
    g_e0_ep2_expected = sorted(G_e0_ep2)
    assert max(captured) < 30.0, (
        "Suspiciously large G — probable cross-env contamination"
    )


# ---------------------------------------------------------------------------
# Test 2 — partial episode buffered, no QEC update until done
# ---------------------------------------------------------------------------

def test_partial_episode_carry():
    """Incomplete episodes are buffered; QEC is updated only after done=True."""
    gamma = 0.9
    E, T  = 2, 3

    # Batch 1: no done in either env → nothing should enter QEC.
    obs1     = torch.randn(E, T, 4)
    rewards1 = torch.ones(E, T)
    dones1   = torch.zeros(E, T, dtype=torch.bool)
    actions1 = torch.zeros(E, T, dtype=torch.long)

    alg = _make_algorithm(gamma=gamma, E=E)
    alg.step(_make_batch(obs1, rewards1, dones1, actions1))

    assert all(alg._carry[e] is not None for e in range(E)), (
        "Carry should be set for every env when no done occurred"
    )
    assert all(alg.qec._sizes[a] == 0 for a in range(alg._num_actions)), (
        "QEC must not be updated before any episode completes"
    )

    # Batch 2: done at t=2 in both envs.
    obs2     = torch.randn(E, T, 4)
    rewards2 = torch.ones(E, T) * 2.0
    dones2   = torch.zeros(E, T, dtype=torch.bool)
    dones2[:, 2] = True
    actions2 = torch.zeros(E, T, dtype=torch.long)

    captured = _capture_G(alg)
    alg.step(_make_batch(obs2, rewards2, dones2, actions2))

    # Each env now completes one episode of length 6 (3 from carry + 3 from batch 2).
    # Expected G for the 6-step episode with r=[1,1,1,2,2,2], gamma=0.9:
    r_full = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    G_expected = _disc_return(r_full, gamma)

    # 2 envs × 6 steps each = 12 transitions total in QEC.
    assert len(captured) == 2 * len(G_expected), (
        f"Expected {2 * len(G_expected)} QEC updates, got {len(captured)}"
    )

    # Both envs see identical rewards and must produce identical G distributions.
    half = len(G_expected)
    np.testing.assert_allclose(
        sorted(captured[:half]), sorted(G_expected), rtol=1e-6,
    )
    np.testing.assert_allclose(
        sorted(captured[half:]), sorted(G_expected), rtol=1e-6,
    )

    # After batch 2 consumed all steps up to done at t=2, carry should be None
    # (last step was t=2 which is done, so nothing remains past it).
    assert all(alg._carry[e] is None for e in range(E)), (
        "Carry should be cleared when the done falls on the last step"
    )


# ---------------------------------------------------------------------------
# Test 3 — E=1 regression: new path matches old lfilter path
# ---------------------------------------------------------------------------

def test_single_env_matches_old_flattened_lfilter():
    """For E=1, the per-env path produces G identical to the old flat lfilter."""
    gamma = 0.9
    E, T  = 1, 8

    rewards_1d = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.1, 4.0]
    # done at t=3 and t=7 (two complete episodes, no trailing partial)
    dones_1d = [False, False, False, True, False, False, False, True]

    rewards = torch.tensor(rewards_1d).unsqueeze(0)       # (1, T)
    dones   = torch.tensor(dones_1d, dtype=torch.bool).unsqueeze(0)  # (1, T)
    obs     = torch.arange(T * 4, dtype=torch.float32).reshape(1, T, 4)
    actions = torch.zeros(1, T, dtype=torch.long)

    alg = _make_algorithm(gamma=gamma, E=E)
    captured = _capture_G(alg)
    alg.step(_make_batch(obs, rewards, dones, actions))

    # Reference: old flat-lfilter approach on the same data.
    rewards_np = np.array(rewards_1d, dtype=np.float64)
    dones_np   = np.array(dones_1d, dtype=bool)
    ends       = np.flatnonzero(dones_np)
    G_ref      = np.empty(T, dtype=np.float64)
    ep_start   = 0
    for ep_end in ends:
        r = rewards_np[ep_start:ep_end + 1]
        G_ref[ep_start:ep_end + 1] = lfilter([1.0], [1.0, -gamma], r[::-1])[::-1]
        ep_start = ep_end + 1

    assert len(captured) == T, f"Expected {T} updates, got {len(captured)}"
    np.testing.assert_allclose(
        sorted(captured), sorted(G_ref.tolist()), rtol=1e-6,
        err_msg="E=1 per-env path diverges from reference lfilter",
    )


# ---------------------------------------------------------------------------
# Test 4 — done not at batch boundary (Bug 1 trigger)
# ---------------------------------------------------------------------------

def test_done_mid_batch_no_contamination():
    """A done mid-batch must not let the next env's rewards leak into the return."""
    gamma = 0.9
    E, T  = 2, 6

    # env 0: done at t=2 only; t=3-5 are partial (buffered, not yet QEC'd)
    # env 1: done at t=5 only
    rewards = torch.tensor([
        [1.0, 1.0, 1.0, 9.0, 9.0, 9.0],   # env 0: ep ends at t=2; large tail
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],   # env 1: one episode
    ])
    dones = torch.zeros(E, T, dtype=torch.bool)
    dones[0, 2] = True
    dones[1, 5] = True
    obs     = torch.randn(E, T, 4)
    actions = torch.zeros(E, T, dtype=torch.long)

    alg = _make_algorithm(gamma=gamma, E=E)
    captured = _capture_G(alg)
    alg.step(_make_batch(obs, rewards, dones, actions))

    # env 0's completed episode is t=0..2 with r=[1,1,1].
    G_e0 = _disc_return([1.0, 1.0, 1.0], gamma)
    # env 1's completed episode is t=0..5 with r=[2,2,2,2,2,2].
    G_e1 = _disc_return([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], gamma)

    expected = sorted(G_e0 + G_e1)
    assert len(captured) == len(expected), (
        f"Expected {len(expected)} QEC updates, got {len(captured)}"
    )
    np.testing.assert_allclose(
        sorted(captured), expected, rtol=1e-6,
        err_msg="G values contaminated across env boundary at mid-batch done",
    )

    # env 0's episode should have max G = G_e0[0], not inflated by the 9.0 tail.
    max_G_e0 = G_e0[0]   # G at t=0 is the largest
    max_G_e1 = G_e1[0]
    # If contaminated, some value > max(max_G_e0, max_G_e1) + epsilon would appear.
    tolerance = 1e-4
    assert max(captured) <= max(max_G_e0, max_G_e1) + tolerance, (
        f"G={max(captured):.4f} exceeds expected max {max(max_G_e0, max_G_e1):.4f} "
        f"— probable cross-env contamination from the large-reward tail"
    )
