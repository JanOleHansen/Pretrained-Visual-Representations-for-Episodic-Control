"""Regression tests for MFEC optimistic-initialisation tie-breaking.

Bug being guarded against
-------------------------
``QEC.estimate_all`` returns ``+inf`` for actions whose buffer holds ``<= k``
entries, so untried actions are always preferred (Blundell et al. 2016 §2).
``QECPolicy.forward`` used to map *every* ``+inf`` onto the same ``1e9``
constant, which made all untried actions exact ties — and ``QValueActor``'s
argmax resolves ties by lowest index.

Measured on a 9-action Ms. Pac-Man spec: with an empty QEC the policy emitted
action 0 for 499/500 states; once action 0's buffer passed ``k`` it emitted
action 1 for 499/500.  The agent played a single fixed action for a whole
episode, cycling 0..8, and seeded the QEC with 9 degenerate single-action
trajectories whose max-returns then persist forever (Eq. 1 never decreases).

The fix adds independent uniform jitter to the optimistic entries only.  Note
the jitter has to be *large* (1e6): float32's ULP at 1e9 is 64.0, so jitter
drawn from ``uniform(0, 1)`` would round back to exactly 1e9 and restore the
tie.  ``test_jitter_survives_float32_rounding`` pins that specifically.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms.mfec import QEC, QECPolicy


NUM_ACTIONS = 9        # ALE/MsPacman-v5
STATE_DIM = 16
K = 11
NUM_STATES = 500


def _make_policy(capacity: int = 1_000) -> QECPolicy:
    qec = QEC(
        num_actions=NUM_ACTIONS,
        capacity=capacity,
        k=K,
        device=torch.device("cpu"),
    )
    # encoder is only used by QECPolicy.embed(), which these tests never call.
    return QECPolicy(qec, encoder=None, num_actions=NUM_ACTIONS)


def _populate(qec: QEC, action: int, n: int, *, value: float = 1.0) -> None:
    """Give ``action`` enough entries that estimate_all stops returning +inf."""
    states = torch.randn(n, STATE_DIM, dtype=torch.float32)
    values = torch.full((n,), value, dtype=torch.float64)
    qec.add_batch(action, states, values)


def _greedy_actions(policy: QECPolicy, states: torch.Tensor) -> torch.Tensor:
    """What QValueActor's argmax would select for each state."""
    return policy(states).argmax(dim=-1)


# ---------------------------------------------------------------------------

def test_empty_qec_gives_roughly_uniform_actions():
    torch.manual_seed(0)
    policy = _make_policy()
    states = torch.randn(NUM_STATES, STATE_DIM, dtype=torch.float32)

    counts = torch.bincount(
        _greedy_actions(policy, states), minlength=NUM_ACTIONS
    )

    assert int(counts.sum()) == NUM_STATES
    # Every action must actually get played.  Expected share is 500/9 ~= 55.6
    # with sd ~= 7.0; these bounds are ~5 sd wide, so they catch the degenerate
    # regime (one action taking ~all 500) without being seed-fragile.
    assert int(counts.min()) >= 20, f"action starved: {counts.tolist()}"
    assert int(counts.max()) <= 100, f"action dominates: {counts.tolist()}"

    # The specific pre-fix signature: 499/500 on the lowest index.
    assert int(counts[0]) < NUM_STATES // 2


def test_partially_populated_qec_spreads_over_the_remaining_actions():
    """The second half of the observed failure: all-action-1 after action 0 fills."""
    torch.manual_seed(0)
    policy = _make_policy()
    _populate(policy.qec, action=0, n=K + 20)

    states = torch.randn(NUM_STATES, STATE_DIM, dtype=torch.float32)
    actions = _greedy_actions(policy, states)
    counts = torch.bincount(actions, minlength=NUM_ACTIONS)

    # Action 0 now has a finite (small) estimate, so the 8 still-optimistic
    # actions all beat it — but they must share the mass, not all land on 1.
    assert int(counts[0]) == 0, "finite estimate beat an optimistic one"
    assert int(counts[1]) <= 150, f"collapsed onto lowest index: {counts.tolist()}"
    assert int(counts[1:].min()) >= 20, f"action starved: {counts.tolist()}"


def test_jitter_survives_float32_rounding():
    """1e9 + uniform(0,1) would round away; the jitter must be big enough."""
    torch.manual_seed(0)
    policy = _make_policy()
    states = torch.randn(NUM_STATES, STATE_DIM, dtype=torch.float32)

    q = policy(states)

    assert q.dtype == torch.float32
    assert torch.isfinite(q).all(), "no +inf may reach QValueActor"
    # Distinct representable values actually exist after rounding.
    assert q.unique().numel() > NUM_STATES, "jitter collapsed to a constant"
    # ...and every optimistic entry stays comfortably above any real return
    # and above the trainer's sentinel threshold.  The upper bound carries one
    # ULP of slack: float32 rounds to nearest, so a sum just under
    # 1e9 + 1e6 can land up to 32.0 above it.
    assert float(q.min()) >= QECPolicy.OPTIMISTIC_VALUE
    assert float(q.max()) <= (
        QECPolicy.OPTIMISTIC_VALUE + QECPolicy.OPTIMISTIC_JITTER + 64.0
    )


def test_finite_estimates_are_never_perturbed():
    """Eval determinism: a QEC with real information must be reproducible."""
    torch.manual_seed(0)
    policy = _make_policy()
    for a in range(NUM_ACTIONS):
        _populate(policy.qec, action=a, n=K + 20, value=float(a))

    states = torch.randn(NUM_STATES, STATE_DIM, dtype=torch.float32)

    first = policy(states)
    second = policy(states)

    assert torch.isfinite(first).all()
    assert float(first.max()) < QECPolicy.OPTIMISTIC_VALUE, "no sentinels expected"
    # Bit-identical across calls — no RNG touched the finite path.
    assert torch.equal(first, second)
