"""Regression tests for NEC optimistic-initialisation tie-breaking.

Bug being guarded against
-------------------------
``DND.estimate_all`` returns ``+inf`` for actions whose table holds ``<= k``
entries, so under-populated actions are always preferred (Pritzel et al. 2017
§3.2).  ``DNDPolicy.forward`` used to map *every* ``+inf`` onto the same ``1e9``
constant, which made those actions exact ties — and ``QValueActor``'s argmax
resolves ties by lowest index.

This is the same defect ``tests/test_mfec_optimistic_tiebreak.py`` pins for
MFEC.  It was left in NEC deliberately for a while (AGENTS.md recorded it as a
known latent defect, on the grounds that the failure had only been *confirmed*
experimentally for MFEC).  Confirmed for NEC too, on a 9-action Ms. Pac-Man
spec over 500 states:

    empty DND                        action 0 for 500/500
    actions 0-3 above k, 4-8 below   action 4 for 500/500

i.e. the policy plays one fixed action until that action's table passes ``k``,
then moves to the next — seeding the DND with ``|A|`` degenerate single-action
trajectories at exactly the moment it is first being populated.

The fix adds independent uniform jitter to the optimistic entries only.  The
jitter has to be *large* (1e6): float32's ULP at 1e9 is 64.0, so jitter drawn
from ``uniform(0, 1)`` would round back to exactly 1e9 and restore the tie.
``test_jitter_survives_float32_rounding`` pins that specifically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.algorithms.nec import DND, DNDPolicy


NUM_ACTIONS = 9        # ALE/MsPacman-v5
OBS_SHAPE = (4, 8, 8)  # (C, H, W) — small stand-in for the 4x84x84 Atari stack
EMBEDDING_DIM = 16
K = 5
NUM_STATES = 500


class _FixedEmbedding(nn.Module):
    """Deterministic (obs -> embedding) stand-in for ``NatureEmbedding``.

    ``DNDPolicy`` normalises the output itself, so this only has to be a
    fixed, differentiable map of the right shape.
    """

    def __init__(self, out_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.lin = nn.Linear(int(np.prod(OBS_SHAPE)), out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.lin(obs.reshape(obs.shape[0], -1))


def _make_policy(capacity: int = 1_000) -> DNDPolicy:
    dnd = DND(
        num_actions=NUM_ACTIONS,
        capacity=capacity,
        k=K,
        kernel_delta=1e-3,
        device=torch.device("cpu"),
    )
    return DNDPolicy(_FixedEmbedding(), dnd, NUM_ACTIONS)


def _populate(dnd: DND, action: int, n: int, *, value: float = 1.0) -> None:
    """Give ``action`` enough entries that estimate_all stops returning +inf."""
    keys = nn.functional.normalize(
        torch.randn(n, EMBEDDING_DIM, dtype=torch.float32), dim=-1
    )
    values = torch.full((n,), value, dtype=torch.float32)
    dnd.write_batch(action, keys, values, 0.1)


def _observations(n: int = NUM_STATES) -> torch.Tensor:
    return torch.randn(n, *OBS_SHAPE, dtype=torch.float32)


def _greedy_actions(policy: DNDPolicy, obs: torch.Tensor) -> torch.Tensor:
    """What QValueActor's argmax would select for each observation."""
    return policy(obs).argmax(dim=-1)


# ---------------------------------------------------------------------------

def test_empty_dnd_gives_roughly_uniform_actions():
    torch.manual_seed(0)
    policy = _make_policy()

    counts = torch.bincount(
        _greedy_actions(policy, _observations()), minlength=NUM_ACTIONS
    )

    assert int(counts.sum()) == NUM_STATES
    # Every action must actually get played.  Expected share is 500/9 ~= 55.6
    # with sd ~= 7.0; these bounds are ~5 sd wide, so they catch the degenerate
    # regime (one action taking ~all 500) without being seed-fragile.
    assert int(counts.min()) >= 20, f"action starved: {counts.tolist()}"
    assert int(counts.max()) <= 100, f"action dominates: {counts.tolist()}"

    # The specific pre-fix signature: 500/500 on the lowest index.
    assert int(counts[0]) < NUM_STATES // 2


def test_partially_populated_dnd_spreads_over_the_remaining_actions():
    """The second half of the observed failure: all-action-4 after 0..3 fill."""
    torch.manual_seed(0)
    policy = _make_policy()
    for a in range(4):
        _populate(policy.dnd, action=a, n=K + 20, value=1.0)

    counts = torch.bincount(
        _greedy_actions(policy, _observations()), minlength=NUM_ACTIONS
    )

    # Actions 0-3 now have finite (small) estimates, so the 5 still-optimistic
    # actions all beat them — but they must share the mass, not all land on 4.
    assert int(counts[:4].sum()) == 0, "finite estimate beat an optimistic one"
    assert int(counts[4]) <= 200, f"collapsed onto lowest index: {counts.tolist()}"
    assert int(counts[4:].min()) >= 40, f"action starved: {counts.tolist()}"


def test_jitter_survives_float32_rounding():
    """1e9 + uniform(0,1) would round away; the jitter must be big enough."""
    torch.manual_seed(0)
    policy = _make_policy()

    q = policy(_observations())

    assert q.dtype == torch.float32
    assert torch.isfinite(q).all(), "no +inf may reach QValueActor"
    # Distinct representable values actually exist after rounding.
    assert q.unique().numel() > NUM_STATES, "jitter collapsed to a constant"
    # ...and every optimistic entry stays comfortably above any real return and
    # above StepTrainer._OPTIMISTIC_Q_THRESHOLD (1e8), which is what keeps these
    # values out of train/q_values.  The upper bound carries one ULP of slack:
    # float32 rounds to nearest, so a sum just under 1e9 + 1e6 can land up to
    # 32.0 above it.
    assert float(q.min()) >= DNDPolicy.OPTIMISTIC_VALUE
    assert float(q.max()) <= (
        DNDPolicy.OPTIMISTIC_VALUE + DNDPolicy.OPTIMISTIC_JITTER + 64.0
    )
    assert float(q.min()) > 1e8, "jittered sentinels must stay above the trainer threshold"


def test_finite_estimates_are_never_perturbed():
    """Eval determinism: a DND with real information must be reproducible."""
    torch.manual_seed(0)
    policy = _make_policy()
    for a in range(NUM_ACTIONS):
        _populate(policy.dnd, action=a, n=K + 20, value=float(a))

    obs = _observations()

    first = policy(obs)
    second = policy(obs)

    assert torch.isfinite(first).all()
    assert float(first.max()) < DNDPolicy.OPTIMISTIC_VALUE, "no sentinels expected"
    # Bit-identical across calls — no RNG touched the finite path.
    assert torch.equal(first, second)
