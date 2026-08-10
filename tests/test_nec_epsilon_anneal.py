"""NEC's epsilon anneal: closed form instead of torchrl's per-frame loop.

`EGreedyModule.step(frames)` iterates `frames` times in Python, one tensor op
each, so annealing a 1600-frame collector batch costs 1600 GPU ops to evaluate
what is a closed-form linear ramp (~36 ms/batch, ~23 s per 1M frames).
`NECAlgorithm.step` computes it directly.

The schedule is `eps <- max(eps_end, eps - delta)` per frame with `delta > 0`.
Because the sequence is monotone decreasing and the clamp is from below, `n`
iterations are exactly `max(eps_end, eps - n*delta)` in real arithmetic — so the
closed form is not an approximation of the loop, it is the loop's exact value.

In float32 the two differ, and **the loop is the less accurate of the two**:
1600 successive subtractions accumulate rounding error, so torchrl's result
drifts from the true ramp by ~2e-5 after one batch and ~4e-4 after forty, always
leaving epsilon slightly HIGHER than intended (annealing slower than configured).
These tests therefore pin the closed form against exact float64 arithmetic
rather than against torchrl's loop.
"""
from __future__ import annotations

import pytest
import torch
from torchrl.data import Categorical
from torchrl.modules import EGreedyModule

EPS_INIT, EPS_END, ANNEAL = 1.0, 0.001, 50_000


def _module():
    return EGreedyModule(spec=Categorical(n=4), eps_init=EPS_INIT,
                         eps_end=EPS_END, annealing_num_steps=ANNEAL)


def _closed_form(chunks):
    """Exactly what NECAlgorithm.step does."""
    m = _module()
    for n in chunks:
        delta = (m.eps_init - m.eps_end) / m.annealing_num_steps
        m.eps.data.copy_(torch.maximum(m.eps_end, m.eps - delta * n))
    return float(m.eps)


def _torchrl_loop(chunks):
    m = _module()
    for n in chunks:
        m.step(n)
    return float(m.eps)


def _exact(chunks):
    """Ground truth in float64."""
    eps = EPS_INIT
    delta = (EPS_INIT - EPS_END) / ANNEAL
    for n in chunks:
        for _ in range(n):
            eps = max(EPS_END, eps - delta)
    return eps


CHUNKS = [
    [1600],            # one collector batch
    [1600] * 5,        # several
    [800] * 40,        # where the loop's drift is largest
    [1] * 5,           # per-frame stepping
    [ANNEAL],          # exactly the annealing horizon
    [123_456],         # far past it -> must clamp
]


@pytest.mark.parametrize("chunks", CHUNKS)
def test_closed_form_matches_exact_arithmetic(chunks):
    assert _closed_form(chunks) == pytest.approx(_exact(chunks), abs=1e-6)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_closed_form_is_at_least_as_accurate_as_torchrl(chunks):
    exact = _exact(chunks)
    assert abs(_closed_form(chunks) - exact) <= abs(_torchrl_loop(chunks) - exact) + 1e-12


def test_agrees_with_torchrl_to_the_loops_own_precision():
    """Same schedule, not a behaviour change — differences are float32 noise."""
    for chunks in CHUNKS:
        assert _closed_form(chunks) == pytest.approx(_torchrl_loop(chunks), abs=1e-3)


def test_clamps_at_eps_end_and_never_below():
    m = _module()
    delta = (m.eps_init - m.eps_end) / m.annealing_num_steps
    m.eps.data.copy_(torch.maximum(m.eps_end, m.eps - delta * 10_000_000))
    assert float(m.eps) == pytest.approx(EPS_END)
