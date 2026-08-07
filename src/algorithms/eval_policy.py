"""Evaluation-time ε-greedy shared by the episodic-control algorithms.

Both MFEC and NEC need a *stochastic* evaluation policy, for the same two
reasons, so the module lives here rather than being duplicated in each
algorithm file.
"""
from __future__ import annotations

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule


class EvalEGreedyModule(EGreedyModule):
    """ε-greedy that applies ε regardless of the ambient ``ExplorationType``.

    ``BaseTrainer.evaluate()`` wraps its rollout in
    ``set_exploration_type(ExplorationType.MODE)``, and torchrl's
    ``EGreedyModule.forward`` is gated on
    ``exploration_type() in (ExplorationType.RANDOM, None)`` — so a stock
    ``EGreedyModule`` placed in the eval chain is silently a **no-op**.  Simply
    adding one to the eval policy therefore does not make evaluation
    stochastic; this subclass forces the gate open.

    Why episodic control needs ε at evaluation
    ------------------------------------------
    1. **Determinism collapses the eval sample.**  With
       ``repeat_action_probability=0.0`` (which MFEC requires — Blundell et al.
       footnote 1 — and which the NEC configs also set, to match the pre-2018
       protocol) ALE is deterministic, and Ms. Pac-Man's opening is insensitive
       to ``NoopResetEnv``.  A deterministic policy therefore replays **the
       same trajectory every episode**: ``eval/return_std`` is identically 0
       and ``num_eval_episodes`` silently collapses to a single sample at N
       times the cost.  Observed on both algorithms.
    2. **The argmax policy is not the policy being trained.**  Both methods
       act ε-greedily and store returns generated that way.  MFEC's QEC values
       are max-over-returns and never decrease (Eq. 1), so exploiting them with
       no ε lands the agent in states whose stored value came from one lucky
       trajectory it can no longer reproduce.  NEC's DND values are optimistic
       in a milder way (+inf until an action holds > k entries), but the same
       train/eval mismatch applies.

    Measured on Ms. Pac-Man with an identical QEC (MFEC):

        get_policy()          / MODE     mean=380.0  std=0.000   (5/5 identical)
        get_explore_policy()  / RANDOM   mean=448.0  std=248.4
        get_explore_policy()  / MODE     mean=380.0  std=0.000   <- eps ignored

    The third row is why this subclass exists.

    Construct it with ``eps_init == eps_end`` and never call ``.step()`` — the
    evaluation rate is a constant, not a schedule.  ``eps = 0.0`` makes it a
    no-op (``rand() < 0`` is never true), restoring pure argmax.
    """

    def forward(self, tensordict):
        with set_exploration_type(ExplorationType.RANDOM):
            return super().forward(tensordict)
