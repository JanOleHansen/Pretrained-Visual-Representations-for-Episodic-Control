"""NEC finalises N-step returns on a sliding window, not at episode end.

The return for step t needs only `r_t..r_{t+n-1}` and the bootstrap state
`s_{t+n}` — **not** the end of the episode.  Only the last `n_step` steps have
to wait.  `step()` therefore keeps a rolling window of at most `n_step` raw
frames per env instead of a whole episode, which with
`StepCounter.max_steps=27_000` is the difference between ~11 MB and ~3 GB per
env (24 GB across 8 envs, in a 32 GB container).

DND writes still happen at **episode end**, as Pritzel et al. specify: the
window changes when returns are *computed*, not when they are written.

These tests pin the behaviour against a closed form.  The fixture fills every
DND table with a single constant value `C`, so `max_a Q(s) = C` for every query
regardless of the embedding, which makes the expected return exactly:

    G_t = sum_{j<m} gamma^j r_{t+j}  +  bootstrap
    m   = min(n_step, T - t)
    bootstrap = gamma^n_step * C          if t + n_step <= T-1
              = gamma^(T-t)  * C          if truncated (done, not terminated)
              = 0                          if a true terminal

`init_random_frames` is set unreachably high so no gradient step runs: the
embedding network and the DND stay frozen for the whole episode, so a run split
across many batches must produce *bitwise the same* returns as one big batch.
That equality is the actual regression test for the window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, LazyTensorStorage, TensorDictReplayBuffer

from src.algorithms.nec import NECAlgorithm


OBS_SHAPE = (1, 4, 4)
NUM_ACTIONS = 2
EMBED_DIM = 8
K = 2
CONST_Q = 7.0
GAMMA = 0.5
N_STEP = 3


def _tiny_net(obs_shape, embedding_dim):
    """Flatten+Linear stand-in — these tests are about episode bookkeeping,
    not the CNN, and NatureEmbedding needs 84x84 inputs."""
    return nn.Sequential(
        nn.Flatten(), nn.Linear(int(np.prod(obs_shape)), embedding_dim)
    )


class _MockEnv:
    def __init__(self):
        self.observation_spec = Composite(
            obs=Bounded(low=0.0, high=1.0, shape=OBS_SHAPE, dtype=torch.float32)
        )
        self.action_spec = Categorical(n=NUM_ACTIONS)
        self.batch_size = torch.Size([])


def _make_alg() -> NECAlgorithm:
    torch.manual_seed(0)
    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=_tiny_net,
        obs_key="obs",
        embedding_dim=EMBED_DIM,
        dnd_capacity=256,
        k=K,
        n_step=N_STEP,
        gamma=GAMMA,
        init_random_frames=10**9,      # never reach the gradient loop
        num_updates=0,
        frames_per_batch=8,
    )
    alg.setup(_MockEnv)
    alg.replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=10_000, device="cpu")
    )

    # Constant-valued DND: max_a Q(s) == CONST_Q for every query, so the
    # bootstrap term is known exactly. Needs > k entries per action or
    # estimate_all returns +inf (optimistic init).
    keys = nn.functional.normalize(torch.randn(NUM_ACTIONS, K + 3, EMBED_DIM), dim=-1)
    for a in range(NUM_ACTIONS):
        alg.dnd.write_batch(a, keys[a], torch.full((K + 3,), CONST_Q), 1.0)
    assert all(s > K for s in alg.dnd._sizes)
    return alg


def _expected_returns(rewards: list[float], *, terminated: bool) -> np.ndarray:
    """Closed form for the fixture's constant-Q DND."""
    T = len(rewards)
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        m = min(N_STEP, T - t)
        g = sum(GAMMA**j * rewards[t + j] for j in range(m))
        if t + N_STEP <= T - 1:
            g += GAMMA**N_STEP * CONST_Q
        elif not terminated:
            g += GAMMA ** (T - t) * CONST_Q
        out[t] = g
    return out


def _batch(obs, actions, rewards, dones, terminated, next_obs):
    """One collector batch, shaped (1, T) as a single-env collector yields."""
    T = len(actions)
    return TensorDict(
        {
            "obs": obs.reshape(1, T, *OBS_SHAPE),
            "action": torch.tensor(actions).reshape(1, T).long(),
            "next": TensorDict(
                {
                    "obs": next_obs.reshape(1, T, *OBS_SHAPE),
                    "reward": torch.tensor(rewards).float().reshape(1, T, 1),
                    "done": torch.tensor(dones).bool().reshape(1, T, 1),
                    "terminated": torch.tensor(terminated).bool().reshape(1, T, 1),
                },
                batch_size=[1, T],
            ),
        },
        batch_size=[1, T],
    )


def _drive(alg, rewards, *, terminated, chunk):
    """Feed one episode of len(rewards) steps in `chunk`-sized batches.

    Returns the n-step returns in step order, as handed to the replay buffer.
    """
    T = len(rewards)
    torch.manual_seed(1)
    obs = torch.rand(T + 1, *OBS_SHAPE)          # obs[i] = s_i, obs[T] = s_T
    actions = [i % NUM_ACTIONS for i in range(T)]

    captured: list[np.ndarray] = []
    real_extend = alg.replay_buffer.extend
    alg.replay_buffer.extend = lambda td: (
        captured.append(td["n_step_return"].numpy().copy()), real_extend(td)
    )[1]

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        last = e == T
        alg.step(_batch(
            obs=obs[s:e],
            actions=actions[s:e],
            rewards=rewards[s:e],
            dones=[False] * (e - s - 1) + [last],
            terminated=[False] * (e - s - 1) + [last and terminated],
            next_obs=obs[s + 1:e + 1],
        ))
    return np.concatenate(captured) if captured else np.array([])


# ---------------------------------------------------------------------------
# 1. Returns match the closed form
# ---------------------------------------------------------------------------

def test_returns_match_closed_form_true_terminal():
    rewards = [1.0, 2.0, -1.0, 3.0, 0.0, 5.0, 2.0, 1.0]
    got = _drive(_make_alg(), rewards, terminated=True, chunk=len(rewards))
    np.testing.assert_allclose(
        got, _expected_returns(rewards, terminated=True), rtol=1e-5, atol=1e-5
    )


def test_returns_match_closed_form_truncated():
    """A StepCounter cutoff must bootstrap the tail, not assume zero future."""
    rewards = [1.0, 2.0, -1.0, 3.0, 0.0, 5.0, 2.0, 1.0]
    got = _drive(_make_alg(), rewards, terminated=False, chunk=len(rewards))
    np.testing.assert_allclose(
        got, _expected_returns(rewards, terminated=False), rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# 2. The window is the point: splitting the stream must change nothing
# ---------------------------------------------------------------------------

def test_returns_are_invariant_to_batch_chunking():
    """Same episode, delivered whole vs in slivers, must give the same returns.

    This is the regression test for the sliding window. Chunk sizes both below
    and above n_step exercise the "nothing matures yet" and "several mature at
    once" paths, and chunk=1 forces a step to sit in the window across many
    calls before it can be finalised.
    """
    rewards = [1.0, 2.0, -1.0, 3.0, 0.0, 5.0, 2.0, 1.0, 4.0, -2.0, 1.0, 0.0]
    reference = _drive(_make_alg(), rewards, terminated=True, chunk=len(rewards))

    for chunk in (1, 2, 3, 5, 7, 11):
        got = _drive(_make_alg(), rewards, terminated=True, chunk=chunk)
        assert len(got) == len(rewards), (
            f"chunk={chunk}: got {len(got)} returns for {len(rewards)} steps — "
            "the window dropped or duplicated transitions"
        )
        np.testing.assert_allclose(
            got, reference, rtol=1e-5, atol=1e-5,
            err_msg=f"chunk={chunk} disagrees with the single-batch result",
        )


# ---------------------------------------------------------------------------
# 3. Memory: the retained window is bounded by n_step
# ---------------------------------------------------------------------------

def test_carry_is_bounded_by_n_step_on_a_long_episode():
    """The whole point. A done-free stream must not accumulate raw frames."""
    alg = _make_alg()
    torch.manual_seed(2)
    chunk = 5
    n_chunks = 40                      # 200 steps, no episode end at all
    obs = torch.rand(chunk + 1, *OBS_SHAPE)

    for _ in range(n_chunks):
        alg.step(_batch(
            obs=obs[:chunk],
            actions=[0] * chunk,
            rewards=[1.0] * chunk,
            dones=[False] * chunk,
            terminated=[False] * chunk,
            next_obs=obs[1:chunk + 1],
        ))

    held = len(alg._carry[0]["obs"])
    assert held <= N_STEP + chunk, (
        f"carry holds {held} raw frames after {n_chunks * chunk} done-free "
        f"steps; must stay within n_step + one batch = {N_STEP + chunk}. "
        "The window is not being drained."
    )
    # The finalised triples still accumulate until the episode ends — that is
    # the deliberate trade (264 B/step instead of 113 KB/step).
    n_pending = sum(len(p[2]) for p in alg._carry[0]["pending"])
    assert n_pending == n_chunks * chunk - held


# ---------------------------------------------------------------------------
# 4. DND writes stay at episode end (paper §3.3), despite the sliding window
# ---------------------------------------------------------------------------

def test_dnd_is_not_written_until_the_episode_ends():
    alg = _make_alg()
    before = [s for s in alg.dnd._sizes]

    torch.manual_seed(3)
    obs = torch.rand(7, *OBS_SHAPE)
    for _ in range(4):                        # 24 done-free steps
        alg.step(_batch(
            obs=obs[:6], actions=[0, 1] * 3, rewards=[1.0] * 6,
            dones=[False] * 6, terminated=[False] * 6, next_obs=obs[1:7],
        ))
    assert alg.dnd._sizes == before, (
        "DND grew mid-episode — writes must be deferred to episode end"
    )

    alg.step(_batch(                          # now end it
        obs=obs[:6], actions=[0, 1] * 3, rewards=[1.0] * 6,
        dones=[False] * 5 + [True], terminated=[False] * 5 + [True],
        next_obs=obs[1:7],
    ))
    assert alg.dnd._sizes != before, "episode ended but nothing was written"


def test_multiple_episodes_in_one_batch():
    """Two dones inside a single collector batch must both be drained."""
    alg = _make_alg()
    torch.manual_seed(4)
    obs = torch.rand(9, *OBS_SHAPE)
    captured = []
    real = alg.replay_buffer.extend
    alg.replay_buffer.extend = lambda td: (
        captured.append(len(td)), real(td))[1]

    alg.step(_batch(
        obs=obs[:8], actions=[0, 1] * 4, rewards=[1.0] * 8,
        dones=[False, False, True, False, False, False, True, False],
        terminated=[False, False, True, False, False, False, True, False],
        next_obs=obs[1:9],
    ))
    # 3 steps in episode 1, 4 in episode 2, 1 left immature in the window.
    assert sum(captured) == 7, f"expected 7 finalised steps, got {sum(captured)}"
    assert len(alg._carry[0]["obs"]) == 1
