"""Regression test: NEC's DND kernel must not be dominated by ``kernel_delta``.

Root cause of the "reward stuck at the minimum" bug
-----------------------------------------------------
``DND.estimate_all`` / ``DND._gradient_step`` weight neighbours by

    w_i = 1 / (‖h - h_i‖^2 + kernel_delta),   kernel_delta = 1e-3  (paper §4, Eq. 5)

``NatureEmbedding``'s output is an *unconstrained* ``nn.Linear`` applied to
CNN features. At initialisation (and for a long time into training, since
nothing in the regression loss pressures embeddings apart in L2 norm) its
outputs have very small magnitude, and two different Atari frames --
differing only in a small foreground region against a mostly-static
background -- routinely land at a *raw* squared distance below 1e-3. When
that happens ``kernel_delta`` dominates the denominator instead of acting as
a small numerical-stability epsilon, so every neighbour gets an
(near-)identical weight regardless of how close it actually is. The
kernel-weighted average then collapses to an almost-uniform average over the
k nearest neighbours for *every* query, so Q(s, a) stops depending on s in
any useful way -- gradient descent cannot fix this because the collapsed
kernel destroys the gradient signal (∂w_i/∂h vanishes in relative terms once
all distances are pinned to the delta floor), regardless of how long
training runs. Confirmed empirically: an un-normalised NEC run on Pong
stayed pinned at episode_reward == -21 (worst possible) for 45k frames even
with epsilon fully annealed to 0.01 and a continuously growing DND.

The fix (``DNDPolicy.forward`` / ``NECAlgorithm.step`` / ``._gradient_step``)
L2-normalises the embedding to the unit sphere before every DND read or
write. This lower-bounds the squared distance between any two *distinct*
unit vectors well above ``kernel_delta``, so the kernel actually
discriminates between near and far neighbours again.

This test guards the numerical property directly (not the full training
loop, which is covered by the smoke tests): for two different raw CNN
embeddings that are realistically close together (a common situation early
in training), the value NECAlgorithm actually feeds into the DND kernel
must have squared distance well above ``kernel_delta`` -- i.e. the policy/
gradient path must normalise before computing distances.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.nec import DND, DNDPolicy
from src.networks import NatureEmbedding


KERNEL_DELTA = 1e-3


def test_raw_nature_embedding_can_collapse_below_kernel_delta():
    """Sanity check on the failure mode itself (not the fix).

    Two Atari-like frames that differ only in a small foreground region
    (realistic frame-to-frame delta) can produce a *raw*, unnormalised
    NatureEmbedding squared distance smaller than kernel_delta. This is the
    condition that breaks the DND kernel if nothing normalises the output.
    """
    torch.manual_seed(0)
    net = NatureEmbedding((4, 84, 84), 64)

    base = torch.zeros(1, 4, 84, 84)
    base[..., 40:44, 10:12] = 1.0  # static paddle region

    frame_a = base.clone()
    frame_a[..., 20:22, 50:52] = 1.0  # ball position A

    frame_b = base.clone()
    frame_b[..., 60:62, 50:52] = 1.0  # ball position B (different optimal action)

    with torch.no_grad():
        ha = net(frame_a)
        hb = net(frame_b)

    raw_sq_dist = ((ha - hb) ** 2).sum().item()
    assert raw_sq_dist < KERNEL_DELTA, (
        "This test documents the failure precondition: if this ever stops "
        "being true for NatureEmbedding's default init, the kernel-collapse "
        "bug this test suite guards against may no longer reproduce this way."
    )


def test_dnd_policy_normalises_before_kernel_lookup():
    """DNDPolicy.forward must normalise embeddings so kernel distances clear
    kernel_delta by a healthy margin, even when the embedding network's raw
    output is nearly identical for genuinely different inputs.
    """
    torch.manual_seed(0)
    embedding_net = NatureEmbedding((4, 84, 84), 64)
    dnd = DND(num_actions=2, capacity=100, k=1, kernel_delta=KERNEL_DELTA,
              device=torch.device("cpu"))
    policy = DNDPolicy(embedding_net, dnd, num_actions=2)

    base = torch.zeros(2, 4, 84, 84)
    base[:, :, 40:44, 10:12] = 1.0
    obs = base.clone()
    obs[0, :, 20:22, 50:52] = 1.0
    obs[1, :, 60:62, 50:52] = 1.0

    # Recreate exactly what DNDPolicy.forward feeds into the DND, by hooking
    # dnd.estimate_all to capture its input.
    captured = {}
    orig_estimate_all = dnd.estimate_all

    def _spy(queries):
        captured["h"] = queries.clone()
        return orig_estimate_all(queries)

    dnd.estimate_all = _spy  # type: ignore[method-assign]

    policy(obs)

    h = captured["h"]
    assert h.shape == (2, 64)

    sq_dist = ((h[0] - h[1]) ** 2).sum().item()
    assert sq_dist > 10 * KERNEL_DELTA, (
        f"Squared distance fed into the DND kernel is {sq_dist:.6g}, which is "
        f"not safely above kernel_delta={KERNEL_DELTA}. This means the "
        "inverse-distance kernel w_i = 1/(dist^2 + kernel_delta) is dominated "
        "by kernel_delta instead of the true distance, collapsing kNN lookups "
        "to a near-uniform average regardless of query -- the DNDPolicy must "
        "normalise embeddings (e.g. F.normalize(h, dim=-1)) before calling "
        "dnd.estimate_all()."
    )

    # Direct check: DNDPolicy's *output* embeddings should be unit-norm.
    norms = h.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_gradient_step_normalises_before_kernel_lookup():
    """NECAlgorithm._gradient_step must also normalise -- it recomputes
    distances independently (for the differentiable path) and would
    otherwise be just as vulnerable to kernel collapse as the policy path.
    """
    import numpy as np
    from tensordict import TensorDict
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

    from src.algorithms.nec import NECAlgorithm

    torch.manual_seed(0)
    np.random.seed(0)

    alg = NECAlgorithm(
        device=torch.device("cpu"),
        embedding_network=NatureEmbedding,
        obs_key="obs",
        embedding_dim=64,
        dnd_capacity=200,
        k=2,
        kernel_delta=KERNEL_DELTA,
        dnd_lr=0.1,
        n_step=5,
        lr=1e-2,
        batch_size=4,
        init_random_frames=0,
        num_updates=1,
    )
    alg._obs_shape = (4, 84, 84)
    alg._num_actions = 2
    alg._buffer_device = torch.device("cpu")
    alg.embedding_net = NatureEmbedding((4, 84, 84), alg.embedding_dim)
    alg.dnd = DND(alg._num_actions, alg.dnd_capacity, alg.k, alg.kernel_delta,
                  alg._buffer_device)
    alg.optimizer = torch.optim.Adam(alg.embedding_net.parameters(), lr=alg.lr)
    alg.replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=100, device="cpu")
    )

    base = torch.zeros(4, 4, 84, 84)
    base[:, :, 40:44, 10:12] = 1.0
    obs = base.clone()
    obs[0, :, 20:22, 50:52] = 1.0
    obs[1, :, 60:62, 50:52] = 1.0
    obs[2, :, 20:22, 50:52] = 1.0
    obs[3, :, 60:62, 50:52] = 1.0
    actions = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([1.0, -1.0, -1.0, 1.0])

    alg.replay_buffer.extend(TensorDict(
        {"obs": obs, "action": actions, "n_step_return": targets},
        batch_size=[4],
    ))
    # Seed the DND so the sparsity guard (`_sizes[a] <= k`) doesn't skip the
    # loss computation for either action. k=2 requires > 2 stored entries, so
    # duplicate the two points per action (3 rows > k).
    with torch.no_grad():
        h0 = alg.embedding_net(obs)
        h0 = nn.functional.normalize(h0, dim=-1)
    alg.dnd.write_batch(0, h0[[0, 1, 0]], targets[[0, 1, 0]], dnd_lr=1.0)
    alg.dnd.write_batch(1, h0[[2, 3, 2]], targets[[2, 3, 2]], dnd_lr=1.0)

    captured = {}
    orig_knn_action = alg.dnd.knn_action

    def _spy(queries, action, k):
        captured.setdefault("h", []).append(queries.clone())
        return orig_knn_action(queries, action, k)

    alg.dnd.knn_action = _spy  # type: ignore[method-assign]

    alg._gradient_step()

    all_h = torch.cat(captured["h"], dim=0)
    norms = all_h.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Strongest guard: statistical check on real Pong frames (not a contrived
# pair). Confirmed empirically before the fix: with a freshly-initialised
# (untrained) NatureEmbedding, 100% of pairwise raw-embedding squared
# distances among 100 real, randomly-sampled Pong frames fell below
# kernel_delta=1e-3 (mean ~5e-5, ~20x smaller than delta) -- i.e. the kernel
# could not discriminate ANY pair of real game states, regardless of how
# different they were. After normalising to the unit sphere, only ~16% of
# pairs remained below kernel_delta.
# ---------------------------------------------------------------------------

def test_real_pong_frames_need_normalisation_for_kernel_to_discriminate():
    """On real (not synthetic) Pong frames, raw NatureEmbedding output is so
    tightly clustered that the DND kernel can't tell states apart at all;
    unit-norm normalisation fixes this for the large majority of pairs.
    """
    pytest = __import__("pytest")
    pytest.importorskip("ale_py")
    import gymnasium as gym
    import torch.nn.functional as F

    torch.manual_seed(0)
    env = gym.make("ALE/Pong-v5", frameskip=4)
    obs, _ = env.reset(seed=0)
    raw_frames = []
    for _ in range(120):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        raw_frames.append(obs)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    import numpy as np
    arr = torch.from_numpy(np.stack(raw_frames)).float() / 255.0  # (N,210,160,3)
    gray = arr.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)     # (N,1,210,160)
    resized = F.interpolate(gray, size=(84, 84), mode="bilinear", align_corners=False)
    obs_t = resized.repeat(1, 4, 1, 1)                            # (N,4,84,84)

    net = NatureEmbedding((4, 84, 84), 64)
    with torch.no_grad():
        h_raw = net(obs_t)
    h_norm = F.normalize(h_raw, dim=-1)

    n = h_raw.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    d_raw = (torch.cdist(h_raw, h_raw) ** 2)[mask]
    d_norm = (torch.cdist(h_norm, h_norm) ** 2)[mask]

    frac_collapsed_raw = (d_raw < KERNEL_DELTA).float().mean().item()
    frac_collapsed_norm = (d_norm < KERNEL_DELTA).float().mean().item()

    # Before the fix this was ~1.0 (every pair collapsed). Guard against
    # regressing back to that: normalisation must keep the collapsed
    # fraction well below "almost everything".
    assert frac_collapsed_norm < 0.5, (
        f"{frac_collapsed_norm:.1%} of normalised-embedding pairs still fall "
        f"below kernel_delta={KERNEL_DELTA} -- the DND kernel is still "
        "largely delta-dominated on real Pong frames, which reproduces the "
        "'reward stuck at the minimum' bug."
    )
