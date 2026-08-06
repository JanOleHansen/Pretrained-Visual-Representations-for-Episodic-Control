"""Model-Free Episodic Control (MFEC) algorithm implementation.

MFEC is a memory-based reinforcement learning algorithm introduced by
Blundell et al. (2016).  Unlike DQN, which trains a neural network to
*approximate* Q-values, MFEC stores actual experiences directly in memory
and retrieves them at decision time using a k-nearest-neighbour (kNN) lookup.

Key ideas
---------
* **Episodic memory (QEC)**: one table per discrete action, each mapping a
  compact state embedding to the best discounted return ever seen from that
  (state, action) pair.
* **Fixed random encoder**: raw pixel frames are reduced to a low-dimensional
  vector by a fixed, random projection matrix — no gradient updates needed.
* **Optimistic initialisation**: if a (state, action) pair has never been
  seen, the Q-estimate is +∞ so the agent is nudged to explore it first.
* **Max-aggregation**: if a state is revisited, we keep the *highest* return
  ever observed, which prevents a bad episode from erasing earlier knowledge.

Algorithm pseudocode (matches the paper)
-----------------------------------------
Input:
    encoder φ           # fixed embedding function (random projection)
    k                   # number of nearest neighbours
    γ                   # discount factor
    ε                   # exploration probability
    capacity            # max entries per action buffer

Initialise:
    for each action a ∈ A:
        Q_EC[a] ← empty buffer of (key, value) pairs

repeat for each episode:
    observe initial observation o_0
    trajectory ← empty list

    for t = 0, 1, 2, … until episode ends:
        s_t ← φ(o_t)                          # encode observation

        if random() < ε:
            a_t ← random action from A
        else:
            for each action a ∈ A:
                Q̂(s_t, a) ← EstimateQ(s_t, a)
            a_t ← argmax_a Q̂(s_t, a)

        execute a_t, observe r_{t+1} and next observation o_{t+1}
        append (s_t, a_t, r_{t+1}) to trajectory

    # Backward Monte Carlo return computation
    G ← 0
    for t = T-1 down to 0:
        G ← r_{t+1} + γ · G                   # discounted return from step t
        (s_t, a_t, _) ← trajectory[t]
        UpdateMemory(s_t, a_t, G)

function EstimateQ(s, a):
    if s is exactly in Q_EC[a]:
        return Q_EC[a][s]                      # exact hit: return stored value
    elif |Q_EC[a]| < k:
        return +∞                              # too few entries → optimistic
    else:
        neighbours ← k nearest keys to s in Q_EC[a]
        return (1/k) · Σ Q_EC[a][s_i] for s_i in neighbours

function UpdateMemory(s, a, G):
    if s is in Q_EC[a]:
        Q_EC[a][s] ← max(Q_EC[a][s], G)       # keep best return seen
        touch s                                # refresh recency (see below)
    else:
        if |Q_EC[a]| ≥ capacity:
            evict least-recently-updated entry
        insert (s, G) into Q_EC[a]

Eviction policy
---------------
Blundell et al. (2016) §2: "we limit the size of the table by removing the
least recently *updated* entry once a maximum size has been reached."  Note
"updated", not "inserted" and not "read": an Eq. (1) max-update refreshes an
entry's recency, a kNN read does not.  This matters a lot on Atari — a plain
FIFO ring buffer evicts the oldest *insertions*, which are exactly the
early-level states the agent re-visits on every single episode.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Callable

import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueActor

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.encoders.factory import make_encoder


#
# Main algorithm class
#

class MFECAlgorithm(BaseAlgorithm):
    """Model-Free Episodic Control.

    Implements the BaseAlgorithm interface expected by StepTrainer.
    The trainer calls setup() once, then step() repeatedly with collected
    batches.  This algorithm performs no gradient updates; the only
    "learning" is inserting experiences into the QEC memory tables.
    """

    def __init__(
            self,
            device: torch.device | None = None,
            *,
            encoder_name: str = "random_projection",
            vae_checkpoint: str | None = None,
            seed: int | None = None,
            obs_key: str = "pixels",
            buffer_size: int = 1_000_000,
            k: int = 11,
            state_dim: int = 64,
            gamma: float = 0.99,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            annealing_frames: int = 1_000_000,
            # Exploration rate used by get_policy() (evaluation).  Blundell et
            # al. (2016) §4.1 report the score of the eps=0.005 policy — they
            # never evaluate a pure argmax.  Set 0.0 for deterministic eval.
            eval_eps: float = 0.005,
            frames_per_batch: int = 1_000,
            max_frames_per_traj: int = -1,
            # Quantisation precision for the exact-match hash key.
            # Each embedding coordinate is multiplied by this factor and rounded
            # to the nearest integer before being converted to bytes.  Higher
            # values preserve more precision (fewer false-negatives) but make
            # keys longer.  1e5 gives five decimal places of float32 stability.
            key_scale: float = 1e5,
            # dino v2
            dinov2_weights: str | None = None,
            dinov2_model_name: str = "dinov2_vitb14",
            dinov2_repo_dir: str | None = None,
            dinov2_image_size: int = 224,
            # resnet
            resnet_weights_path: str | None = None,
            resnet_model_name: str = "resnet18",
            resnet_image_size: int = 224,

    ) -> None:

        super().__init__(device)
        self.encoder_name = encoder_name
        self.vae_checkpoint = vae_checkpoint
        self.seed = seed
        self.obs_key = obs_key
        self.buffer_size = buffer_size
        self.k = k
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.annealing_frames = annealing_frames
        self.eval_eps = eval_eps
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self.key_scale = key_scale
        #dinov2
        self.dinov2_weights = dinov2_weights
        self.dinov2_model_name = dinov2_model_name
        self.dinov2_repo_dir = dinov2_repo_dir
        self.dinov2_image_size = dinov2_image_size
        #resnet
        self.resnet_weights_path = resnet_weights_path
        self.resnet_model_name = resnet_model_name
        self.resnet_image_size = resnet_image_size

        self._collected_frames = 0

    #
    # Setup
    #

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)

        self._buffer_device = (
            self.device if self.device is not None else torch.device("cpu")
        )

        sample_shape = obs_shape[-3:]
        obs_flat_dim = int(np.prod(sample_shape))
        in_channels = sample_shape[0]

        self.encoder = make_encoder(
            self.encoder_name,
            obs_flat_dim=obs_flat_dim,
            in_channels=in_channels,
            state_dim=self.state_dim,
            vae_checkpoint_path=self.vae_checkpoint,
            device=self._buffer_device,
            seed=self.seed,
            #dinov2
            dinov2_weights_path=self.dinov2_weights,
            dinov2_model_name=self.dinov2_model_name,
            dinov2_repo_dir=self.dinov2_repo_dir,
            dinov2_image_size=self.dinov2_image_size,
            #resnet
            resnet_weights_path=self.resnet_weights_path,
            resnet_model_name=self.resnet_model_name,
            resnet_image_size=self.resnet_image_size
        )

        # Detect number of parallel envs from the proof env's batch_size.
        # ParallelEnv(E, fn).batch_size = torch.Size([E]); single env → Size([]).
        env_bs = proof_env.batch_size
        self._num_envs: int = int(env_bs[0]) if len(env_bs) == 1 else 1
        # Per-env carry buffers for partial episodes that span batch boundaries.
        self._carry: list[dict | None] = [None] * self._num_envs

        self.qec = QEC(
            num_actions, self.buffer_size, self.k, self._buffer_device,
            key_scale=self.key_scale,
        )

        self._num_actions = num_actions

        self.qec_policy = QECPolicy(self.qec, self.encoder, num_actions)

        # Policy chain: pixels -> state_embedding -> Q(s,·) -> action.
        #
        # Splitting the embedding into its own TensorDictModule means the
        # collector writes `state_embedding` into every collected transition,
        # so step() reads it back instead of embedding the same frames a
        # second time.  Safe because MFEC's encoder is frozen (see
        # src/encoders/base.py): the cached vector is bit-identical to a
        # recomputation.  This would NOT be valid for NEC, whose CNN changes
        # between collection and step().
        self._embed_module = TensorDictModule(
            _EmbedModule(self.encoder),
            in_keys=[self.obs_key],
            out_keys=["state_embedding"],
        )

        self.q_actor = QValueActor(
            module=self.qec_policy,
            spec=action_spec,
            in_keys=["state_embedding"],
        )

        # Both e-greedy modules need an *unbatched* action spec.  proof_env is
        # the TRAINING env, so with num_envs=4 its action_spec carries that
        # batch dim (shape [4]) — but BaseTrainer.evaluate() builds a single
        # env via make_env(num_envs=1), which factory.make_env returns bare
        # (no ParallelEnv), so the eval action is a scalar of shape [].
        # EGreedyModule.forward only auto-expands a spec when it is unbatched
        # (`not len(spec.shape)`); a [4] spec meeting a [] action raises
        # "Action spec shape does not match the action shape" at the first
        # evaluate().  An unbatched spec is correct for both callers: it
        # expands to [4] under the collector and compares equal at eval.
        action_spec_unbatched = action_spec
        for _ in range(len(env_bs)):
            action_spec_unbatched = action_spec_unbatched[0]

        # `device=` is NOT optional here, and passing `spec` does not cover it:
        # EGreedyModule builds its `eps` buffer from the `device` kwarg alone
        # (torchrl exploration.py) and `forward` raises if `action.device` and
        # `eps.device` disagree.  Omitting it left eps on CPU for both modules.
        # Training hid the bug — Collector(device=...) calls `.to()` on the
        # explore policy and moved `greedy_module.eps` as a side effect — but
        # `_policy` never reaches the collector, so `eval_greedy_module` stayed
        # on CPU and the first evaluate() on GPU died on a cuda action.
        self.greedy_module = EGreedyModule(
            spec=action_spec_unbatched,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=self._buffer_device,
        )

        # Constant eps for evaluation — NOT annealed, never `.step()`ed.
        # eval_eps=0.0 makes this a no-op (`rand() < 0` is never true), which
        # restores the old pure-argmax behaviour.
        self.eval_greedy_module = _EvalEGreedyModule(
            spec=action_spec_unbatched,
            eps_init=self.eval_eps,
            eps_end=self.eval_eps,
            annealing_num_steps=1,
            device=self._buffer_device,
        )

        # Eval (get_policy) needs the embedding step too, so all policies are
        # built on the same chain.
        self._policy = _SharedPolicy(
            self._embed_module, self.q_actor, self.eval_greedy_module
        )
        self._explore_policy = _SharedPolicy(
            self._embed_module, self.q_actor, self.greedy_module
        )

    #
    #   Embedding helper
    #

    #: Observation bytes per encoder forward pass.  Chunking exists to bound
    #: peak memory when embedding a whole collector batch of pixel frames.
    #: It must NOT be sized by ``num_envs`` (as it was until this was caught in
    #: review): with frames_per_batch=1024 and num_envs=4 that is 256 forward
    #: passes of batch size 4, which is free for a random-projection matmul but
    #: leaves a GPU almost idle for a ViT and pays 256x the launch overhead.
    #: 64 MB of raw observation is ~166 frames of (3, 210, 160) float32 or
    #: ~2400 frames of (1, 84, 84).
    _EMBED_CHUNK_BYTES = 64 * 1024 * 1024

    def _embed_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Embed ``(..., C, H, W)`` observations in memory-bounded chunks."""
        obs_flat = obs.reshape(-1, *obs.shape[-3:])
        n = obs_flat.shape[0]
        if n == 0:
            return obs_flat.new_zeros((0, self.state_dim))

        per_obs = obs_flat[0].numel() * obs_flat.element_size()
        chunk = max(1, self._EMBED_CHUNK_BYTES // max(1, per_obs))
        chunk = min(chunk, n)

        if chunk >= n:
            return self.qec_policy.embed(obs_flat)
        return torch.cat(
            [self.qec_policy.embed(obs_flat[i: i + chunk])
             for i in range(0, n, chunk)],
            dim=0,
        )

    #
    #   Training step
    #

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one batch of collected transitions and update QEC memory.

        MFEC has no neural-network parameters to optimise.  The update is:
          1. Compute discounted Monte Carlo returns per env (Bug 1 fix).
          2. Buffer partial episodes across batch boundaries (Bug 2 fix).
          3. Exact-match states (hash lookup, O(1)) → max-aggregate the stored value.
          4. Novel states → insert into the ring buffer via add_batch().

        The collector batch from SyncDataCollector with ParallelEnv(E, fn) has
        batch_size = (E, T).  Computing returns on the flat (E*T,) representation
        mixes trajectories from different envs, corrupting the learning signal.
        This method always operates on the (E, T) structure before flattening.

        Parameters
        ----------
        batch : TensorDict
            Transitions from SyncDataCollector.  Shape (E, T) or (T,) for E=1.

        Returns
        -------
        dict
            Scalar metrics including train/exact_hit_rate.
        """

        # --- 1. Determine per-env shape -----------------------------------
        # Collector yields (E, T) with ParallelEnv(E, fn); (T,) with single env.
        bs = batch.batch_size
        if len(bs) == 2:
            E, T = int(bs[0]), int(bs[1])
        elif len(bs) == 1:
            E, T = 1, int(bs[0])
        else:
            raise ValueError(f"Unexpected batch shape: {bs}")
        n = E * T

        self.greedy_module.step(n)
        self._collected_frames += n

        dev = self._buffer_device

        # --- 2. Obtain one embedding per collected frame -------------------
        # Preferred path: reuse what the policy already computed during
        # collection (see _EmbedModule in setup()).  This is exact, not an
        # approximation: MFEC's encoder is frozen by contract
        # (src/encoders/base.py — "identical pixels -> identical embedding"),
        # so re-embedding here would reproduce the same numbers at double the
        # cost.  With a random projection that waste is invisible; with a
        # DINOv2 ViT it is half the total runtime.
        states_all = batch.get("state_embedding", default=None)
        if states_all is not None:
            states_all = states_all.reshape(n, -1).to(dev)
        else:
            # Fallback: no policy-written embedding in this batch. Reachable
            # only if the collector ran a random policy (init_random_frames>0),
            # which get_collector_config() pins to 0 for MFEC.
            states_all = self._embed_observations(batch[self.obs_key]).to(dev)

        states_2d = states_all.reshape(E, T, -1)                  # (E, T, d)

        # Flatten to 1D then reshape to (E, T) — handles the (E,T,1) reward shape.
        rewards_2d = (batch["next", "reward"].cpu().numpy()
                      .flatten().astype(np.float64).reshape(E, T))
        dones_2d   = (batch["next", "done"].cpu().numpy()
                      .flatten().astype(bool).reshape(E, T))
        actions_2d = batch["action"].to(dev).reshape(n).long().reshape(E, T)

        # --- 3. Per-env discounted return computation with carry-over ------
        #
        # Only complete episodes (steps up to and including done=True) contribute
        # to QEC updates. The trailing partial trajectory of each env is buffered
        # and prepended to that env's data in the next step() call so returns are
        # always computed over full episodes (Algorithm 1, Blundell et al. 2016).
        #
        # Guard: resize carry if num_envs changed (e.g. checkpoint with different E).
        if len(self._carry) != E:
            self._carry = [None] * E

        collect_states:  list[torch.Tensor] = []
        collect_G:       list[torch.Tensor] = []
        collect_actions: list[torch.Tensor] = []

        for env_idx in range(E):
            states_e  = states_2d[env_idx]         # (T, d)
            rewards_e = rewards_2d[env_idx]         # (T,)
            dones_e   = dones_2d[env_idx]           # (T,) bool
            actions_e = actions_2d[env_idx]         # (T,) long

            # Prepend buffered partial episode from the previous batch.
            carry = self._carry[env_idx]
            if carry is not None:
                states_e  = torch.cat([carry["states"],  states_e],  dim=0)
                rewards_e = np.concatenate([carry["rewards"], rewards_e])
                dones_e   = np.concatenate([carry["dones"],   dones_e])
                actions_e = torch.cat([carry["actions"], actions_e], dim=0)

            ends = np.flatnonzero(dones_e)

            if len(ends) == 0:
                # No complete episode in this env — buffer everything.
                self._carry[env_idx] = {
                    "states":  states_e,
                    "rewards": rewards_e,
                    "dones":   dones_e,
                    "actions": actions_e,
                }
                continue

            last = int(ends[-1])  # index of last done (inclusive end of last episode)

            # Discounted return for the complete portion [0 .. last].
            G_e      = np.empty(last + 1, dtype=np.float64)
            ep_start = 0
            for ep_end in ends:
                ep_end = int(ep_end)
                r = rewards_e[ep_start : ep_end + 1]
                G_e[ep_start : ep_end + 1] = lfilter(
                    [1.0], [1.0, -self.gamma], r[::-1]
                )[::-1]
                ep_start = ep_end + 1

            collect_states.append(states_e[:last + 1])
            collect_G.append(torch.tensor(G_e, dtype=torch.float64, device=dev))
            collect_actions.append(actions_e[:last + 1])

            # Buffer the trailing partial trajectory for the next batch.
            if last < len(states_e) - 1:
                self._carry[env_idx] = {
                    "states":  states_e[last + 1:],
                    "rewards": rewards_e[last + 1:],
                    "dones":   dones_e[last + 1:],
                    "actions": actions_e[last + 1:],
                }
            else:
                self._carry[env_idx] = None

        # --- 4. Early-exit if no complete episode this batch ---------------
        #
        # train/exact_hit_rate is deliberately *absent* here rather than 0.0.
        # A Ms. Pac-Man episode is ~600 decisions and frames_per_batch is 1024
        # over 4 envs, so most batches close no episode and make no QEC
        # queries; reporting 0.0 for those made the logged series oscillate
        # between 0 and the real rate and read like a broken memory.  Omitting
        # the key leaves a gap, matching _IntervalStats' convention in
        # StepTrainer ("a genuine gap instead of a fabricated point").
        if not collect_states:
            return {
                "train/epsilon":        float(self.greedy_module.eps),
                "train/qec_size":       float(np.mean(self.qec._sizes)),
            }

        states_gpu  = torch.cat(collect_states,  dim=0)    # (N_done, d)
        G_gpu       = torch.cat(collect_G,       dim=0)    # (N_done,)
        actions_gpu = torch.cat(collect_actions, dim=0).flatten().long()  # (N_done,)

        # --- 5. GPU-native per-action update (argsort + bincount) ----------
        sorted_idx     = torch.argsort(actions_gpu, stable=True)
        sorted_states  = states_gpu[sorted_idx]
        sorted_values  = G_gpu[sorted_idx]
        sorted_actions = actions_gpu[sorted_idx]

        counts  = torch.bincount(sorted_actions, minlength=self._num_actions)
        offsets = torch.zeros(self._num_actions + 1, dtype=torch.long, device=dev)
        offsets[1:] = counts.cumsum(0)
        # One CPU sync to read A+1 segment boundary integers.
        offsets_cpu = offsets.cpu().tolist()

        total_queries = 0
        total_hits    = 0

        for a in range(self._num_actions):
            seg_s, seg_e = offsets_cpu[a], offsets_cpu[a + 1]
            if seg_s == seg_e:
                continue

            act_states = sorted_states[seg_s:seg_e]   # zero-copy view
            act_values = sorted_values[seg_s:seg_e]
            m = seg_e - seg_s
            total_queries += m

            # --- Exact-match path: O(m) hash lookup, no kNN needed --------
            keys_a = self.qec._make_keys(act_states)
            k_to_s = self.qec._key_to_slot[a]

            total_hits += sum(1 for key in keys_a if key in k_to_s)

            act_values_list = act_values.tolist()
            best_row: dict[bytes, int] = {}
            for i, key in enumerate(keys_a):
                j = best_row.get(key)
                if j is None or act_values_list[i] > act_values_list[j]:
                    best_row[key] = i

            hit_rows: list[int] = []
            hit_slots: list[int] = []
            hit_keys: list[bytes] = []
            novel_rows: list[int] = []

            for key, i in best_row.items():
                slot = k_to_s.get(key)
                if slot is not None:
                    hit_rows.append(i)
                    hit_slots.append(slot)
                    hit_keys.append(key)
                else:
                    novel_rows.append(i)


            if hit_rows:
                rows_t = torch.tensor(hit_rows, dtype=torch.long, device=dev)
                slots_t = torch.tensor(hit_slots, dtype=torch.long, device=dev)

                self.qec.values[a, slots_t] = torch.maximum(self.qec.values[a, slots_t], act_values[rows_t])

                # Eq. (1) max-update counts as an update, so refresh recency —
                # the paper evicts the least recently *updated* entry (§2).
                self.qec.touch(a, hit_keys)

            # --- Novel states → insert, evicting the LRU entry if full ----

            if novel_rows:
                rows_t = torch.tensor(novel_rows, dtype=torch.long, device=dev)
                self.qec.add_batch(a, act_states[rows_t], act_values[rows_t])

            
        hit_rate = total_hits / total_queries if total_queries > 0 else 0.0

        return {
            "train/epsilon":        float(self.greedy_module.eps),
            "train/qec_size":       float(np.mean(self.qec._sizes)),
            "train/exact_hit_rate": hit_rate,
        }

    #
    #   Policy access
    #

    def get_policy(self) -> TensorDictModule:
        # Must be the full chain: q_actor alone reads "state_embedding", which
        # only _embed_module writes.
        #
        # Ends in _EvalEGreedyModule (constant eval_eps, paper §4.1), NOT a bare
        # argmax — see that class for why a deterministic eval policy makes
        # eval/return_std identically 0 and understates the agent.
        return self._policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    #
    #   Evaluation diagnostics
    #

    def reset_eval_metrics(self) -> None:
        self.qec.reset_lookup_stats()

    def eval_metrics(self) -> dict[str, float]:
        """How often the *evaluation* policy actually found the state in QEC.

        MFEC's score is only as good as its recall: if the eval rollout never
        resolves a query to a stored (state, action) pair, every Q estimate is
        a k-neighbour mean, all actions of a state score alike, and the agent
        plays at random however good the memory is.  That failure is invisible
        in ``eval/return_mean`` alone — it just looks like a bad policy — so
        the hit rate is logged next to it.  A rate near 0 means train-time and
        eval-time observations are not the same bytes.

        Note the denominator differs from ``train/exact_hit_rate``, so the two
        are not comparable point for point.  Here it is every *(state, action)*
        pair the policy asks about — all ``|A|`` of them per frame — whereas
        ``step()`` only queries the single action actually taken.  Blundell et
        al. (2016) §4.1's "about ... 50% for Ms. PAC-MAN" is the latter.
        """
        queries, exact, near = self.qec.lookup_stats()
        if queries == 0:
            return {}
        return {
            "eval/exact_hit_rate":  exact / queries,
            "eval/memory_hit_rate": (exact + near) / queries,
        }

    #
    #   Checkpointing
    #

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict={},
            optimizer_state_dict={},
            extra={
                "encoder_state":    self.encoder.state(),
                "qec_state":        self.qec.__getstate__(),
                "collected_frames": self._collected_frames,
                "carry":            self._serialise_carry(),
                # EGreedyModule keeps the live epsilon in its own buffer, and
                # MFEC has no policy_state_dict to carry it.  Without this a
                # resumed run silently restarts exploration at eps_init and
                # re-anneals from scratch, even though _collected_frames says
                # the anneal finished long ago.
                "greedy_state":     self.greedy_module.state_dict(),
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.encoder.load_state(state.extra["encoder_state"])
        self._collected_frames = int(state.extra["collected_frames"])

        # QEC's device is pickled inside qec_state and torch.load's
        # map_location does not rewrite it; re-pin to the device this run owns.
        qec_state = dict(state.extra["qec_state"])
        qec_state["device"] = self._buffer_device
        self.qec.__setstate__(qec_state)

        greedy_state = state.extra.get("greedy_state")
        if greedy_state is not None:
            self.greedy_module.load_state_dict(greedy_state)

        if "carry" in state.extra:
            self._deserialise_carry(state.extra["carry"])
        else:
            self._carry = [None] * self._num_envs

    def _serialise_carry(self) -> list:
        """Convert per-env carry buffers to numpy for torch.save serialisation."""
        out = []
        for c in self._carry:
            if c is None:
                out.append(None)
            else:
                out.append({
                    "states":  c["states"].cpu().numpy(),
                    "rewards": c["rewards"],
                    "dones":   c["dones"],
                    "actions": c["actions"].cpu().numpy().astype(np.int64),
                })
        return out

    def _deserialise_carry(self, data: list) -> None:
        """Restore per-env carry buffers from serialised numpy arrays."""
        dev = self._buffer_device
        self._carry = []
        for c in data:
            if c is None:
                self._carry.append(None)
            else:
                self._carry.append({
                    "states":  torch.from_numpy(c["states"]).to(dev),
                    "rewards": c["rewards"],
                    "dones":   c["dones"],
                    "actions": torch.from_numpy(c["actions"]).long().to(dev),
                })


# ---------------------------------------------------------------------------
# Policy module
# ---------------------------------------------------------------------------

class _EmbedModule(nn.Module):
    """φ as its own module: ``(..., C, H, W)`` pixels -> ``(..., d)`` embedding.

    Exists so the collector persists the embedding under ``state_embedding``
    and ``MFECAlgorithm.step()`` never has to recompute it.  Returns ``self``
    on deepcopy so the collector shares the one frozen encoder.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def forward(self, obs):
        leading = obs.shape[:-3]
        states = self.encoder.embed(obs)          # (prod(leading) or 1, d)
        if leading:
            return states.reshape(*leading, states.shape[-1])
        return states.squeeze(0)


class QECPolicy(nn.Module):
    """Q(s, ·) by QEC lookup.

    Takes a **state embedding**, not raw pixels — ``_EmbedModule`` runs φ one
    step earlier in the policy chain so the result can be reused by ``step()``.
    ``encoder`` is still held (and still shared on deepcopy) so ``embed()``
    remains available as a convenience for tests and offline analysis.

    Optimistic initialisation
    -------------------------
    ``QEC.estimate_all`` returns ``+inf`` for any action whose buffer holds
    ``<= k`` entries, so untried actions are always preferred (Blundell et al.
    2016 §2).  ``forward`` maps those sentinels onto a large finite value —
    ``QValueActor`` cannot argmax over ``inf`` — **plus independent uniform
    jitter per (state, action)**.

    The jitter is load-bearing, not cosmetic.  Mapping every ``+inf`` onto a
    single constant makes all untried actions exact ties, and ``argmax``
    resolves ties by *lowest index*: with an empty QEC the policy emitted
    action 0 for essentially every state, then action 1 once action 0's buffer
    passed ``k``, and so on.  The agent played one fixed action per episode,
    cycling 0..A-1, and seeded the QEC with A degenerate single-action
    trajectories whose max-returns then persist forever (Eq. 1 never
    decreases).  Jitter makes the argmax uniform over the untried set instead.

    Finite estimates are never perturbed, so a QEC that has real information
    about a state is exactly as deterministic as before — which is the
    guarantee that matters for evaluation.  The jitter draws from the global
    torch RNG, so seeded runs stay reproducible.
    """

    #: Finite stand-in for the ``+inf`` optimistic estimate.  Must exceed any
    #: achievable Monte-Carlo return so an untried action always wins the
    #: argmax against a real Q-value (unclipped Ms. Pac-Man tops out ~3e4).
    OPTIMISTIC_VALUE = 1e9

    #: Width of the uniform jitter added to each optimistic entry.
    #:
    #: This cannot be made small.  ``q_values`` is float32, where the ULP at
    #: 1e9 is 64.0 — jitter drawn from ``uniform(0, 1)`` would round straight
    #: back to exactly 1e9 and restore the very tie it is meant to break.
    #: 1e6 leaves ~15k distinct representable values per entry while keeping
    #: every optimistic entry (<= 1.001e9) far above any real return, and far
    #: above the sentinel threshold the trainer uses to keep these values out
    #: of ``train/q_values``.
    OPTIMISTIC_JITTER = 1e6

    def __init__(self, qec, encoder, num_actions):
        super().__init__()
        self.qec = qec
        self.encoder = encoder
        self.num_actions = num_actions

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy = cls.__new__(cls)
        memo[id(self)] = copy
        super(QECPolicy, copy).__init__()
        copy.qec = self.qec            # share memory by reference
        copy.encoder = self.encoder    # share encoder by reference
        copy.num_actions = self.num_actions
        return copy

    def embed(self, obs):
        return self.encoder.embed(obs)

    def forward(self, states):
        leading = states.shape[:-1]
        flat = states.reshape(-1, states.shape[-1])
        q_values = self.qec.estimate_all(flat)

        # Optimistic entries -> large finite value + per-entry jitter, so the
        # downstream argmax picks uniformly among untried actions rather than
        # always returning the lowest index.  See the class docstring.
        optimistic = torch.isposinf(q_values)
        if optimistic.any():
            jitter = torch.rand_like(q_values) * self.OPTIMISTIC_JITTER
            q_values = torch.where(
                optimistic,
                self.OPTIMISTIC_VALUE + jitter,
                q_values,          # finite estimates pass through untouched
            )

        if leading:
            return q_values.reshape(*leading, self.num_actions)
        return q_values.squeeze(0) if q_values.shape[0] == 1 else q_values


class _EvalEGreedyModule(EGreedyModule):
    """ε-greedy that applies ε regardless of the ambient ``ExplorationType``.

    ``BaseTrainer.evaluate()`` wraps its rollout in
    ``set_exploration_type(ExplorationType.MODE)``, and torchrl's
    ``EGreedyModule.forward`` is gated on
    ``exploration_type() in (ExplorationType.RANDOM, None)`` — so a stock
    ``EGreedyModule`` placed in the eval chain is silently a **no-op**.  Simply
    adding one to ``_policy`` therefore does not make evaluation stochastic;
    verified by measurement (see below).

    Why MFEC needs ε at evaluation at all
    -------------------------------------
    Blundell et al. (2016) §4.1 report the score of the ε = 0.005 policy; the
    paper never evaluates a pure argmax.  That is not incidental.  With
    ``repeat_action_probability=0.0`` (which MFEC requires — footnote 1) ALE is
    deterministic, and Ms. Pac-Man's opening is insensitive to ``NoopResetEnv``,
    so a deterministic policy replays **the same trajectory every episode**:
    ``eval/return_std`` is identically 0 and ``num_eval_episodes`` silently
    collapses to a single sample at N times the cost.

    Worse, the pure-argmax policy is not the policy being trained.  QEC values
    are max-over-returns and therefore optimistic — Eq. (1) never decreases —
    so exploiting them without any ε lands the agent in states whose stored
    value came from one lucky ε-greedy trajectory it can no longer reproduce.
    Measured on Ms. Pac-Man with an identical QEC:

        get_policy()          / MODE     mean=380.0  std=0.000   (5/5 identical)
        get_explore_policy()  / RANDOM   mean=448.0  std=248.4
        get_explore_policy()  / MODE     mean=380.0  std=0.000   <- eps ignored

    The third row is why this subclass exists.
    """

    def forward(self, tensordict):
        with set_exploration_type(ExplorationType.RANDOM):
            return super().forward(tensordict)


class _SharedPolicy(TensorDictSequential):
    #Returns self on deepcopy so a single EGreedyModule is shared with the collector.

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# ---------------------------------------------------------------------------
# Fused episodic memory — all actions in one GPU tensor pair
# ---------------------------------------------------------------------------

class QEC:
    """Q-value Episodic Controller — fused GPU tensors + per-action hash maps.

    GPU tensors (states, values) hold all action buffers in a single allocation:

        states : (num_actions, capacity, state_dim)  float32
        values : (num_actions, capacity)             float64

    One ``OrderedDict`` per action provides O(1) exact-match lookup,
    implementing the paper's Eq. (2) "case 1" without any distance computation:

        _key_to_slot[a] : bytes → int   (quantised embedding → slot index)

    The dict is kept in **least-recently-updated-first** order, so it doubles
    as the LRU queue the paper's eviction rule needs (§2): ``popitem(last=False)``
    yields the entry to discard, and ``move_to_end(key)`` refreshes an entry
    whose value was just max-updated via Eq. (1).  Both are O(1).

    estimate_all() checks the dict first and only falls back to kNN (cdist)
    for novel queries; a *read* deliberately does not refresh recency, since
    the paper evicts the least recently **updated** entry.

    Parameters
    ----------
    num_actions : int
    capacity    : int — max entries per action
    k           : int — number of nearest neighbours for novel states
    device      : torch.device
    key_scale   : float — multiply embeddings by this before rounding to int32
                  for the hash key.  Higher → finer quantisation; must keep
                  values within int32 range (~2.1 × 10⁹).  Default 1e5 gives
                  five decimal places of float32 precision.
    """

    # Maximum intermediate (queries × stored) tensor in bytes before chunking.
    _CHUNK_BYTES = 256 * 1024 * 1024

    #: Relative tolerance for treating the nearest stored key as *the same
    #: state* (Eq. 2, case 1) rather than a neighbour to average over.
    #:
    #: It has to be relative: the quantity compared is an L2 distance in
    #: embedding space, and that space is chosen by the encoder.  A 64-d random
    #: projection of an 84×84 frame has ``‖φ(o)‖ ≈ 2``; a frozen DINOv2 or
    #: ResNet embedding is 10–50× that.  The absolute ``1e-5`` this replaces
    #: was effectively zero for every encoder.
    #:
    #: The value is squeezed between two measured quantities.  Over 2.25 M
    #: frame pairs from a random rollout on Ms. Pac-Man (random projection,
    #: ``state_dim=64``):
    #:
    #:   * closest pair of genuinely *different* frames:  1.1e-3
    #:   * embedding drift from 1e-6 of per-pixel float noise:  7.8e-6
    #:
    #: ``3e-5 · (1 + ‖q‖) ≈ 9e-5`` sits ~12x under the first and ~11x over the
    #: second.  Erring low is deliberate: with the eval env now built on the
    #: training env's device (see ``env_worker_device``) the drift this absorbs
    #: should be zero, so this is defence in depth — and merging two distinct
    #: states is a silently wrong Q value, while missing a rescue only costs
    #: one lookup.
    _NEAR_EXACT_RTOL = 3e-5

    def __init__(
        self,
        num_actions: int,
        capacity:    int,
        k:           int,
        device:      torch.device,
        key_scale:   float = 1e5,
    ) -> None:
        self.num_actions = num_actions
        self.capacity    = capacity
        self.k           = k
        self.device      = device
        self._key_scale  = key_scale

        self.states: torch.Tensor | None = None   # (A, C, d) — lazy init
        self.values = torch.empty(num_actions, capacity, dtype=torch.float64, device=device)

        # Per-action fill level.  Slots [0, _sizes[a]) are live.
        self._sizes = [0] * num_actions

        # Per-action exact-match hash map, doubling as the LRU queue:
        # iteration order is least-recently-updated first.
        self._key_to_slot: list[OrderedDict[bytes, int]] = [
            OrderedDict() for _ in range(num_actions)
        ]

        self.reset_lookup_stats()

    # ------------------------------------------------------------------
    # Read-path instrumentation
    # ------------------------------------------------------------------

    def reset_lookup_stats(self) -> None:
        """Zero the counters :meth:`lookup_stats` reports.

        The near-exact tally stays a device tensor: it is accumulated inside
        ``estimate_all``, which the collector calls once per env step, and
        reading a GPU bool-sum back per action would put ``|A|`` forced syncs
        on the policy's hot path.  It is converted once, in ``lookup_stats``.
        """
        self._lookup_queries = 0
        self._lookup_exact = 0
        self._lookup_near = torch.zeros((), dtype=torch.long, device=self.device)

    def lookup_stats(self) -> tuple[int, int, int]:
        """``(queries, exact_hits, near_exact_hits)`` since the last reset.

        One "query" is one *(state, action)* pair asked of
        :meth:`estimate_all`, i.e. ``|A|`` per frame the policy sees — every
        candidate action, not only the one taken.  A policy whose hit rate is
        0 is not exploiting episodic memory at all, however full the buffers
        are; it is answering every question with a k-neighbour mean.

        Calls that short-circuit on an empty memory are not counted.
        """
        return self._lookup_queries, self._lookup_exact, int(self._lookup_near)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_keys(self, states: torch.Tensor) -> list[bytes]:
        """Convert a batch of float32 embeddings to stable bytes hash keys.

        Each embedding row is multiplied by _key_scale and rounded to the
        nearest int32, then converted to raw bytes.  This gives a key that
        is identical for the same observation (the projection is deterministic)
        and robust to float noise up to 0.5 / _key_scale per coordinate.

        The paper relies on exact state re-encounters: same pixels → same key.
        The random projection is fixed, so identical pixels always yield the
        same float32 embedding, and hence the same key.

        Parameters
        ----------
        states : (B, d) float32 — on any device

        Returns
        -------
        list of B bytes objects (one per row)
        """
        q = torch.round(states * self._key_scale).to(torch.int32)
        q_cpu = q.cpu().contiguous()
        return [q_cpu[i].numpy().tobytes() for i in range(q_cpu.shape[0])]

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init_states(self, state_dim: int) -> None:
        if self.states is None:
            self.states = torch.empty(
                self.num_actions, self.capacity, state_dim,
                dtype=torch.float32, device=self.device,
            )

    # ------------------------------------------------------------------
    # Q-value estimation — called from QECPolicy.forward()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_all(self, queries: torch.Tensor) -> torch.Tensor:
        """Estimate Q(s, a) for all actions.

        Per-query, per-action strategy (Eq. 2 of Blundell et al. 2016):
          1. Dict lookup: O(1) — return stored value directly (no cdist).
          2. kNN fallback: chunked cdist for queries not in the dict.

        Parameters
        ----------
        queries : (B, d) float32 — any device

        Returns
        -------
        (B, A) float32 — on queries.device; +inf where memory is too sparse
        """
        B     = queries.shape[0]
        A     = self.num_actions
        dev_q = queries.device

        max_size = max(self._sizes)
        if self.states is None or max_size == 0:
            return torch.full((B, A), float("inf"), dtype=torch.float32, device=dev_q)

        if queries.device != self.device:
            queries = queries.to(self.device)

        # One GPU→CPU sync to compute all keys (B rows × d int32 values).
        keys = self._make_keys(queries)   # list[bytes], length B

        result = torch.full((A, B), float("inf"), dtype=torch.float32, device=self.device)

        for a in range(A):
            size_a = self._sizes[a]
            k_to_s = self._key_to_slot[a]

            # --- Exact-match path: O(B) dict lookups, zero GPU work -----
            hit_b: list[int]   = []
            hit_slots: list[int] = []
            miss_b: list[int]  = []
            for b, key in enumerate(keys):
                slot = k_to_s.get(key)
                if slot is not None:
                    hit_b.append(b)
                    hit_slots.append(slot)
                else:
                    miss_b.append(b)

            self._lookup_queries += B
            self._lookup_exact += len(hit_b)

            if hit_b:
                hit_b_t = torch.tensor(hit_b,    dtype=torch.long, device=self.device)
                hit_s_t = torch.tensor(hit_slots, dtype=torch.long, device=self.device)
                result[a, hit_b_t] = self.values[a, hit_s_t].float()

            if not miss_b:
                continue   # all queries hit — no kNN needed for this action

            # +inf for actions with too few stored entries (optimistic init)
            if size_a <= self.k:
                continue   # result[a, miss_b] stays +inf

            # --- kNN fallback for misses (chunked cdist) -----------------
            miss_b_t   = torch.tensor(miss_b, dtype=torch.long, device=self.device)
            miss_q     = queries[miss_b_t]     # (miss_count, d)
            k_fetch    = min(self.k + 1, size_a)
            # Distances are used only for the ranking knn_action already did;
            # see the near-exact note below for why they are not read here.
            _, idx     = self.knn_action(miss_q, a, k_fetch)  # (miss_count, k_fetch)

            # Near-exact in the kNN result: the nearest stored key is
            # numerically indistinguishable from the query, so this *is* Eq. 2
            # case 1 and must return the stored value, not a k-neighbour mean.
            #
            # This is the last line of defence for the hash key, which is
            # brittle by construction — `round(state * key_scale)` flips as
            # soon as a coordinate crosses a .5 boundary, so a drift of 1e-7
            # per pixel already changes ~40% of keys and 1e-6 changes all of
            # them.  Falling through to the mean is not graceful degradation:
            # it hands every action of a state near-identical Q values and the
            # argmax becomes noise.
            #
            # `dists` cannot be used for this test.  torch.cdist takes the
            # `x^2 + y^2 - 2xy` shortcut once either side exceeds 25 rows (it
            # always does here), and that cancels catastrophically at short
            # range: measured 4.2e-4 for a pair whose true distance is 2.8e-6.
            # Ranking survives the error, a near-zero threshold does not — the
            # absolute `< 1e-5` this replaces was below cdist's own noise floor
            # and so fired essentially at random.  Recomputing the single
            # winner's distance directly is exact and costs O(miss x d).
            top1_dist = torch.linalg.vector_norm(
                miss_q - self.states[a, idx[:, 0]], dim=-1
            )
            # Tolerance is relative to the query norm so it means the same
            # thing for a random projection and for a ViT embedding.
            near_exact = top1_dist <= self._NEAR_EXACT_RTOL * (
                1.0 + torch.linalg.vector_norm(miss_q, dim=-1)
            )
            self._lookup_near += near_exact.sum()
            k_use      = min(self.k, k_fetch)
            knn_vals   = self.values[a, idx[:, :k_use]].float()   # (miss_count, k_use)
            knn_avg    = knn_vals.mean(dim=-1)
            exact_vals = self.values[a, idx[:, 0]].float()
            result[a, miss_b_t] = torch.where(near_exact, exact_vals, knn_avg)

        return result.T.to(dev_q)   # (B, A)

    # ------------------------------------------------------------------
    # kNN for a single action — used by estimate_all() for misses
    # ------------------------------------------------------------------

    @torch.no_grad()
    def knn_action(
        self,
        queries: torch.Tensor,   # (m, d)
        action:  int,
        k:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked kNN search for one action (exact L2, GPU-resident).

        Chunks the stored vectors to stay within _CHUNK_BYTES so the
        intermediate distance matrix never triggers OOM.

        Returns
        -------
        dists   : (m, k_eff) L2 distances — on self.device
        indices : (m, k_eff) slot indices into self.states[action]
        """
        size  = self._sizes[action]
        k_eff = min(k, size)

        if queries.device != self.device:
            queries = queries.to(self.device)

        m = queries.shape[0]
        d = queries.shape[1]

        # An empty buffer (or query set) has no neighbours to return.  Without
        # this, chunk_size collapses to 0 below and the loop raises
        # `ValueError: range() arg 3 must not be zero`.  estimate_all() gates
        # on `size_a > self.k` first, but this is a public method.
        if k_eff <= 0 or m == 0:
            return (
                torch.full((m, 0), float("inf"), device=self.device),
                torch.zeros((m, 0), dtype=torch.long, device=self.device),
            )

        chunk_size = max(1, self._CHUNK_BYTES // (m * d * 4))
        chunk_size = min(chunk_size, size)

        best_dists = torch.full((m, k_eff), float("inf"), device=self.device)
        best_idx   = torch.zeros((m, k_eff), dtype=torch.long,  device=self.device)

        for cs in range(0, size, chunk_size):
            ce  = min(cs + chunk_size, size)
            cd  = torch.cdist(queries, self.states[action, cs:ce])
            ck  = min(k_eff, ce - cs)
            chd, chi = cd.topk(ck, dim=1, largest=False)
            chi = chi + cs

            merged_d = torch.cat([best_dists, chd], dim=1)
            merged_i = torch.cat([best_idx,   chi], dim=1)
            _, keep  = merged_d.topk(k_eff, dim=1, largest=False)
            best_dists = merged_d.gather(1, keep)
            best_idx   = merged_i.gather(1, keep)

        return best_dists, best_idx

    # ------------------------------------------------------------------
    # Insertion — ring-buffer scatter + dict maintenance
    # ------------------------------------------------------------------

    def add_batch(
        self,
        action: int,
        states: torch.Tensor,   # (n, d) float32
        values: torch.Tensor,   # (n,) float64
    ) -> None:
        """Insert (state, value) pairs, evicting the least-recently-updated
        entry once the per-action buffer is full (Blundell et al. 2016 §2).

        While the buffer is still filling, slots are handed out sequentially.
        Once ``_sizes[action] == capacity``, each insertion reclaims the slot
        of the LRU entry via ``popitem(last=False)``.  Newly written keys go
        to the most-recent end of ``_key_to_slot[action]``.

        Duplicate keys are collapsed up front (keeping the larger value, per
        Eq. 1) so one slot is never claimed twice in a single call.
        """
        n = len(states)
        if n == 0:
            return

        if states.device != self.device:
            states = states.to(self.device)
            values = values.to(self.device)

        self._init_states(states.shape[1])

        if n > self.capacity:
            states = states[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity

        new_keys = self._make_keys(states)   # one GPU→CPU sync (n rows)

        # --- Collapse duplicate keys within this batch, keeping max value ---
        # step() already de-duplicates, so this is normally a no-op; it keeps
        # add_batch() correct as a standalone API.
        values_list = values.tolist()
        best_row: dict[bytes, int] = {}
        for i, key in enumerate(new_keys):
            j = best_row.get(key)
            if j is None or values_list[i] > values_list[j]:
                best_row[key] = i
        if len(best_row) != n:
            keep = sorted(best_row.values())
            keep_t = torch.tensor(keep, dtype=torch.long, device=self.device)
            states   = states[keep_t]
            values   = values[keep_t]
            new_keys = [new_keys[i] for i in keep]
            n = len(new_keys)

        # --- Resolve a slot for every key (LRU eviction when full) ----------
        k_to_s = self._key_to_slot[action]
        slots_list:    list[int] = []
        existing_rows: list[int] = []
        for i, key in enumerate(new_keys):
            slot = k_to_s.get(key)
            if slot is not None:
                # Already stored (step() filters these out; defensive path).
                # Refresh recency and preserve Eq. (1)'s "never decrease".
                k_to_s.move_to_end(key)
                existing_rows.append(i)
            elif self._sizes[action] < self.capacity:
                slot = self._sizes[action]
                self._sizes[action] += 1
                k_to_s[key] = slot
            else:
                _, slot = k_to_s.popitem(last=False)   # least recently updated
                k_to_s[key] = slot
            slots_list.append(slot)

        slots = torch.tensor(slots_list, dtype=torch.long, device=self.device)

        if existing_rows:
            rows = torch.tensor(existing_rows, dtype=torch.long, device=self.device)
            values = values.clone()
            values[rows] = torch.maximum(values[rows], self.values[action, slots[rows]])

        self.states[action, slots] = states
        self.values[action, slots] = values

    # ------------------------------------------------------------------
    # LRU bookkeeping
    # ------------------------------------------------------------------

    def touch(self, action: int, keys: list[bytes]) -> None:
        """Mark ``keys`` as most-recently-updated for ``action``.

        Called by ``MFECAlgorithm.step()`` after an Eq. (1) max-update on an
        exact match, so those entries move to the back of the eviction queue.
        Keys that are no longer present are ignored.
        """
        k_to_s = self._key_to_slot[action]
        for key in keys:
            if key in k_to_s:
                k_to_s.move_to_end(key)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        d: dict = {
            "num_actions":   self.num_actions,
            "capacity":      self.capacity,
            "k":             self.k,
            "device":        self.device,
            "key_scale":     self._key_scale,
            "_sizes":        list(self._sizes),
            "action_states": None,
            "action_values": None,
        }
        if self.states is None:
            return d

        action_states, action_values, sizes = [], [], []
        for a in range(self.num_actions):
            slots = list(self._key_to_slot[a].values())
            if not slots:
                action_states.append(None)
                action_values.append(np.array([], dtype=np.float64))
                sizes.append(0)
                continue
            # Emit in LRU order (least-recently-updated first) so __setstate__
            # can rebuild the OrderedDict with the same eviction priority.
            order = torch.tensor(slots, dtype=torch.long, device=self.device)
            action_states.append(self.states[a, order].cpu().numpy())
            action_values.append(self.values[a, order].cpu().numpy())
            sizes.append(len(slots))

        # Sizes are re-derived from the dicts: the saved arrays are compacted
        # to LRU order, so slot i on reload is the i-th least-recent entry.
        d["_sizes"]        = sizes
        d["action_states"] = action_states
        d["action_values"] = action_values
        return d

    def __setstate__(self, d: dict) -> None:
        self.num_actions = d["num_actions"]
        self.capacity    = d["capacity"]
        self.k           = d["k"]
        self.device      = d["device"]
        self.reset_lookup_stats()   # __init__ is bypassed on unpickle
        self._key_scale  = d.get("key_scale", 1e5)   # default for old checkpoints
        self._sizes      = list(d["_sizes"])
        # Pre-LRU checkpoints carry a "_write_ptrs" key; it is ignored — those
        # arrays were saved rotated into write order, which seeds the LRU queue
        # with the old FIFO ordering.  Nothing else about them changes.

        dev = self.device
        self.values = torch.empty(self.num_actions, self.capacity, dtype=torch.float64, device=dev)

        # Rebuilt from the restored tensors below, in saved (LRU) order.
        self._key_to_slot = [OrderedDict() for _ in range(self.num_actions)]

        if d["action_states"] is None:
            self.states = None
            return

        state_dim = next(
            (s.shape[1] for s in d["action_states"]
             if s is not None and s.ndim == 2 and s.shape[0] > 0),
            None,
        )
        if state_dim is None:
            self.states = None
            return

        self.states = torch.empty(
            self.num_actions, self.capacity, state_dim, dtype=torch.float32, device=dev
        )
        for a, (s_np, v_np) in enumerate(zip(d["action_states"], d["action_values"])):
            sz = self._sizes[a]
            if sz > 0 and s_np is not None:
                self.states[a, :sz] = torch.from_numpy(s_np).to(dev)
                self.values[a, :sz] = torch.from_numpy(v_np).to(dev)

        # Rebuild the dicts from the restored tensors (no need to pickle them).
        # Slots were written in saved order, which is LRU order, so inserting
        # by ascending slot index restores the eviction queue exactly.
        for a in range(self.num_actions):
            sz = self._sizes[a]
            if sz == 0:
                continue
            keys_a = self._make_keys(self.states[a, :sz])
            k_to_s = self._key_to_slot[a]
            for slot, key in enumerate(keys_a):
                k_to_s[key] = slot