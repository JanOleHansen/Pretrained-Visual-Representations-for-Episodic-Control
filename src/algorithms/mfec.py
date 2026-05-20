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

function UpdateMemory(s, a, G, time):
    if s is in Q_EC[a]:
        Q_EC[a][s] ← max(Q_EC[a][s], G)       # keep best return seen
    else:
        if |Q_EC[a]| ≥ capacity:
            evict least-recently-used entry from Q_EC[a]
        insert (s, G) into Q_EC[a]
"""

from __future__ import annotations
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor
from sklearn.neighbors import KDTree

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


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
            # Key used to retrieve the pixel observation from a TensorDict.
            obs_key: str = "pixels",

            #   Episodic memory
            # Maximum number of (state, return) entries stored per action.
            # Older entries (by LRU timestamp) are evicted when this is reached.
            buffer_size: int = 1_000_000,
            # How many nearest neighbours to average when estimating a Q-value
            # for an unseen state.  Higher k → smoother estimates, higher cost.
            k: int = 11,
            # Dimensionality of the projected state embedding.  Lower values
            # reduce memory and kNN cost; higher values preserve more structure.
            state_dim: int = 64,

            #   Discount
            # γ — future rewards are worth γ^t times a reward t steps away.
            gamma: float = 0.99,

            #   ε-greedy exploration schedule
            # At the start of training, every action is random (eps_start=1.0).
            # ε is linearly annealed to eps_end over annealing_frames steps.
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            annealing_frames: int = 1_000_000,

            #   Data collection
            # Number of environment steps collected before each call to step().
            frames_per_batch: int = 1_000,
            # Hard cap on episode length; -1 means no limit.
            max_frames_per_traj: int = -1,
    ) -> None:
        
        super().__init__(device)

        # Store all hyperparameters as instance attributes so setup() and
        # step() can access them.
        self.obs_key = obs_key
        self.buffer_size = buffer_size
        self.k = k
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.annealing_frames = annealing_frames
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj

        # Running count of all environment frames seen so far.  Used as an
        # LRU timestamp when inserting entries into the QEC buffers.
        self._collected_frames = 0

    # 
    # Setup — called once before training starts
    # 

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)

        # Strip any leading ParallelEnv batch dims — keep only (C, H, W)
        sample_shape = obs_shape[-3:]
        obs_flat_dim = int(np.prod(sample_shape))
        self.projection = np.random.randn(obs_flat_dim, self.state_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

        self.qec = QEC(range(num_actions), self.buffer_size, self.k)
        self._num_actions = num_actions

        self.qec_policy = QECPolicy(self.qec, self.projection, num_actions)

        self.q_actor = QValueActor(
            module=self.qec_policy,
            spec=action_spec,
            in_keys=[self.obs_key],
        )

        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
        )

        self._explore_policy = TensorDictSequential(self.q_actor, self.greedy_module)

    #
    #   Training step — no gradient updates, memory-only update
    #

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one batch of collected transitions and update QEC memory.

        MFEC has no neural-network parameters to optimise.  The update is:
          1. Compute discounted Monte Carlo returns for each episode in the batch.
          2. Insert (state_embedding, return) pairs into the per-action QEC buffers.

        The trainer may call this with data from partial episodes (the batch
        boundary does not coincide with episode boundaries).  We detect episode
        ends via the "done" flag and process each episode independently.

        Parameters
        ----------
        batch : TensorDict
            Transitions from SyncDataCollector.  May span multiple episodes.

        Returns
        -------
        dict
            Scalar metrics logged by the trainer (epsilon, average buffer fill).
        """

        # Flatten any leading batch/time axes so we have a single sequence of transitions indexed by t = 0 … n-1.
        batch = batch.reshape(-1)
        n = batch.numel()  # total number of transitions

        # Advance the ε-greedy annealing schedule by n steps.
        self.greedy_module.step(n)
        self._collected_frames += n

        # Extract arrays from the TensorDict.  We work in numpy because the QEC memory and KD-tree are numpy/scikit-learn objects.
        obs        = batch[self.obs_key]                                           # [n, C, H, W]
        actions_np = batch["action"].cpu().numpy().flatten().astype(int)           # [n]
        rewards_np = batch["next", "reward"].cpu().numpy().flatten().astype(np.float64)  # [n]
        dones_np   = batch["next", "done"].cpu().numpy().flatten().astype(bool)    # [n]

        # Embed all n observations in a single vectorised call to avoid the overhead of calling embed() inside the per-step loop.
        states = self.qec_policy.embed(obs)  # (n, state_dim) numpy array

        #   Episode segmentation and Monte Carlo return computation
        # Walk through the batch and split it at "done" boundaries.  For each
        # completed episode we compute the discounted return G_t for every step
        # and write the (state, return) pairs into QEC.
        episode_start = 0
        for t in range(n):
            # A "done" flag marks the *last* transition of an episode. We also flush the trailing partial episode when we reach the end
            # of the batch (t == n - 1), even if no done flag was set — the returns are still valid; the episode just continues in the next batch.
            if dones_np[t] or t == n - 1:
                ep          = slice(episode_start, t + 1)
                ep_states   = states[ep]      # (ep_len, state_dim)
                ep_actions  = actions_np[ep]  # (ep_len,)
                ep_rewards  = rewards_np[ep]  # (ep_len,)
                ep_len      = len(ep_rewards)

                # Compute discounted Monte Carlo returns by walking backward:
                #   G[T-1] = r[T-1]
                #   G[t]   = r[t] + γ · G[t+1]
                # This gives the total discounted return from each step t to the end of the episode, which is what MFEC stores in QEC.
                G = np.zeros(ep_len)
                G[-1] = ep_rewards[-1]
                for i in range(ep_len - 2, -1, -1):
                    G[i] = ep_rewards[i] + self.gamma * G[i + 1]

                # Absolute frame index of the first step in this episode. Used as an LRU timestamp so older entries are evicted first.
                frame_time = self._collected_frames - n + episode_start
                for i in range(ep_len):
                    self.qec.update(
                        ep_states[i],    # state embedding
                        ep_actions[i],   # action taken
                        G[i],            # discounted return from this step
                        frame_time + i,  # LRU timestamp
                    )

                episode_start = t + 1  # next episode begins after this done flag

        self.qec.rebuild_trees()  # one rebuild per step() call instead of per insert
        return {
            # Current ε value — tells us how often the agent acts randomly.
            "train/epsilon": float(self.greedy_module.eps),
            # Mean number of entries stored across all per-action buffers. Grows toward buffer_size, then stays flat once memory is full.
            "train/qec_size": float(np.mean([len(b) for b in self.qec.buffers])),
        }

    #
    #   Policy access — called by StepTrainer
    #

    def get_policy(self) -> TensorDictModule:
        """Return the greedy (deterministic) policy used at evaluation time."""
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        """Return the ε-greedy policy used during data collection."""
        return self._explore_policy

    def get_collector_config(self) -> CollectorConfig:
        """Tell StepTrainer how to configure SyncDataCollector."""
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    #
    #   Checkpointing
    #

    def _get_training_state(self) -> TrainingState:
        """Serialise everything needed to resume training from a checkpoint.

        Because MFEC has no neural-network weights, the state is just the
        random projection matrix and the full contents of every ActionBuffer.
        """
        return TrainingState(
            step=0,
            policy_state_dict={},     # no learnable parameters
            optimizer_state_dict={},  # no optimiser
            extra={
                "projection": self.projection,
                # Each ActionBuffer stores three parallel lists.  We save all
                # three so the kNN tree can be rebuilt exactly after loading.
                "qec_state": [
                    {"states": b.states, "values": b.values, "times": b.times}
                    for b in self.qec.buffers
                ],
                "collected_frames": self._collected_frames,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        """Restore algorithm state from a checkpoint created by _get_training_state."""
        self.projection = state.extra["projection"]
        # Update the reference inside QECPolicy so it uses the restored matrix,
        # not the one initialised in setup().
        self.qec_policy.projection = self.projection
        self._collected_frames = int(state.extra["collected_frames"])

        for buf, saved in zip(self.qec.buffers, state.extra["qec_state"]):
            buf.states = saved["states"]
            buf.values = saved["values"]
            buf.times  = saved["times"]
            # Rebuild the spatial index so kNN lookups work immediately after
            # loading without needing to re-insert every entry.
            if buf.states:
                buf._tree = KDTree(np.array(buf.states))


# ---------------------------------------------------------------------------
# Policy module — converts observations to Q-value vectors
# ---------------------------------------------------------------------------

class QECPolicy(nn.Module):
    """Non-parametric nn.Module that estimates Q-values via kNN memory lookup.

    This module holds no learnable weights.  It wraps the QEC memory so that
    QValueActor can call it through TorchRL's TensorDict interface.

    embed() and forward() are split so that MFECAlgorithm.step() can call
    embed() on its own (to get embeddings for the whole batch efficiently)
    without going through the full forward pass.
    """

    def __init__(self, qec: "QEC", projection: np.ndarray, num_actions: int) -> None:
        super().__init__()
        self.qec = qec
        self.projection = projection   # fixed (H*W, state_dim) Gaussian matrix
        self.num_actions = num_actions

    def embed(self, obs: torch.Tensor) -> np.ndarray:
        obs_np = obs.cpu().numpy()
        # Flatten leading dims; keep last C*H*W as the feature vector
        flat = obs_np.reshape(-1, self.projection.shape[0])
        return flat @ self.projection

    def forward(self, obs):
        states = self.embed(obs)
        
        def estimate_action(action):
            return [self.qec.estimate(s, action) for s in states]
        
        # Query all actions in parallel
        results = list(self.qec.executor.map(
            estimate_action, range(self.num_actions)
        ))
        
        # results shape: (num_actions, batch) → transpose to (batch, num_actions)
        q_values = np.array(results, dtype=np.float32).T
        q_values = np.where(np.isinf(q_values), 1e9, q_values)
        
        out = torch.tensor(q_values, dtype=torch.float32, device=obs.device)
        leading = obs.shape[:-3]
        if leading:
            return out.reshape(*leading, self.num_actions)
        else:
            return out.squeeze(0) if out.shape[0] == 1 else out

# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

class QEC:
    """Q-value Episodic Controller.

    Holds one ActionBuffer per discrete action and exposes the two core
    operations from the MFEC paper: estimate() and update().

    Parameters
    ----------
    actions     : iterable of action indices (e.g. range(num_actions))
    buffer_size : maximum entries per ActionBuffer before LRU eviction
    k           : number of nearest neighbours for Q-value estimation
    """

    def __init__(self, actions, buffer_size: int, k: int) -> None:
        # One independent buffer per action.  Using a tuple prevents accidental
        # reassignment of individual buffers.
        self.buffers = tuple(ActionBuffer(buffer_size) for _ in actions)
        self.k = k
        self.executor = ThreadPoolExecutor(max_workers=len(self.buffers))

    def rebuild_trees(self):
            for buf in self.buffers:
                if buf.states:
                    buf._tree = KDTree(np.array(buf.states))

    def estimate(self, state: np.ndarray, action: int) -> float:
        """Estimate Q(state, action) from the episodic memory.

        Three cases (in priority order):
        1. Exact match  → return the stored return directly.
        2. Fewer than k entries in buffer  → return +inf (optimistic, encourages
           exploration of this action).
        3. General case → average the k nearest neighbours' returns.

        Parameters
        ----------
        state  : np.ndarray, shape (state_dim,) — embedded observation
        action : int — index into self.buffers

        Returns
        -------
        float
        """
        buffer = self.buffers[action]

        # Check for an exact state match first (O(log n) via KD-tree).
        state_index = buffer.find_state(state)
        if state_index is not None:
            # We have seen this exact state before — return the best return.
            return buffer.values[state_index]

        if len(buffer) <= self.k:
            # Not enough data to compute a meaningful kNN average.
            # +inf signals "unknown but optimistic" to the argmax selector.
            return float('inf')

        # General case: retrieve the k nearest stored states and average their
        # returns.  This is the core kNN regression step from the paper.
        neighbours = buffer.find_k_nearest(state, self.k)  # list of buffer indices
        value = sum(buffer.values[idx] for idx in neighbours)
        return value / self.k

    def update(self, state: np.ndarray, action: int, return_: float, time: int) -> None:
        """Insert or update a (state, return) entry in the action's buffer.

        If the state already exists, we keep the *maximum* of the old and new
        returns (optimistic update).  This prevents a single unlucky episode
        from overwriting previously observed high returns.

        Parameters
        ----------
        state   : np.ndarray, shape (state_dim,)
        action  : int
        return_ : discounted Monte Carlo return G_t computed for this step
        time    : absolute frame index used as an LRU eviction timestamp
        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index is not None:
            # State already in memory — update value (keep max) and timestamp
            # (keep most recent so the entry survives LRU eviction).
            max_value = max(buffer.values[state_index], return_)
            max_time  = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            # New state — add to the buffer.  If the buffer is full, the oldest
            # entry is evicted automatically inside ActionBuffer.add().
            buffer.add(state, return_, time)


class ActionBuffer:
    """Fixed-capacity nearest-neighbour memory for a single action.

    Internally maintains three parallel lists — states, values, times —
    along with a scikit-learn KDTree for fast O(log n) kNN queries.

    The KDTree is rebuilt after every structural change (add or replace).
    This keeps the implementation simple at the cost of O(n log n) rebuild
    time per insert.  For very large buffers a more incremental structure
    (e.g. FAISS) could be used instead.

    Eviction policy: when at capacity, the entry with the *smallest* timestamp
    (least recently used) is replaced by the incoming entry, provided the
    incoming entry has a more recent timestamp.

    Parameters
    ----------
    capacity : int — maximum number of (state, return) entries to store
    """

    def __init__(self, capacity: int) -> None:
        # KD-tree is built lazily; None means the buffer is empty.
        self._tree: KDTree | None = None
        self.capacity = capacity
        self.states: list[np.ndarray] = []  # list of (state_dim,) arrays
        self.values: list[float] = []       # corresponding best returns
        self.times:  list[int]   = []       # corresponding LRU timestamps

    def find_state(self, state: np.ndarray) -> int | None:
        """Return the index of an exactly matching stored state, or None.

        Uses the KD-tree to find the nearest neighbour and then verifies
        numerical equality with np.allclose (tolerates floating-point noise).
        """
        if self._tree is None:
            return None  # buffer is empty

        # query() returns (distances, indices).  We ask for 1 neighbour and
        # extract its index: [1][0][0] = indices → first query point → first hit.
        neighbour_index = self._tree.query([state], k=1)[1][0][0]
        if np.allclose(self.states[neighbour_index], state):
            return int(neighbour_index)
        return None

    def find_k_nearest(self, state: np.ndarray, k: int) -> list[int]:
        """Return the buffer indices of the k nearest stored states.

        Returns an empty list when the buffer is empty (tree not built yet).
        """
        if self._tree is None:
            return []
        # query([state], k=k) → (distances, indices), shape (1, k) each.
        # [1][0] gives the index array for our single query point.
        return self._tree.query([state], k=k)[1][0]

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_index = int(np.argmin(self.times))
            if time > self.times[min_time_index]:
                self.states[min_time_index] = state
                self.values[min_time_index] = value
                self.times[min_time_index] = time
        # tree rebuild deferred!

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time
        # tree rebuild deferred!

    def __len__(self) -> int:
        return len(self.states)