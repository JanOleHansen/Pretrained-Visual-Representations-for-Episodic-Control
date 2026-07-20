# Agent instructions for torchrl-hydra-template

## Project overview

A modular reinforcement learning research template built on
[TorchRL](https://github.com/pytorch/rl) and
[Hydra](https://github.com/facebookresearch/hydra). Three composable components —
**Environment**, **Algorithm**, **Trainer** — are wired together by `src/train.py`.

Implemented experiments:

| Algorithm | Environment      | Experiment config              |
|-----------|------------------|--------------------------------|
| DQN       | CartPole-v1      | `experiment=dqn/cartpole`      |
| DQN       | ALE/Pong-v5      | `experiment=dqn/pong`          |
| DDPG      | HalfCheetah-v4   | `experiment=ddpg/halfcheetah`  |
| A2C       | HalfCheetah-v4   | `experiment=a2c/halfcheetah`   |
| MFEC      | ALE/Pong-v5      | `experiment=mfec/pong`         |
| MFEC      | ALE/Breakout-v5  | `experiment=mfec/breakout`     |
| MFEC      | ALE/Qbert-v5     | `experiment=mfec/qbert`        |
| NEC       | ALE/Pong-v5      | `experiment=nec/pong`          |

## Design principles

1. **Readable algorithm code.** Each algorithm file should read close to the
   pseudocode from the paper. `step()` is short and corresponds to the update
   equations. Long config-shuffling and framework glue belong elsewhere.
2. **Hard separation of responsibilities.**
   - **Algorithm** owns everything that affects the learning curve: network, replay
     buffer, loss, optimiser, exploration, target-net schedule, and the collector
     config (`frames_per_batch`, `init_random_frames`, ...). All hyperparameters live
     as keyword arguments on `__init__`.
   - **Trainer** owns the loop. It creates the collector from
     `algorithm.get_collector_config()`, calls `algorithm.step(batch)`, manages the
     device, fires callbacks, and checkpoints.  Nothing on the trainer config affects
     reward or sample efficiency.
   - **Environment** is a fixed task definition: env name + transform list. It does
     not know about the algorithm.
3. **One source of truth per concern.** HP defaults live in the algorithm's
   `__init__` (with type hints + docstrings). YAML mirrors them for overrides.
4. **Callable factories via Hydra.** Design choices that are `Callable`s (replay
   buffer, network) are configured in `configs/algorithm/*.yaml` with `_partial_`
   and nested `_target_` nodes. **`src/train.py` and `src/eval.py` build the
   algorithm with `hydra.utils.instantiate(cfg.algorithm, device=None)`** so those
   nested configs become real callables. Plain `OmegaConf.to_container` + `**kwargs`
   would pass dicts instead of partials.

## Algorithm constructor pattern

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # Design choices: factories injected as Callables
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(...),
        # Q-net factory; setup() passes (obs_shape, num_actions) — see below.
        network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_q_net, num_cells=[120, 84], activation_class=nn.ReLU
        ),
        # Observation tensordict key (e.g. "observation" for vector obs, "pixels" for image obs).
        obs_key: str = "observation",
        # Scalar HPs
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 100,
        hard_update_freq: int = 50,
    ) -> None:
        super().__init__(device)
        # ... store kwargs verbatim ...
```

Rules:
- `*` makes every HP keyword-only.
- `BaseAlgorithm.__init__(device)` — **no `cfg` parameter**. Algorithms read env
  specs from `make_env()` inside `setup()`.
- `replay_buffer` is a **no-arg** factory returning a `ReplayBuffer`.
- `network` (DQN) is a factory called as **`network(obs_shape, num_actions)`** —
  positional `obs_shape` is the raw observation shape tuple (e.g. `(4,)` for
  CartPole, `(4, 84, 84)` for stacked Atari frames) and `num_actions` is the
  discrete action count. For DDPG, `actor_network` and `value_network` use the
  same call signature with the continuous action vector size. Use the helpers
  in `src/networks.py`:
    - `make_mlp_q_net(obs_shape, num_actions, *, num_cells, activation_class)` —
      flattens `obs_shape` into a torchrl `MLP`. Default for vector observations.
    - `NatureDQN(obs_shape, num_actions, *, ...)` — Mnih et al. 2015 ConvNet+MLP
      head. Default for image observations.
    - `make_mlp_ddpg_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for a deterministic actor (DDPG); no final tanh, the algorithm
      wraps it in `TanhModule` to rescale to the action spec.
    - `make_mlp_ddpg_critic(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-action critic; takes `[obs, action]` concatenated by `ValueOperator`.
    - `make_mlp_a2c_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for an A2C stochastic actor; outputs `2 * action_dim` features
      that `NormalParamExtractor` splits into `loc` and `scale` for TanhNormal.
    - `make_mlp_a2c_value(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-value (V(s)) critic; `action_dim` is unused but kept for signature
      parity with the actor factory.
  All keep everything after the two positional args **kwarg-only**, so a Hydra
  `_partial_` config can pre-bind kwargs without colliding with `setup()`'s call.
- `obs_key` selects which tensordict key the observation comes from. Vector
  envs (CartPole) use `"observation"`; pixel envs (Atari with `from_pixels=True`)
  use `"pixels"`. The key is forwarded to `QValueActor.in_keys` and used to read
  the spec for the network factory.
- **Activation class in YAML:** `torchrl.modules.MLP` expects `activation_class`
  to be a **type** (it instantiates internally). In Hydra YAML, **`_target_:
  torch.nn.ReLU` nests an instantiation** and produces a module instance, which
  breaks `MLP`. Use **`hydra.utils.get_class`** instead:

  ```yaml
  activation_class:
    _target_: hydra.utils.get_class
    path: torch.nn.ReLU
  ```

- Scalar HPs are plain kwargs and **do** appear in YAML.

## `step(batch)` shape

```python
def step(self, batch: TensorDict) -> dict[str, float]:
    # 1. Always — anneal exploration, store transitions
    batch = batch.reshape(-1)
    self.greedy_module.step(batch.numel())
    self.replay_buffer.extend(batch)
    self._collected_frames += batch.numel()

    # 2. Warm-up gate
    if self._collected_frames < self.init_random_frames:
        return {"train/epsilon": float(self.greedy_module.eps)}

    # 3. Optimisation loop — sample, loss, backward, optimiser, target update
    for j in range(self.num_updates):
        sample = self.replay_buffer.sample(self.batch_size).to(self.device)
        loss = self.loss_module(sample)["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.target_updater.step()

    return {"train/q_loss": ..., "train/epsilon": ...}
```

The trainer never touches the replay buffer, target network or epsilon — those are
algorithm internals. Per-batch metrics (`train/episode_reward`,
`train/episode_length`, `train/q_values`) and timing (`time/collect`,
`time/step`, `time/speed`) are computed by `StepTrainer` from the collector
batch and merged into the algorithm's metrics dict at logging boundaries.
This mirrors the torchrl SOTA DQN reference and keeps batch-level bookkeeping
out of the algorithm.

### On-policy variant (A2C)

A2C drops three of those internals entirely: no long-term replay buffer, no
target networks, no warm-up. Each `step(batch)` runs `GAE` on the rollout
under `no_grad`, refills a one-shot buffer with `SamplerWithoutReplacement`,
and does one epoch of mini-batch updates with `A2CLoss`. The buffer in
`a2c.py` is built directly in `setup()` (not exposed as a `_partial_`
factory) because its size is locked to `frames_per_batch / mini_batch_size`
— it's an implementation detail of the on-policy schedule, not a research
choice. `get_collector_config()` returns `init_random_frames=0` since the
stochastic actor explores from frame zero.

## Instantiation in `src/train.py` / `src/eval.py`

```python
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

algorithm = instantiate(cfg.algorithm, device=None)  # recursive; resolves _partial_ factories

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)

TrainerClass = get_class(cfg.trainer._target_)
trainer = TrainerClass(cfg=cfg, algorithm=algorithm, environment=environment)
```

The **environment** is still unpacked from a flat dict. The **algorithm** must use
`instantiate` whenever its YAML contains nested `_target_` / `_partial_` nodes
(e.g. `replay_buffer`, `network`).

YAML values override Python defaults where present; absent keys fall back to
constructor defaults.

## Environment

`Environment.__init__` accepts:
- `name`: gymnasium env id (e.g. `"CartPole-v1"`, `"ALE/Pong-v5"`).
- `transforms`: list of `_target_`-keyed dicts, each instantiated as a
  `torchrl.envs.transforms` object and composed on top of the base env.
  Always include `StepCounter` explicitly. Add `RewardSum` if you want
  `train/episode_reward` in the trainer metrics — it populates the
  `("next", "episode_reward")` key the trainer reads.
- `gym_kwargs`: optional dict forwarded straight to `GymEnv` (e.g.
  `{"frame_skip": 4, "from_pixels": true, "pixels_only": false,
  "categorical_action_encoding": true}` for Atari).
- `gym_backend`: optional backend name (`"gymnasium"`); if set, the GymEnv
  construction is wrapped in `set_gym_backend(...)`.

```yaml
# configs/environment/cartpole.yaml
name: CartPole-v1
transforms:
  - _target_: torchrl.envs.transforms.StepCounter
  - _target_: torchrl.envs.transforms.RewardSum
```

```yaml
# configs/environment/pong_train.yaml — pixel-based Atari env
name: ALE/Pong-v5
gym_backend: gymnasium
gym_kwargs:
  frame_skip: 4
  from_pixels: true
  pixels_only: false
  categorical_action_encoding: true
transforms:
  - _target_: torchrl.envs.NoopResetEnv
    noops: 30
    random: true
  # ... (see configs/environment/pong_train.yaml for the full SOTA stack)
```

The factory in `src/environments/factory.py` supports gymnasium only.
For >1 `num_envs`, workers run on CPU (`ParallelEnv` with `mp_start_method="spawn"`).

### Separate evaluation env

`BaseTrainer` accepts an optional `eval_environment: Environment | None` arg.
When set, `evaluate()` builds its eval env from it; otherwise it falls back to
`environment`. Wire it in via Hydra package overrides:

```yaml
# configs/train.yaml (and eval.yaml)
defaults:
  - environment: ???
  - environment@eval_environment: null   # default: no separate eval env

# configs/experiment/dqn/pong.yaml
defaults:
  - override /environment: pong_train
  - override /environment@eval_environment: pong_eval
```

`src/train.py` and `src/eval.py` build the eval `Environment` via
`cfg.get("eval_environment")` and pass it to the trainer constructor.

Use this when training-time and evaluation-time observations should differ
(e.g. Atari, where `EndOfLifeTransform` and `SignTransform` are train-only and
`VecNorm` is dropped at eval because its running stats are not checkpointed).

## Trainer

`StepTrainer` is the only trainer.  It:
- creates a `torchrl.collectors.Collector` from `algorithm.get_collector_config()`
  and `cfg.trainer.total_frames`;
- iterates the collector, calls `algorithm.step(batch)`, and fires
  `ON_STEP_END` callbacks at logging boundaries;
- delegates device resolution to `src/utils/device.py`.

`BaseTrainer` owns env lifecycle, `evaluate(num_episodes)` (greedy rollout), and
checkpoint orchestration.

## File map

```
src/
  train.py                  — entry point; instantiate(cfg.algorithm); environment **kwargs
  eval.py                   — evaluation entry point; same algorithm instantiation
  networks.py               — network factories: make_mlp_q_net, NatureDQN,
                              NatureEmbedding (NEC CNN trunk, no Q-head),
                              make_mlp_ddpg_actor, make_mlp_ddpg_critic,
                              make_mlp_a2c_actor, make_mlp_a2c_value
  algorithms/
    base.py                 — BaseAlgorithm ABC; TrainingState and CollectorConfig dataclasses
    dqn.py                  — DQNAlgorithm; replay/network factories (defaults + setup contract)
    ddpg.py                 — DDPGAlgorithm; actor/critic/replay/noise factories
    a2c.py                  — A2CAlgorithm; on-policy actor/critic with GAE + A2CLoss
    nec.py                  — NECAlgorithm; DND class, DNDPolicy, N-step returns, dual updates
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: gymnasium + transforms list + gym_kwargs/gym_backend
  trainers/
    BaseTrainer.py          — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    StepTrainer.py          — StepTrainer (Collector-driven loop)
  callbacks/                — ProgressCallback, CheckpointCallback, WandBLogger, TensorBoardLogger
  utils/                    — device resolution, seeding, callback builders
configs/
  algorithm/dqn.yaml        — DQN HPs (CartPole defaults); _partial_ replay_buffer + network
  algorithm/dqn_atari.yaml  — DQN HPs (Atari/NatureDQN defaults; pixel obs)
  algorithm/ddpg.yaml       — DDPG HPs (HalfCheetah defaults); _partial_ actor/critic/noise
  algorithm/a2c.yaml        — A2C HPs (HalfCheetah/MuJoCo defaults); _partial_ actor/value
  algorithm/mfec_atari.yaml — MFEC HPs (Atari defaults: buffer_size=1M, k=11, state_dim=64)
  algorithm/nec.yaml        — NEC HPs (base defaults); _partial_ embedding_network + replay_buffer
  algorithm/nec_atari.yaml  — NEC HPs (Atari defaults per Pritzel et al. 2017 Table S1)
  environment/cartpole.yaml — env kwargs (name, transforms)
  environment/pong_train.yaml     — Atari Pong (training transforms incl. EndOfLife + Sign + VecNorm; DQN only)
  environment/pong_mfec_train.yaml — Atari Pong for MFEC (same stack WITHOUT VecNorm; see note below)
  environment/pong_eval.yaml      — Atari Pong (eval transforms; drops EndOfLife + Sign + VecNorm)
  environment/breakout_train.yaml — Atari Breakout (training transforms; no VecNorm — MFEC only)
  environment/breakout_eval.yaml  — Atari Breakout (eval transforms)
  environment/qbert_train.yaml    — ALE/Qbert-v5 (training transforms; no VecNorm — MFEC only)
  environment/qbert_eval.yaml     — ALE/Qbert-v5 (eval transforms)
  environment/halfcheetah.yaml — HalfCheetah-v4 (DoubleToFloat + InitTracker)
  experiment/dqn/cartpole.yaml — composed CartPole experiment
  experiment/dqn/pong.yaml     — composed Atari Pong experiment
  experiment/ddpg/halfcheetah.yaml — composed DDPG HalfCheetah experiment
  experiment/a2c/halfcheetah.yaml — composed A2C HalfCheetah experiment
  experiment/mfec/pong.yaml    — MFEC on Pong (40M frames, num_envs=16)
  experiment/mfec/breakout.yaml — MFEC on Breakout (1M frames)
  experiment/mfec/qbert.yaml   — MFEC on Q*Bert (40M frames, num_envs=16)
  experiment/nec/pong.yaml     — NEC on Pong (40M frames, num_envs=16)
  logger/{wandb,tensorboard}.yaml
  paths/default.yaml
  train.yaml, eval.yaml
tests/
  test_smoke.py             — DQN-on-CartPole, DQN-on-Pong, DDPG-on-HalfCheetah, A2C-on-HalfCheetah, MFEC-on-Pong, NEC-on-Pong smoke tests
```

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py` following the kwargs pattern above. Use
   `Callable` factories for design choices (inline lambdas, `functools.partial`,
   or small helpers). Document the **call signature** each factory must satisfy
   (e.g. `network(obs_shape, num_actions)`).
2. Implement `setup(make_env)`, `step(batch) -> dict`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Create `configs/algorithm/my_algo.yaml` with `_target_`, scalar HPs, and any
   `_partial_` / nested `_target_` blocks for factories. Use `instantiate`-
   compatible patterns (see DQN: replay buffer + partial `MLP`).
4. Create `configs/experiment/my_algo/<env>.yaml` composing your algo + env.
5. **Update `README.md` and `AGENTS.md`.**
6. Add a smoke test in `tests/test_smoke.py`.

## VecNorm + MFEC incompatibility

**Never add `VecNorm` to an MFEC environment config.** VecNorm updates its
running mean/std with every collected frame, so the same raw pixel observation
produces different normalised float32 values at different timesteps.
MFEC's exact-match dict (`QEC._key_to_slot`) requires bit-identical
embeddings for the same game state to produce the same quantised hash key;
VecNorm breaks this invariant and drives `train/exact_hit_rate` permanently
to 0 even when 165k+ states are stored.

Concretely:

| env config | VecNorm | intended for |
|---|---|---|
| `pong_train.yaml` | ✓ | DQN pong only |
| `pong_mfec_train.yaml` | ✗ | MFEC pong |
| `breakout_train.yaml` | ✗ | MFEC breakout |
| `qbert_train.yaml` | ✗ | MFEC Q*Bert |

The fixed random projection already compresses 28 k-pixel observations
adequately without online whitening.

## Reward scale for Atari MFEC runs

All MFEC training configs include `SignTransform` (reward clipped to
`{-1, 0, +1}`), so `train/episode_reward` (from `RewardSum`) counts clipped
reward events — **not** the true game score. For example, "57" on Q*Bert means
~57 positive reward events, not a score of 57 points. Always compare against
the paper using `eval/return_mean`, which is computed from the `*_eval.yaml`
environment that drops `SignTransform`.

## NEC — DND values, blend-only (deviates from the paper here)

`NECAlgorithm` (Pritzel et al. 2017) differs from MFEC in two ways that
require special care:

### 1. DND `values` tensor is a plain (non-grad) tensor, NOT gradient-enabled

The paper's §3.4 backpropagates into both the embedding network *and* the
DND keys/values (values at a lower LR than the blend rate α). This
implementation does **not** do that: `DND.values` is a plain `torch.Tensor`
with no `requires_grad`, updated only by the in-place blend rule

```python
self.values.data[action, slots] += dnd_lr * (target - old)   # blend update
self.values.data[action, slots]  = new_vals                  # ring-buffer insert
```

and the optimizer covers the embedding network only:

```python
optimizer = torch.optim.Adam(self.embedding_net.parameters(), lr=self.lr)
```

`_gradient_step` reads `dnd.values[a, indices]` as a frozen constant — the
regression-loss gradient reaches the CNN only through the distance term
(`∂w_i/∂h`), never into `values` itself.

**Why:** an earlier version matched the paper (`values` as a gradient-enabled
leaf, included in Adam). This broke because `values` is a *ring buffer*:
Adam's per-slot momentum/variance state is tied to slot *position*, not
entry *identity*. When a slot is evicted and a new, unrelated (key, value)
pair is written into it, the optimizer's stale momentum for the *previous*
occupant gets applied to the *new* one on the next step, and combined with
the separate blend-rule writes touching the same slot out-of-band from
autograd, this drove values negative over time. Freezing `values` (blend
rule only) avoids that interaction. This matches the deviation the
reference GitHub repo (github.com/EndingCredits/Neural-Episodic-Control)
makes from the paper, for a related reason.

If you want to restore paper-faithful gradient updates on `values`, you'd
need to solve the stale-momentum-on-eviction problem first (e.g. reset
Adam's per-slot state on eviction, or use a plain low-LR SGD step for
`values` instead of Adam) — don't just re-add `self.dnd.values` to the
optimizer's parameter list, that's the change that caused the original
regression.

### 2. N-step returns (per-env, complete-episodes-only)

NEC uses bootstrapped N-step returns:

    Q^(N)(s_t, a_t) = Σ_{j=0}^{N-1} γ^j r_{t+j} + γ^N max_{a'} Q(s_{t+N}, a')

The per-env carry-over logic is identical to MFEC's (the `(E, T)` shape
pattern, partial-episode buffering in `_carry`, `lfilter` for MC returns).
The NEC-specific addition is a DND bootstrap correction applied after the
full MC pass:

```python
# mc: full Monte Carlo returns via lfilter
gamma_n = gamma ** n_step
correction = gamma_n * (q_max_at_t_plus_n - mc[n_step:])
n_step_G[:T - n_step] = where(valid, mc[:T - n_step] + correction, mc[:T - n_step])
```

Bootstrapping uses the CURRENT DND state (written by previous episodes in
the same batch, or prior batches).  The carry stores RAW observations (not
pre-computed embeddings) so they are re-embedded with the current network
at the start of each step() call.

### 3. NatureEmbedding factory

`src/networks.NatureEmbedding(obs_shape, embedding_dim, *, ...)` is a CNN
trunk + single dense layer — the same convolutional body as NatureDQN but
with the MLP Q-head replaced.  It is called as
`embedding_network(obs_shape, embedding_dim)` in `NECAlgorithm.setup()`,
matching the `network(obs_shape, num_actions)` convention used by DQN.

### 4. Exact-match dict and VecNorm

NEC's DND also uses a quantised-embedding hash dict (`_key_to_slot`) for
O(1) blend-on-revisit detection.  Unlike MFEC's fixed random projection,
the CNN embedding changes during training, so the same observation will
produce different hash keys after network updates.  The exact-match dict
remains useful for recent re-visits (within the same or adjacent batches)
and for warm-starting early training.

The same VecNorm restriction that applies to MFEC applies to NEC
environments — use `pong_mfec_train.yaml` (no VecNorm) for NEC Pong.

## What not to do

- Do not place learning-affecting knobs on `trainer:` or `environment:` configs.
- Do not create `XxxConfig` dataclasses.
- Do not add `cfg: DictConfig` to `BaseAlgorithm` or pass `cfg=cfg` to algorithms.
- Do not pass `cfg.environment` directly to `Environment()` — unpack as `**kwargs`.
- Do not add `OmegaConf` imports to `base.py`.
- Do not add `VecNorm` to any MFEC environment config (see section above).

## Running

```shell
python src/train.py experiment=dqn/cartpole
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3
python src/train.py experiment=dqn/cartpole 'logger=[wandb]'  # experiments default to wandb; plain CLI defaults to tensorboard
python src/train.py experiment=dqn/pong            # Atari Pong (40M frames, GPU)
python src/train.py experiment=ddpg/halfcheetah    # DDPG continuous control (1M frames)
python src/train.py experiment=a2c/halfcheetah     # A2C on-policy continuous control (1M frames)
python src/train.py experiment=mfec/pong           # MFEC on Pong (40M frames, GPU)
python src/train.py experiment=mfec/qbert          # MFEC on Q*Bert (40M frames, GPU)
pytest tests/test_smoke.py -v
```
