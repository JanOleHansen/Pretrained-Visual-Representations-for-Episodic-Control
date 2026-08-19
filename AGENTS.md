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
| MFEC      | any ALE game     | `experiment=mfec/{rp_gray,rp_rgb,vae,dinov2,resnet,clip,mae} game=<Game>` |
| NEC       | ALE/Pong-v5      | `experiment=nec/pong`          |
| NEC       | ALE/MsPacman-v5  | `experiment=nec/mspacman{,_dinov2,_clip,_mae}` |
| NEC       | ALE/Qbert-v5     | `experiment=nec/qbert{,_dinov2,_clip,_mae}`    |
| NEC       | ALE/Frostbite-v5 | `experiment=nec/frostbite{,_dinov2,_clip,_mae}` |

The last three rows are the **NEC encoder ablation**: 3 games x 4 encoders,
twelve arms, all learning-relevant settings held identical. See "The NEC encoder
ablation" below before touching any of them.

The MFEC row is the **MFEC encoder ablation**: 7 encoders x whichever `game` is
passed, run over the same three games (Ms. Pac-Man, Q\*bert, Frostbite). It has
no per-game files by design — see "`game` — one token per Atari game" below,
which also spells out the Frostbite invocations.

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
algorithm internals. Episode metrics (`train/episode_reward`,
`train/episode_length`, `train/episodes_completed`, `train/q_values`) and timing
(`time/collect`, `time/step`, `time/speed`) are computed by `StepTrainer` from
the collector batches and merged into the algorithm's metrics dict at logging
boundaries, keeping batch-level bookkeeping out of the algorithm.

**Episode metrics are interval means, not per-batch samples.**
`StepTrainer._IntervalStats.update()` runs on *every* collector batch;
`flush()` runs only on a logging boundary, returns the mean over the whole
interval, and resets. This is not cosmetic: with `frames_per_batch=1024` and
`log_every_n_steps=10_000`, measuring only the boundary-crossing batch (as the
old `_batch_metrics` did) discarded ~90% of finished episodes and left ~1-2
Ms. Pac-Man episodes behind each logged point — a 1-2 sample estimate of a
quantity with several hundred points of per-episode spread, which reads as pure
noise. `train/episodes_completed` exposes the sample size behind each point.

Two consequences worth knowing:
- An interval in which **no** episode finished emits no `train/episode_reward`
  at all (a genuine gap), never a stale or zero value.
- `train/q_values` excludes executed-action values above
  `StepTrainer._OPTIMISTIC_Q_THRESHOLD` (1e8). Episodic-control policies
  substitute ~1e9 for state-actions their memory cannot evaluate, and averaging
  those in makes the metric read 1e9 for the whole warm-up. An algorithm that
  reports its own `train/q_values` from `step()` still wins (`setdefault`).

Guarded by `tests/test_interval_metrics.py`.

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

### Periodic in-training evaluation (`EvalCallback`)

`BaseTrainer.evaluate(num_episodes)` runs greedy episodes against
`eval_environment` and returns `eval/return_mean` etc. By default this only
runs when you invoke `src/eval.py` manually against a checkpoint — `train/*`
metrics (from `RewardSum` on the *training* env) are clipped/life-based for
Atari (see "Reward scale for Atari MFEC runs" below — the same clipping
applies to any Atari algorithm whose train env includes `SignTransform`,
not just MFEC) and are **not** comparable to paper-reported scores.

`trainer.eval_every_n_steps` runs evaluation periodically during training and
lands `eval/return_mean` in the normal training logs (TensorBoard/W&B) next to
`train/*`, with no separate `src/eval.py` invocation needed. Set it to `null`
to disable:

```yaml
# configs/experiment/nec/pong.yaml
trainer:
  eval_every_n_steps: 10_000   # shared probe cadence -- see below
  num_eval_episodes: 5
```

#### The shared probe budget

`configs/train.yaml` and **every** config under `configs/experiment/` hold these
identical, so any two runs are read off the same x-axis at the same sample
points. Do not change one config in isolation:

| knob | value | note |
|---|---|---|
| `trainer.total_frames`       | `100_000` | agent steps = 400k raw ALE frames at action repeat 4 |
| `trainer.log_every_n_steps`  | `5_000`   | 20 logged points per run |
| `trainer.eval_every_n_steps` | `10_000`  | 10 eval points per run |
| memory capacity              | `100_000` | `algorithm.buffer_size` (MFEC); `algorithm.replay_buffer.storage.max_size` (DQN/DDPG/NEC) |

Capacity is **pinned equal to `total_frames`**. That makes the no-eviction bound
structural rather than empirical — a run inserts at most one entry per decision,
so capacity can never bind on any `|A|`. **Move the four together:** raising
`total_frames` without raising capacity reintroduces eviction mid-run, which
changes the algorithm partway through and at a different moment in each arm.

This budget is an Atari-100k-style probe, far short of the papers' 40M-frame
runs, so scores off it are **not** comparable to published tables.

**NEC's exploration schedule is part of the set that moves with `total_frames`.**
`annealing_frames` and `init_random_frames` are agent-step counts, exactly like
`total_frames`, so they are only meaningful relative to it. An earlier version
of this section noted that they were "deliberately *not* rescaled" with the cut
from 1M to 100k and suggested overriding them per run. That was wrong — the
values were simply left behind, and the cost was large:

| | before | after |
|---|---|---|
| `annealing_frames` (12 ablation arms) | `50_000` = 50% of the run | `10_000` = 10% |
| `init_random_frames` (12 ablation arms) | `12_500` = 12.5% | `4_800` = 4.8% |
| `annealing_frames` (`nec_atari.yaml` default, inherited by `nec/{pong,hero}`) | `4_000_000` — ε reaches only **0.975** by step 100k | `10_000` |
| `init_random_frames` (same default) | `50_000` = **half the run** uniform random | `4_800` |
| share of collected frames taking a uniform random action | **26.6%** (ablation) / **~99%** (pong, hero) | 6.2% / 7.1% |

The floor on both is when the DND becomes usable: NEC writes only at EPISODE
END, and an action holding `<= k = 50` entries answers `+inf`, so there is
nothing to exploit until the first round of episodes has been written. Measured
episode length under a random policy on the real NEC env stacks (agent steps):

| Q*bert | Frostbite | Ms. Pac-Man | H.E.R.O. | Pong |
|---|---|---|---|---|
| 314 | 371 | 498 | 656 | 922 |

At `num_envs: 8` that puts the first writes between ~2.5k and ~7.4k collected
frames, at 165-1230 entries per action — all well past `k`. `4_800` is also
exactly three collector batches at `frames_per_batch: 1_600`; torchrl rounds
`init_random_frames` **up** to a batch multiple and warns when it has to
(`collectors/_single.py`), so the old `12_500` was silently `12_800`.

Guarded by
`tests/test_nec_ablation_parity.py::test_the_exploration_schedule_fits_inside_the_probe_budget`.
For a paper-scale run (`total_frames: 2_500_000`) restore `annealing_frames:
4_000_000`.

Parity is guarded by `tests/test_encoder_factory.py`
(`test_every_arm_shares_one_training_budget`,
`test_the_budget_is_held_equal_across_games_too`).

`build_callbacks` (`src/utils/instantiate.py`) only adds `EvalCallback` when
`eval_every_n_steps` is truthy, and places it **before** the logger callbacks
in the callback list — `EvalCallback.on_step_end` mutates the shared
`metrics` dict in place, and callbacks fire in list order, so the loggers see
the merged `eval/*` keys on the same `on_step_end` call. This only makes
sense when a dedicated `eval_environment` is configured (see above); without
one, `evaluate()` falls back to the (clipped/life-based) training env and
periodic eval buys you nothing beyond more episodes.

## Trainer

`StepTrainer` is the only trainer.  It:
- creates a `torchrl.collectors.Collector` from `algorithm.get_collector_config()`
  and `cfg.trainer.total_frames`;
- iterates the collector, calls `algorithm.step(batch)`, and fires
  `ON_STEP_END` callbacks at logging boundaries;
- delegates device resolution to `src/utils/device.py`.

`BaseTrainer` owns env lifecycle, `evaluate(num_episodes)` (greedy rollout), and
checkpoint orchestration.

### Paper-ready metrics (logged live, no post-processing)

`evaluate()` and `StepTrainer._training_loop()` emit these so the thesis figures
come straight off W&B. See README "Paper-ready metrics" for the table.

- `eval/hns` — human-normalised score `(return_mean − random)/(human − random)`.
  Baselines in `src/utils/atari_scores.py` (Wang et al. 2016 / SPR Table 3, 26
  Atari 100k games). The game is resolved in `BaseTrainer.__init__` from
  `run.game`/`game`/`environment.name` via `resolve_game`, with
  `throw_on_resolution_failure=False` because `run.game` is often the
  `${hydra:...}` interpolation that raises outside a live Hydra run. Absent (not
  zero) for a non-benchmark game.
- `eval/value_return_corr` (+ `eval/value_return_n`) — Pearson corr between the
  memory's value for the **taken** action and its realised discounted
  return-to-go, pooled over eval steps. Uses `self.algorithm.gamma`; sentinels
  (`|v| ≥ 1e8`) and non-finite pairs dropped; needs ≥2 points with non-zero
  variance or it is omitted. `_taken_action_value` reads `QValueActor`'s
  `action_value`/`action` and handles categorical and one-hot encodings;
  returns `None` (metric skipped) when the policy exposes no action-value.
- `sys/gpu_mem_peak_gb` / `time/elapsed_min` — peak CUDA allocation (counter
  reset at loop start) and cumulative wall-clock; last values are the run's peak
  and total. `_training_loop` reads `self.device` via `getattr` so the stub in
  `tests/test_interval_metrics.py` (which bypasses `setup()`) still runs.
- `eval/num_episodes` — sample size per eval point; `eval/return_std` is
  **omitted** at `num_eval_episodes=1` rather than logged as `0.0`.

The **cross-game aggregate** (mean/median/IQM HNS + CIs) is the one thing that
cannot be a per-run scalar (one run = one game). `scripts/aggregate_results.py`
pulls the runs and computes it with a stratified bootstrap, recomputing
`eval/hns` from `eval/return_mean` for runs that predate the metric.

## File map

```
src/
  train.py                  — entry point; instantiate(cfg.algorithm); environment **kwargs
  eval.py                   — evaluation entry point; same algorithm instantiation
  train_vae.py              — offline VAE pretraining entry point for the MFEC "vae" encoder
                              (not an Algorithm/Trainer — standalone Hydra script)
  networks.py               — network factories: make_mlp_q_net, NatureDQN,
                              make_mlp_ddpg_actor, make_mlp_ddpg_critic,
                              make_mlp_a2c_actor, make_mlp_a2c_value;
                              NEC embedding networks — NECEmbeddingNetwork
                              (contract Protocol), NatureEmbedding (CNN trunk,
                              no Q-head; default), DINOv2Embedding / CLIPEmbedding /
                              MAEEmbedding (finetunable ViT + channel adapter +
                              param_groups; the last two import their optional
                              backbone package lazily — see "Adding a new NEC
                              embedding network")
  algorithms/
    base.py                 — BaseAlgorithm ABC; TrainingState and CollectorConfig dataclasses
    dqn.py                  — DQNAlgorithm; replay/network factories (defaults + setup contract)
    ddpg.py                 — DDPGAlgorithm; actor/critic/replay/noise factories
    a2c.py                  — A2CAlgorithm; on-policy actor/critic with GAE + A2CLoss
    mfec.py                 — MFECAlgorithm; QEC memory (LRU eviction), QECPolicy (pluggable
                              encoder), MC returns
    nec.py                  — NECAlgorithm; DND class, DNDPolicy, N-step returns, dual updates
  encoders/                 — pluggable MFEC state embeddings (see "Encoders (MFEC)" below)
    base.py                 — Encoder contract: embed() / state() / load_state()
    factory.py               — make_encoder(name, ...)
    random_projectins.py    — RandomProjectionEncoder (default; filename has a typo, kept as-is)
    vae_encoder.py           — VAEEncoder (frozen ConvVAE, embeds mean ++ log-std)
    dino_v2_encoder.py       — DINOv2Encoder (frozen ViT, CLS token, ImageNet stats)
    resnet_encoder.py        — ResNetEncoder (frozen ImageNet ResNet, pooled features)
    clip_encoder.py          — CLIPEncoder (frozen CLIP vision tower, projected + L2;
                              needs the OPTIONAL open_clip_torch package, imported lazily)
    mae_encoder.py           — MAEEncoder (frozen MAE ViT, mean-pooled PATCH tokens, NOT
                              L2-normalised; needs the OPTIONAL timm package, imported
                              lazily.  The only non-similarity pretraining objective)
  models/
    conv_vae.py              — ConvVAE (Blundell et al. 2016 App. D architecture);
                              vae_loss/gaussian_nll/kl_diag_gaussian used by train_vae.py
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: gymnasium + transforms list + gym_kwargs/gym_backend
                              + `seed` (one derived stream per ParallelEnv worker; see
                              "Environment seeding" below)
  trainers/
    BaseTrainer.py          — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    StepTrainer.py          — StepTrainer (Collector-driven loop)
  callbacks/                — ProgressCallback, CheckpointCallback, EvalCallback,
                              WandBLogger, TensorBoardLogger
  utils/                    — device resolution, seeding (seed_everything + derive_seed),
                              callback builders
configs/
  algorithm/dqn.yaml        — DQN HPs (CartPole defaults); _partial_ replay_buffer + network
  algorithm/dqn_atari.yaml  — DQN HPs (Atari/NatureDQN defaults; pixel obs)
  algorithm/ddpg.yaml       — DDPG HPs (HalfCheetah defaults); _partial_ actor/critic/noise
  algorithm/a2c.yaml        — A2C HPs (HalfCheetah/MuJoCo defaults); _partial_ actor/value
  algorithm/mfec_atari.yaml — MFEC HPs (Blundell et al. 2016 §4.1 Atari defaults: buffer_size=1M,
                              k=11, state_dim=64, gamma=1.0, constant eps=0.005;
                              encoder_name=random_projection, vae_checkpoint=null,
                              seed=${trainer.seed} — see "Environment seeding").
                              NOTE §4.2's Labyrinth settings differ (k=50, gamma=0.99)
  algorithm/nec.yaml        — NEC HPs (base defaults); defaults-lists the
                              embedding_network config group + _partial_ replay_buffer
  algorithm/nec_atari.yaml  — NEC HPs (Atari defaults per Pritzel et al. 2017 §4;
                              the paper has NO hyperparameter table — see "NEC" below
                              for which values it states vs. swept-and-unreported)
  algorithm/embedding_network/  — NEC encoder config group (swap with
                              `algorithm/embedding_network=<name>`)
    nature.yaml             — NatureDQN trunk + dense layer (DEFAULT; the paper's net)
    dinov2_finetune.yaml    — finetunable DINOv2 ViT (weights_path required;
                              image_size is the throughput knob)
    clip_finetune.yaml      — finetunable CLIP vision tower (needs the optional
                              `open_clip_torch` extra; QuickGELU pairing and
                              patch-size divisibility are both hard errors)
    mae_finetune.yaml       — finetunable MAE ViT-B/16 (needs the optional `timm`
                              extra; mean-pooled PATCH tokens, image_size=112 to
                              match the CLIP arm's token count, divisibility by 16
                              is a hard error)
  environment/cartpole.yaml — env kwargs (name, transforms)
  environment/pong_train.yaml     — Atari Pong (training transforms incl. EndOfLife + Sign + VecNorm; DQN only)
  environment/pong_mfec_train.yaml — Atari Pong for MFEC (same stack WITHOUT VecNorm; see note below)
  environment/pong_eval.yaml      — Atari Pong (eval transforms; drops EndOfLife + Sign + VecNorm)
  environment/breakout_train.yaml — Atari Breakout (training transforms; no VecNorm — MFEC only)
  environment/breakout_eval.yaml  — Atari Breakout (eval transforms)
  environment/qbert_train.yaml    — ALE/Qbert-v5 (training transforms; no VecNorm — MFEC only)
  environment/qbert_eval.yaml     — ALE/Qbert-v5 (eval transforms)
  environment/mspacman_train.yaml — ALE/MsPacman-v5 (training transforms; no VecNorm — MFEC only)
  environment/mspacman_eval.yaml  — ALE/MsPacman-v5 (eval transforms)
  environment/mspacman_train_singleframe.yaml — same, minus CatFrames (paper-exact VAE encoder input)
  environment/mspacman_eval_singleframe.yaml  — eval counterpart, no CatFrames
  environment/atari_mfec_train.yaml — ALE/MsPacman-v5, paper-faithful MFEC stack: single frame,
                              no SignTransform, repeat_action_probability=0.0 (see "MFEC on Atari" below)
  environment/atari_mfec_eval.yaml  — eval counterpart (identical by design; nothing to strip)
  environment/mspacman_mfec_train_dinov2.yaml — the atari_mfec_train stack with GrayScale/Resize
                              dropped (DINOv2 needs 3 channels and resizes internally); task
                              settings held identical so encoder is the only variable
  environment/mspacman_mfec_eval_dinov2.yaml  — eval counterpart (identical by design)
  environment/atari_mfec_train_rgb.yaml — the generalised version of the two above, for ANY
                              RGB PVM (dinov2, resnet, ...); byte-equivalent to the _dinov2
                              pair, which is kept only so existing runs stay reproducible.
                              Prefer this one for new encoder arms.
  environment/atari_mfec_eval_rgb.yaml  — eval counterpart (identical by design)
  environment/mspacman_nec_train.yaml — ALE/MsPacman-v5 for NEC: action-repeat 4, NO
                              SignTransform (paper §4 names Ms. Pac-Man as a game where
                              NEC's lack of reward clipping is what produces the result)
  environment/mspacman_nec_eval.yaml  — eval counterpart (drops EndOfLife)
  environment/qbert_nec_train.yaml — ALE/Qbert-v5 for NEC: the mspacman_nec_train stack
                              with the game swapped. NO SignTransform (Q*bert's graded
                              25/100/500/1000 rewards would be flattened by it), sticky
                              actions off, 27,000-step episode cap. NOT the same as
                              qbert_train.yaml, which clips, caps at 4,500 and leaves
                              v5's 0.25 sticky actions on
  environment/qbert_nec_eval.yaml  — eval counterpart (drops EndOfLife)
  environment/frostbite_nec_train.yaml — ALE/Frostbite-v5 for NEC: same stack again.
                              Paper §4 names Frostbite in the no-clipping list, and the
                              game is the clearest case (an ice floe pays 10, an igloo
                              several hundred plus time bonus). 18 actions. There is no
                              clipped `frostbite_train.yaml` — this game reaches the repo
                              only through the episodic-control arms
  environment/frostbite_nec_eval.yaml  — eval counterpart (drops EndOfLife)
  environment/hero_nec_train.yaml — ALE/Hero-v5 for NEC: same idea, NO SignTransform
                              (paper §4 names H.E.R.O. in the same list). Use this rather
                              than hero_train.yaml for any NEC H.E.R.O. run.
  environment/hero_train.yaml     — ALE/Hero-v5 with SignTransform (MFEC / DQN)
  environment/hero_eval.yaml      — ALE/Hero-v5 (eval transforms)
  environment/halfcheetah.yaml — HalfCheetah-v4 (DoubleToFloat + InitTracker)
  experiment/dqn/cartpole.yaml — composed CartPole experiment
  experiment/dqn/pong.yaml     — composed Atari Pong experiment
  experiment/ddpg/halfcheetah.yaml — composed DDPG HalfCheetah experiment
  experiment/a2c/halfcheetah.yaml — composed A2C HalfCheetah experiment
  experiment/mfec/pong.yaml    — MFEC on Pong (40M frames, num_envs=16)
  experiment/mfec/breakout.yaml — MFEC on Breakout (1M frames)
  experiment/mfec/qbert.yaml   — MFEC on Q*Bert (40M frames, num_envs=16)
  experiment/mfec/rp_gray.yaml — the PAPER BASELINE of the encoder ablation: random
                              projection over a single 84x84 grayscale frame, Blundell et al.
                              §3's exact phi.  100k decisions = 400k emulator frames, num_envs=4.
                              Game-generic: pass `game=Frostbite` (or Qbert, MsPacman, ...).
                              Carries the CANONICAL buffer_size comment — the other five arms
                              point back at it.  See "The encoder ablation" for those arms;
                              every non-phi knob is held equal across them AND across games.
  experiment/mfec/vae.yaml — same env pair and budget, encoder_name=vae (the paper's
                              OTHER phi, §3).  vae_checkpoint is required, no default.
                              Uses atari_mfec_* — NOT mspacman_*_singleframe, which is a
                              DQN-style stack whose SignTransform would make its
                              episode_reward a pellet count rather than a game score.
  experiment/mfec/dinov2.yaml — same budget, encoder_name=dinov2 (frozen ViT-S/14,
                              state_dim=384) on the shared atari_mfec_*_rgb env pair.
                              dinov2_weights is required and has no usable default — the
                              checked-in path is cluster-local.  NOTE: this config used the
                              DQN-style mspacman_train_dinov2 env pair until Aug 2026; runs
                              logged before that carry reward clipping, sticky actions and a
                              4,500-step cap, and are NOT comparable to mspacman.yaml.
  experiment/mfec/resnet.yaml — same, with encoder_name=resnet (frozen ImageNet
                              resnet18, state_dim=512) + the atari_mfec_*_rgb env pair.
                              resnet_weights_path may be null (torchvision downloads
                              IMAGENET1K_V1); set a path on an offline cluster.
  experiment/mfec/rp_rgb.yaml — the ENCODER CONTROL: random projection over the
                              same RGB observations the PVM arms get, state_dim=64.  Without
                              it, PVM-vs-baseline is confounded with grayscale-vs-RGB.
  experiment/mfec/clip.yaml — same, with encoder_name=clip (frozen CLIP ViT-B-32
                              vision tower, projected + L2-normalised, state_dim=512) + the
                              atari_mfec_*_rgb env pair.  Needs the optional
                              open_clip_torch package; clip_weights_path may be null
                              (open_clip downloads), set a path on an offline cluster.
  experiment/mfec/mae.yaml — same, with encoder_name=mae (frozen MAE ViT-B/16,
                              mean-pooled patch tokens, state_dim=768) + the
                              atari_mfec_*_rgb env pair.  THE ARM THAT MAKES THE
                              ABLATION A STUDY: the only phi whose pretraining
                              objective is not a similarity objective (masked pixel
                              reconstruction), so it is the only one that varies the
                              causal variable the other three PVMs hold fixed.  Needs
                              the optional timm package; mae_weights_path may be null
                              (timm pulls it from the HuggingFace hub), set a path on
                              an offline cluster.  Slowest arm (197 tokens vs the CLIP
                              arm's 50) and widest phi (768), so also the largest eager
                              QEC allocation — see "QEC memory is sized by state_dim".
  experiment/nec/pong.yaml     — NEC on Pong (100k agent steps = 400k raw frames, the shared
                              probe budget; num_envs=8); keeps the clipped env — Pong's
                              rewards are already in [-1, 1] so SignTransform is a no-op
                              there, not a deviation.  Carries NO `algorithm:` block, so it
                              inherits nec_atari.yaml wholesale (num_updates: 400, i.e.
                              paper §4's rate, unlike the twelve ablation arms)
  experiment/nec/hero.yaml     — NEC on H.E.R.O. (100k agent steps = 400k raw frames;
                              num_envs=8); uses hero_nec_train (unclipped).  Same
                              no-`algorithm:`-block inheritance as nec/pong.yaml
  experiment/nec/mspacman.yaml — NEC on Ms. Pac-Man (100k agent steps = 400k raw frames,
                              num_envs=8); uses mspacman_nec_train (unclipped).  The
                              CANONICAL ablation arm: the measurements behind num_updates,
                              eps_end, annealing_frames, init_random_frames, eval_eps and
                              num_envs live here and the other eleven point back at it
  experiment/nec/mspacman_dinov2.yaml — same task and env pair, with the embedding_network
                              group swapped to dinov2_finetune (finetuned ViT-S/14) and
                              run.encoder=dinov2 so the run dir does not collide.  The
                              PVM arm of the NEC encoder ablation against mspacman.yaml.
                              weights_path is required and the checked-in path is
                              cluster-local.  NOTE it keeps the 4x84x84 NEC env, unlike
                              MFEC's DINOv2 arm which needs an RGB env pair — the
                              finetuned adapter handles channels, a frozen ViT cannot
  experiment/nec/mspacman_clip.yaml — same task and env pair, with the
                              embedding_network group swapped to clip_finetune
                              (finetuned CLIP ViT-B-32) and run.encoder=clip.  The
                              contrastive-PVM arm.  Needs the optional
                              `open_clip_torch` extra; weights_path defaults to null
                              (open_clip downloads the `openai` tag — set a local
                              path on an offline node)
  experiment/nec/mspacman_mae.yaml — same task and env pair, with the
                              embedding_network group swapped to mae_finetune
                              (finetuned MAE ViT-B/16) and run.encoder=mae.  The
                              RECONSTRUCTION arm, and the one that makes the NEC
                              ablation a study rather than a bake-off: the only
                              embedding network whose pretraining objective is not a
                              similarity objective.  Needs the optional `timm` extra;
                              weights_path defaults to null (timm pulls the
                              `vit_base_patch16_224.mae` tag from the HuggingFace hub —
                              set a local path on an offline node).  Runs at
                              image_size=112 (7x7=49 tokens), matching the CLIP arm's
                              token count so the two PVM arms differ in objective and
                              not in compute
  experiment/nec/qbert.yaml, qbert_dinov2.yaml, qbert_clip.yaml, qbert_mae.yaml — the
                              Q*bert arms of the same four-encoder ablation, on
                              qbert_nec_train/eval.  6 actions, the cheapest of the
                              three games per gradient step (cost is
                              num_updates x |A| exact kNN scans)
  experiment/nec/frostbite.yaml, frostbite_dinov2.yaml, frostbite_clip.yaml,
  frostbite_mae.yaml — the
                              Frostbite arms, on frostbite_nec_train/eval. 18 actions,
                              the most expensive of the three; the ViT arms here are the
                              heaviest runs in the study
                              ALL TWELVE ablation files repeat their shared settings
                              verbatim (Hydra experiment configs do not compose with
                              each other) — a change that belongs to the comparison has
                              to land in all twelve
  logger/{wandb,tensorboard}.yaml
  paths/default.yaml
  train.yaml, eval.yaml, train_vae.yaml
tests/
  test_env_seeding.py       — ParallelEnv workers get reproducible, non-colliding streams;
                              a num_envs=1 env must not re-seed the parent process
  test_smoke.py             — DQN-on-CartPole, DQN-on-Pong, DDPG-on-HalfCheetah, A2C-on-HalfCheetah, MFEC-on-Pong, NEC-on-Pong, NEC-on-Pong-with-DINOv2, NEC-on-Pong-with-CLIP, NEC-on-Pong-with-MAE smoke tests
  test_mfec_encoder_refactor.py — encoder-abstraction transparency: setup() wiring, embed()
                              shape/determinism, forward(), deepcopy sharing, checkpoint round-trip
  test_mfec_estimator_gap.py — eval_eps defaults to 0.0 (greedy eval) and QEC exposes the
                              exact-vs-kNN estimator gap as eval/exact_minus_knn_value
  test_clip_encoder.py      — CLIP PVM encoder: lazy open_clip import, projected+L2 output,
                              CLIP (not ImageNet) stats, no centre crop, determinism,
                              state round-trip, and an MFEC end-to-end run on the RGB env.
                              Runs against a stub open_clip, so CI needs no package.
  test_nec_optimistic_tiebreak.py — the NEC half of "Optimistic init must be tie-broken
                              RANDOMLY": an empty DND must not make argmax play one fixed
                              action, a partially-populated one must spread over the
                              still-optimistic actions, the jitter must survive float32
                              rounding at 1e9 (ULP 64.0) and stay above the trainer's 1e8
                              sentinel threshold, and FINITE estimates must stay
                              bit-identical across calls (eval determinism)
  test_nec_ablation_parity.py — the twelve encoder-ablation arms differ ONLY in the
                              encoder: every algorithm- and trainer-side knob is compared
                              arm-vs-nature-baseline and across games (this is what
                              catches a num_envs or num_eval_episodes drift), plus the
                              exploration schedule is checked to fit inside
                              trainer.total_frames — init_random_frames a whole number of
                              collector batches, annealing_frames <= 20% of the budget,
                              warm-up shorter than the anneal
  test_nec_embedding_network.py — NEC embedding-network config group: shape/dtype contract,
                              gradient flow (proves the encoder is genuinely trainable),
                              Hydra-composition architecture regression, config-swap
                              setup()+step() end-to-end, DINOv2 group selection (stub
                              backbone), MAE group selection (stub timm)
  test_nec_dinov2_finetune.py — DINOv2Embedding itself: channel-adapter init, param_groups
                              and setup()'s use of them, finetuning through step(),
                              gradients reaching the backbone, checkpoint round-trip.
                              Second tier (NEC_DINOV2_REAL=1, NEC_DINOV2_REPO_DIR=... if
                              offline) builds the genuine dinov2_vits14
  test_nec_clip_finetune.py — CLIPEmbedding: lazy open_clip import (AST-checked),
                              QuickGELU guard, patch-size divisibility guard,
                              BatchNorm warning, text-tower removal, CLIP-vs-ImageNet
                              stats, param groups, finetuning through step(),
                              checkpoint round-trip.  Second tier (NEC_CLIP_REAL=1)
                              builds the genuine ViT-B-32-quickgelu; third
                              (CLIP_WEIGHTS=...) loads the real OpenAI checkpoint
  test_nec_mae_finetune.py  — MAEEmbedding: lazy timm import (AST-checked), PATCH-token
                              pooling (poisoned-prefix stub) vs cls, prefix count read
                              not assumed, patch-size divisibility guard, ImageNet-vs-CLIP
                              stats, whole-frame resize, adapter init, param groups,
                              finetuning through step(), checkpoint round-trip, and the
                              learning-knob parity of all three _mae experiment configs
                              against their nature arms.  Second tier (NEC_MAE_REAL=1)
                              builds the genuine ViT-B/16 architecture with
                              pretrained=False; third (MAE_WEIGHTS=... or
                              NEC_MAE_DOWNLOAD=1) loads real weights and pins that timm
                              resamples pos_embed at image_size=112 from BOTH the hub
                              and a local file
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
| `mspacman_train.yaml` | ✗ | MFEC Ms. Pac-Man |
| `mspacman_train_singleframe.yaml` | ✗ | MFEC Ms. Pac-Man + VAE encoder (paper-exact) |
| `atari_mfec_train.yaml` | ✗ | MFEC Ms. Pac-Man (paper-faithful; see "MFEC on Atari" below) |
| `mspacman_mfec_train_dinov2.yaml` | ✗ | MFEC Ms. Pac-Man + frozen DINOv2 encoder (paper-faithful) |
| `mspacman_nec_train.yaml` | ✗ | NEC Ms. Pac-Man (also: no SignTransform) |
| `qbert_nec_train.yaml` | ✗ | NEC Q*bert (also: no SignTransform) |
| `frostbite_nec_train.yaml` | ✗ | NEC Frostbite (also: no SignTransform) |
| `hero_nec_train.yaml` | ✗ | NEC H.E.R.O. (also: no SignTransform) |

The fixed random projection already compresses 28 k-pixel observations
adequately without online whitening.

## Reward scale for Atari MFEC runs

Most MFEC training configs include `SignTransform` (reward clipped to
`{-1, 0, +1}`), so `train/episode_reward` (from `RewardSum`) counts clipped
reward events — **not** the true game score. For example, "57" on Q*Bert means
~57 positive reward events, not a score of 57 points. Compare against the paper
using `eval/return_mean`, which is computed from the `*_eval.yaml` environment
that drops `SignTransform`. Set `trainer.eval_every_n_steps` to get
`eval/return_mean` logged periodically during training instead of running
`src/eval.py` separately after the fact — see "Periodic in-training
evaluation" above.

The exception is the `atari_mfec_*` pair, which has no `SignTransform` at
all (see below): there `train/episode_reward` and `eval/return_mean` measure
the same thing, so a gap between them is a real train/eval discrepancy rather
than a units mismatch.

## MFEC on Atari — what a DQN-style env config gets wrong

Reusing a DQN Atari transform stack for MFEC quietly breaks the algorithm.
`atari_mfec_train.yaml` / `atari_mfec_eval.yaml` are the corrected
reference pair; `pong`, `breakout` and `qbert` have **not** been migrated yet.

| Setting | DQN-style default | MFEC needs | Why |
|---|---|---|---|
| `repeat_action_probability` | `0.25` (`ALE/*-v5` default; TorchRL does not touch it) | `0.0` | Eq. (1) is max-over-returns and never decreases — footnote 1: "not suited to rational action selection in stochastic environments". Sticky actions also collapse `train/exact_hit_rate`, which §4.1 measures at ~50% on Ms. Pac-Man. |
| `SignTransform` | present | **absent** | MFEC stores raw Monte-Carlo returns, so clipping makes a dot (10 pts) worth as much as a ghost (200–1600) or fruit (100–5000). |
| `CatFrames` | `N=4` | **absent** | §3 embeds a single 84×84 frame (D = 7056; their "28 KBytes per frame" is exactly 84·84·4 bytes). A stack folds history into the state and makes exact re-encounters rarer. |
| `gamma` | `0.99` | `1.0` | §4.1: "The discount rate was set to γ = 1." The 0.99 in §4.2 is the *Labyrinth* setting. |
| `eps_end` | `0.05` | `0.005` | §4.1: "higher exploration rates were not as beneficial". |

`EndOfLifeTransform` is dropped too — MFEC's backward replay (Algorithm 1,
lines 9–11) is defined over whole episodes. Note it was already **inert** in
the DQN-style configs: `_step` *reads* `done_key` (which defaults to `"done"`)
but writes the life-loss signal only to `eol_key` (`"end-of-life"`), never back
to `"done"`. Consuming it means pointing a loss or return computation at
`"end-of-life"`, which nothing here does — MFEC reads `batch["next", "done"]`.
Measured on `mspacman_train` over 3,000 random steps: `end-of-life` fired 18×
while `done` fired 6×, i.e. episodes still ended only at game over. So adding
the transform without also rewiring the consumer buys nothing.

### `frame_skip` is not what it looks like

`GymEnv._build_env` unconditionally does `kwargs["frameskip"] = self.frame_skip`
and sets `wrapper_frame_skip = 1`, so `frame_skip` is **forwarded to ALE** and
*overrides* the `ALE/*-v5` registry default rather than stacking a second
action repeat on top of it. Consequences:

- `frame_skip: 4` gives 4 emulator frames per decision — correct, per §4.1.
- **Omitting** `frame_skip` does not fall back to v5's `frameskip=4`; it sets
  ALE's frameskip to `GymEnv`'s default of `1`. Always set it explicitly.

### Frame units when comparing to the paper

`trainer.total_frames` and the logged step count are agent **decisions**
(`StepTrainer` does `self._step += batch.numel()`). The paper's x-axis is ALE
emulator frames at 4 per decision (§4.1: "An hour of game play corresponds to
approximately 200,000 frames"). So **paper frames = 4 × logged step**: the
shared probe budget of `total_frames: 100_000` is 400k emulator frames, and
`total_frames: 12_500_000` covers Figure 1's full 50M-frame range.

### QEC eviction is LRU, not FIFO

§2: "we limit the size of the table by removing the least recently *updated*
entry". `QEC._key_to_slot[a]` is an `OrderedDict` kept in
least-recently-updated-first order, so it doubles as the LRU queue:
`popitem(last=False)` picks the victim and `QEC.touch()` refreshes an entry
after an Eq. (1) max-update. A kNN **read** deliberately does not refresh
recency. A FIFO ring buffer would evict the oldest *insertions*, which on
Atari are exactly the early-level states re-visited every episode.

### Environment seeding

**`seed_everything(cfg.trainer.seed)` in `src/train.py` seeds the parent
process only, and that is not enough.** `ParallelEnv` uses
`mp_start_method="spawn"`, so every env worker is a fresh interpreter whose
`torch`/`numpy`/`random` RNGs come from OS entropy. Before this was fixed,
nothing called `set_seed` anywhere, and two runs after an identical
`seed_everything(42)` produced different per-worker start states — measured.
A five-seed sweep (`seed: 42,43,44,45,46`) was therefore **not** five
controlled seeds; only the random-projection matrix varied, and in
`mfec/rp_gray.yaml` (which left `algorithm.seed: null`) not even that.

Multi-seed runs no longer need a dedicated config — the `*_5seed.yaml` files
were deleted in `270e98a`. `configs/algorithm/mfec_atari.yaml` sets
`seed: ${trainer.seed}`, so sweeping the trainer seed varies the encoder seed
with it, and `run.name` carries the seed so output dirs and W&B runs do not
collide:

```shell
python src/train.py -m experiment=mfec/rp_gray trainer.seed=42,43,44,45,46
```

`run.group` (= `run.name` without the seed) is passed to the W&B logger, so the
sweep lands as one group the UI can average with a standard-error band.

Why it is fatal rather than cosmetic: with `repeat_action_probability=0.0` an
ALE reset is fully deterministic, so `NoopResetEnv`'s `torch.randint` draw is
**the only** source of episode-start diversity — and it runs inside the worker,
where `Transform` has no `_set_seed` hook for a parent-side `set_seed` to reach.

The fix, in `src/environments/factory.py`:

- `make_env(..., seed=<master>)` builds **one `env_fn` per worker** (a list, not
  one shared callable) — that partial is the only code that runs inside the
  spawned process.
- Each worker calls `seed_everything(derive_seed(master, i))` *before*
  constructing anything, then `env.set_seed(...)` for the emulator.
- `seed_process` is `num_envs > 1` only. A `num_envs=1` env is built in the
  parent, and `BaseTrainer.evaluate()` builds one **per eval interval** — so
  re-seeding there would reset the trainer's exploration stream every 200 k
  steps and make training exploration periodic.
- `BaseTrainer` derives two streams: `_SEED_STREAM_TRAIN` for the collector's
  envs and `_SEED_STREAM_EVAL` **keyed on `self._step`** for the eval env, so
  successive evaluations do not replay identical episodes.

`derive_seed` (`src/utils/seeding.py`) hashes with blake2b rather than doing
`base + i`, for two reasons: `base + i` aliases across runs (seeds `42..46`,
4 workers → `(44, worker 1)` and `(45, worker 0)` both give 45, so two
"independent" runs replay each other), and Python's `hash()` is salted per
interpreter, which is exactly the property that breaks in a spawned worker.

Guarded by `tests/test_env_seeding.py`, including a negative control: on the
pre-fix path (`seed=None`) two identical constructions must *not* reproduce.

### Eval must preprocess observations on the training env's device

**This is the bug that made `eval/return_mean` sit at random-play level (~350
on Ms. Pac-Man) while `train/episode_reward` climbed past 3 000.**

`env_worker_device(num_envs, device)` (`src/environments/factory.py`) is the
device an env — and therefore its whole `ToTensorImage → GrayScale → Resize`
stack — actually runs on. A CUDA context cannot survive the spawn into a
`ParallelEnv` worker, so **`num_envs > 1` forces the transforms onto CPU** no
matter what `trainer.accelerator` says; the collector only moves the finished
observation to the GPU afterwards. `BaseTrainer.evaluate()` used to build its
own `num_envs=1` eval env on `str(self.device)`, i.e. on **CUDA** — so on a GPU
box the identical ALE frame went through CPU kernels during training and CUDA
kernels during evaluation.

Bilinear-antialias `Resize` is not bit-identical across the two, and MFEC keys
its memory on `round(embedding * key_scale)`. Measured (64-d random projection
of an 84×84 frame): **1e-7 of drift per pixel already changes ~40 % of hash
keys and 1e-6 changes all of them.** Evaluation then answers every query with a
k-neighbour mean, which is near-constant across actions, so the argmax is
noise. Measured on Ms. Pac-Man after 60 k decisions (train 1 858):

| drift injected into eval observations | `eval/return_mean` | exact-hit rate |
|---|---|---|
| none (train and eval on the same device) | **2 076** | 0.13 |
| 1e-7 (one float32 ULP) | 390 | 0.05 |
| 1e-6 | 292 | 0.00 |
| 1e-5 | 348 | 0.00 |

So: `evaluate()` builds the eval env on `self._env_device` (set in
`BaseTrainer.setup()` from `env_worker_device`) and moves the tensordict onto
`self.device` for the policy call and back for the env step — the same device
flow the collector uses. Do **not** "simplify" that back to
`device=str(self.device)`. Guarded by
`tests/test_mfec_eval_lookup.py::test_evaluate_builds_eval_env_on_the_env_device_not_the_accelerator`.

The same reasoning applies to any future consumer that has to reproduce the
collector's observation bytes outside the collector.

**Standalone `src/eval.py` must be given the training run's `trainer.num_envs`
and `trainer.accelerator`**, because that pair is what `_env_device` is derived
from and it is not stored in the checkpoint. `configs/eval.yaml` defaults to
`num_envs: 1, accelerator: cpu`; evaluating a checkpoint trained with
`num_envs: 4, accelerator: gpu` from those defaults happens to work (both give
`cpu`), but adding `trainer.accelerator=gpu` without `trainer.num_envs=4`
re-creates the mismatch. `eval/exact_hit_rate` near 0 on a non-empty QEC is the
signature.

### Near-exact kNN hits: recompute the distance, never trust `cdist`

`QEC.estimate_all` treats a kNN top-1 that is numerically indistinguishable
from the query as an exact re-encounter (Eq. 2 case 1) — the last line of
defence when a hash key has drifted. Two things this must get right:

1. **`torch.cdist` cannot be used for the test.** It takes the
   `x² + y² − 2xy` shortcut as soon as either side exceeds 25 rows (always,
   here), which cancels catastrophically at short range: measured **4.2e-4 for
   a pair whose true distance is 2.8e-6**. Ranking survives that error; a
   near-zero threshold does not. The old `dists[:, 0] < 1e-5` was *below
   `cdist`'s own noise floor* and therefore fired essentially at random.
   `estimate_all` recomputes the winner's distance directly with
   `linalg.vector_norm(q - top1)`, which is O(miss × d) and exact.
2. **The tolerance is relative** (`QEC._NEAR_EXACT_RTOL`, 3e-5 × `(1 + ‖q‖)`).
   The comparison is an L2 distance in embedding space and that space is the
   encoder's: `‖φ(o)‖ ≈ 2` for a 64-d random projection, 10–50× that for
   DINOv2/ResNet. An absolute threshold is meaningless across encoders.

   The margin is tighter than it looks, so do not widen this casually. Measured
   over 2.25 M frame pairs from a random rollout on Ms. Pac-Man (random
   projection, `state_dim=64`): the closest pair of genuinely *different*
   frames is **1.1e-3** apart — not the ~0.14 the median suggests — while
   1e-6 of per-pixel float noise moves an embedding by 7.8e-6. `≈9e-5` sits
   ~12× under the first and ~11× over the second. Err low: a false merge is a
   silently wrong Q value, a missed rescue costs one lookup.

   Drift above ~1e-5 per pixel is deliberately **not** covered — no CPU/CUDA
   kernel pair differs by that much, and a regime that does is a bug to fix at
   the source, not to absorb here. Past the tolerance the lookup degrades to
   the k-neighbour mean, which is at least unbiased; measured, that still plays
   at ~1100 rather than collapsing, because false merges no longer happen.

Guarded by `tests/test_mfec_eval_lookup.py`.

### `eval/exact_hit_rate` — the metric that makes this failure visible

`BaseAlgorithm` has two optional hooks, `reset_eval_metrics()` and
`eval_metrics()`, which `BaseTrainer.evaluate()` calls around the rollout and
merges into the `eval/*` dict (default: no-ops returning `{}`). MFEC uses them
to report `eval/exact_hit_rate`, `eval/memory_hit_rate` and
`eval/exact_minus_knn_value` (see "Two estimators, one argmax") from `QEC`'s
read-path counters. A collapsed `eval/return_mean` with a hit rate near 0 means
the memory is never being reached — a preprocessing/observation problem, not a
bad policy. `evaluate()` also reports `eval/episode_length`, which separates
"plays badly" from "episode ends early".

The eval denominator is every *(state, action)* pair the policy asks about
(all `|A|` per frame); `train/exact_hit_rate` only counts the single action
actually taken, so the two differ by roughly `|A|` and are not comparable
point for point. §4.1's "~50 % for Ms. PAC-MAN" is the train-side quantity.

`train/exact_hit_rate` is now **omitted** on batches that closed no episode
rather than reported as `0.0`. A Ms. Pac-Man episode is ~600 decisions against
`frames_per_batch: 1024` over 4 envs, so most batches make no QEC queries at
all, and the fabricated zeros made the logged series oscillate between 0 and
the real rate. This matches `_IntervalStats`' convention in `StepTrainer`.

### Exact-match keys must be invariant to batch shape

`RandomProjectionEncoder.embed` accumulates the projection in **float64**.
Training embeds `num_envs` rows per call while `BaseTrainer.evaluate` builds a
single env and embeds 1 row; BLAS picks a different reduction order for those
shapes, and in float32 the resulting ~1e-6 error is the same order as the
1e-5 quantisation step at the default `key_scale`. Measured on CPU: 15 of 16
observations hashed differently at batch 1 vs batch 16 in float32, and 0 of 16
in float64. Without this, evaluation silently never takes the exact-match
path. Guarded by `tests/test_mfec_qec_dict.py::test_embed_key_invariant_to_batch_shape`.

## Encoders (MFEC)

MFEC's `φ` (state embedding) is a pluggable `Encoder` (`src/encoders/base.py`),
not hardcoded into `QECPolicy`. `MFECAlgorithm.setup()` builds it via
`make_encoder(encoder_name, ...)` (`src/encoders/factory.py`).
`QECPolicy.__deepcopy__` shares the encoder by reference (same object the
collector's policy uses), same as `qec`.

Available: `random_projection` (default), `vae`, `dinov2`, `resnet`
(frozen ImageNet ResNet, `src/encoders/resnet_encoder.py`; `state_dim` comes
off `fc.in_features`, `fc` becomes `nn.Identity`, and `weights_path=None` is
legal — torchvision downloads `IMAGENET1K_V1`), `clip`, and `mae`. The RGB
backbones (`dinov2`, `resnet`, `clip`, `mae`) share the `atari_mfec_train_rgb` /
`atari_mfec_eval_rgb` env pair.

### `clip` — an OPTIONAL dependency, imported lazily

`CLIPEncoder` (`src/encoders/clip_encoder.py`) needs `open_clip_torch`, which
is deliberately kept out of the base dependencies — it is a
`[project.optional-dependencies]` extra, installed with **`uv sync --extra
clip`**. `factory.py` imports `CLIPEncoder` at module scope, so `clip_encoder.py`
must keep `import open_clip` **inside `__init__`** — hoisting it to module
scope makes the missing package break every MFEC run, `random_projection`
included. Pinned by
`tests/test_encoder_factory.py::test_importing_the_factory_does_not_require_open_clip`.
The whole test file for it runs with a stub `open_clip` injected into
`sys.modules`, so CI never needs the package.

Three choices in it are load-bearing and follow from MFEC using φ as a
**metric**, not as network input:

1. **Projected embedding, not the pre-projection token.** `model.visual` is
   what open_clip's `encode_image` calls, and it includes the projection into
   the joint image-text space (512-d for ViT-B-32/ViT-B-16). That projection
   is where the contrastive loss lives. `timm`'s CLIP entries expose the 768-d
   pre-projection pooled token instead — do not swap to them without knowing
   this. Keeping only `visual` also drops the ~63 M-param text tower.
2. **L2 normalisation, on by default.** `‖a−b‖² = 2 − 2·cos(a,b)` on the unit
   sphere, so MFEC's Euclidean kNN becomes exactly cosine kNN — the metric
   CLIP was trained under. It also pins `‖φ‖` to 1.0, making the QEC's
   *relative* near-exact tolerance `3e-5·(1+1) = 6e-5`, tighter than the random
   projection's 9.8e-5 (i.e. conservative — unlike DINOv2, whose `‖φ‖ ≈ 20`
   widens it ~6×). `clip_normalize: false` ablates it.
3. **CLIP's own mean/std and no centre crop.** The constants differ from
   ImageNet's, and stock CLIP inference uses the former; `ResNetEncoder` and
   `DINOv2Encoder` correctly use ImageNet's because *their* backbones were
   trained with them. The centre crop in CLIP's reference pipeline would cut a
   210×160 Atari frame down to its middle and discard the maze edges and the
   score row, so the whole frame is resized instead — losing pixels is worse
   than distorting them when the embedding is a memory key.

Model names use open_clip's hyphenated spelling (`ViT-B-32`), not OpenAI's
`ViT-B/32`. Like `resnet` and unlike `dinov2`, `clip_weights_path: null` is
legal — open_clip then resolves `clip_pretrained_tag` over the network, so set
a path on an offline cluster.

**QuickGELU pairing is checked, and the check is load-bearing.** OpenAI's CLIP
was trained with QuickGELU activations; open_clip's plain `ViT-B-32` config
uses standard GELU. open_clip loads the mismatch anyway and only emits a
`UserWarning` — trivially lost in a multi-day log — after which the run
produces a *subtly wrong embedding geometry*, i.e. corrupts exactly the
property the CLIP arm exists to measure. `CLIPEncoder.__init__` raises instead.
`openai` → `ViT-B-32-quickgelu`; `laion2b_*` / `datacomp_*` → plain
`ViT-B-32`. The check keys off `clip_pretrained_tag` even when
`clip_weights_path` is set, because the tag is the checkpoint's provenance
declaration and the local file does not make a wrong architecture right.
Guarded by `tests/test_clip_encoder.py`.

Same float32-ViT key-stability caveat as DINOv2 — and it **fails**; see below.

### `mae` — the second OPTIONAL dependency, also imported lazily

`MAEEncoder` (`src/encoders/mae_encoder.py`) needs `timm`, kept out of the base
dependencies exactly like `open_clip_torch`: it is the `mae` extra, installed
with **`uv sync --extra mae`**. `factory.py` imports `MAEEncoder` at module
scope, so `mae_encoder.py` must keep `import timm` **inside `__init__`** — the
same rule, the same failure mode (every MFEC run down, `random_projection`
included), pinned by
`tests/test_encoder_factory.py::test_importing_the_factory_does_not_require_timm`.
`tests/test_mae_encoder.py` runs against a stub `timm` in `sys.modules`, so CI
never needs the package.

The extra now covers **two** arms: this frozen MFEC encoder and NEC's finetuned
`algorithm/embedding_network=mae_finetune` (`src.networks.MAEEmbedding`, see
"3. Embedding networks"). The lazy-import rule is stricter on the NEC side —
`src/networks.py` is imported by every DQN/DDPG/A2C config, so a module-scope
`import timm` there breaks the whole repo rather than only MFEC. Pinned by
`tests/test_nec_mae_finetune.py::test_timm_is_not_a_module_level_import_of_networks`.

**Why the arm exists.** `resnet` (ImageNet labels), `dinov2` (self-distillation)
and `clip` (contrastive) all optimise for semantic discriminability. MAE's
objective is masked *pixel reconstruction* and never compares two images, so it
is the only arm that varies the kind of objective rather than the backbone — the
one property MFEC's use of φ as a *metric* is actually about. A low MAE score is
the hypothesis, not a bug.

Four choices are load-bearing:

1. **Pooling is done here, not by timm, and the default is not timm's.** MAE's
   CLS token is never directly supervised — the reconstruction loss lives on the
   patch tokens — so `mae_pooling: mean` averages the **patch** tokens, dropping
   the model's `num_prefix_tokens` prefix entries (read, never assumed to be 1:
   a ViT with register tokens reports more). `cls` ablates the choice. Taking
   CLS by default would have measured our pooling and written it up as a
   property of MAE.
2. **`global_pool` is never passed to `create_model`, and `'avg'` is a trap.**
   timm's default for the `.mae` tag is `global_pool='token'` (the tag's
   `default_cfg` sets `num_classes=0` and says nothing about pooling, so
   `VisionTransformer`'s own default applies) — i.e. CLS. But "fix it with
   `global_pool='avg'`" is worse: that flips `use_fc_norm`, making `self.norm`
   an `nn.Identity` and `self.fc_norm` a **freshly initialised LayerNorm the MAE
   pretrain checkpoint does not contain**, so `forward_features` would return
   un-normalised tokens. Building with the default and pooling by hand keeps the
   pretrained final LayerNorm in the forward path. Both claims are asserted by
   `test_the_timm_default_pooling_is_token_not_avg`, which needs timm installed
   but no download (`pretrained=False`) and is skipped where timm is absent.
3. **No L2 normalisation** — the deliberate difference from `clip`. CLIP
   normalises because cosine is the metric its loss was computed under; MAE has
   no metric objective, and `dinov2`/`resnet` do not normalise either. Leaving
   MAE raw keeps `clip` the *only* arm on the unit sphere, so the other three
   differ only in backbone. ImageNet mean/std and a bilinear resize, for the
   same "hold it constant with dinov2/resnet" reason.
4. **Local weights go through `pretrained_cfg_overlay=dict(file=...)`**, not a
   raw `torch.load` + `load_state_dict` (the `DINOv2Encoder` approach). That
   routes the file through timm's `checkpoint_filter_fn`, which unwraps a
   `{"model": ...}` wrapper, remaps the original `mae_pretrain_vit_base.pth`
   key names and resamples `pos_embed` — a strict raw load rejects the upstream
   MAE release outright. Like `resnet`/`clip` and unlike `dinov2`,
   `mae_weights_path: null` is legal (timm pulls from the HuggingFace hub, so
   set a path on an offline cluster).

Use the `.mae` tag, **not** `.mae_ft_in1k` — the latter is MAE pretraining
followed by supervised ImageNet finetuning, which would quietly make this a
second supervised arm. Costs: 197 tokens against the CLIP arm's 50 at equal
depth and width (≈4x the φ time per frame, and φ is the bottleneck), and the
widest `state_dim` in the study at 768 — see the QEC memory section.

Same float32-ViT key-stability caveat as DINOv2 and CLIP; it has not been
measured on a GPU yet, and `key b/s = 0.000` is the expected result.

### MEASURED: every float32 encoder fails key stability on CUDA

`scripts/encoder_diagnostics.py --device cuda`, 150 Ms. Pac-Man frames:

| encoder | dim | key b/s | key rep | adj AUC | rel contr |
|---|---|---|---|---|---|
| `random_projection` | 64 | **1.000** | 1.000 | 0.955 | 0.627 |
| `dinov2` ViT-S/14 | 384 | **0.000** | 1.000 | 0.991 | 0.697 |
| `resnet` resnet18 | 512 | **0.000** | 1.000 | 0.991 | 0.690 |
| `clip` ViT-B-32-quickgelu | 512 | **0.000** | 1.000 | 0.990 | 0.728 |

`mae` is absent because it was added after this measurement and has not been run
on a GPU yet — do that (`--mae`) before trusting the arm. It is a float32 ViT, so
`key b/s = 0.000` is the expected result, not a finding.

`key rep` is 1.000 everywhere, so the encoders *are* deterministic. What breaks
is **batch-shape** invariance: cuBLAS selects float32 GEMM kernels by batch
size, so `φ(x)` inside a 16-row batch differs from `φ(x)` alone in the last
bits. A key is `round(φ · key_scale)` over `d` coordinates and survives only if
**none** of them lands within the drift of a grid boundary:

```
P(key survives) ≈ (1 - 2 · drift · key_scale) ** d
```

At `d=384`, `drift≈1e-6`, `key_scale=1e5` that is ~1e-18 — hence exactly 0.000.
`RandomProjectionEncoder` escapes only because float64 accumulation puts its
drift at ~1e-15, not because 64 dimensions are special.

**Do not "fix" this by lowering `key_scale`.** The ladder search the script now
prints (`key_scale*`) will report a value that restores 1.000, but for a 384-d
encoder that value is ~10 — a 0.1 grid, coarser than the nearest-neighbour
separation, which merges genuinely distinct states into one key. That is a
silently wrong Q-value, strictly worse than the current behaviour. Nor is
"accumulate in float64" actionable: it works for one matmul, not a ViT.

**What actually happens.** The consequences are:

- **Training is unaffected.** The collector embeds `num_envs` rows on every
  policy call and `step()` reuses those cached embeddings, so every QEC write
  and every train-time query uses the same batch shape. `train/exact_hit_rate`
  is real.
- **Evaluation loses only the O(1) path — *if and only if* convolutions run at
  true FP32.** `estimate_all` falls through to the near-exact rescue, which
  accepts the top-1 neighbour when `‖q − s₁‖ ≤ 3e-5·(1+‖q‖)`. At FP32 the drift
  is ~1e-6 while the nearest *distinct* frame sits at ~30% of the mean pairwise
  distance, so the rescue resolves to the same entry the dict would have found
  and returns the identical stored value, just via kNN instead of a hash lookup.
- **`eval/exact_hit_rate` reads 0 for every PVM arm** and is not comparable
  across encoders. Use `eval/memory_hit_rate`, which counts both paths.

#### CORRECTION: the rescue does NOT survive TF32 — it must be pinned off

The paragraph above was written against an assumed ~1e-6 drift. That is an
**FP32** number, and `torch.backends.cudnn.allow_tf32` **defaults to `True`**:
on Ampere and later every convolution runs at TF32's 10-bit mantissa, unit
roundoff 2⁻¹¹ ≈ **4.9e-4** per operation rather than FP32's 2⁻²⁴ ≈ 6e-8.
(`torch.backends.cuda.matmul.allow_tf32` defaults to `False`, so conv backbones
are hit hardest — but a ViT's patch embedding is a `Conv2d`, so no arm is
exempt. `random_projection` has no convolution and is unaffected.)

Measured against the rescue's own budget, on 400 real Ms. Pac-Man frames with
`resnet18`:

| quantity | value |
|---|---|
| `‖φ(o)‖` | 25.0 |
| rescue budget `3e-5·(1+‖q‖)` | 7.8e-4 in L2 |
| ...per coordinate over `d = 512` | ~3.1e-5 relative |
| **TF32 unit roundoff, one operation** | **4.9e-4 — ~16x over budget** |
| nearest *distinct* frame | 0.307 (400x headroom; false merges were never the risk) |

**Measured consequence on the `mfec/resnet` Ms. Pac-Man run (5 seeds, 100k
decisions):** `eval/memory_hit_rate` identically **0.000** at every eval point
on every seed — dict path *and* rescue dead. All `|A|` Q-estimates per frame
were k-neighbour means over nearly the same neighbourhood, so the argmax was
noise and `eval/return_mean` sat at random play (~400 against MsPacman's 307.3)
while `train/episode_reward` climbed past 2000. `eval/exact_minus_knn_value` is
absent from those runs for the same reason: `value_stats` had zero samples on
the exact side, which is also why the guard below did not catch it.

Note `eval/value_return_corr` read a healthy **0.6–0.85** throughout. It does
not flag this: kNN neighbours are temporally local, so their stored returns
correlate with the realised return-to-go for the trivial reason that both decay
toward episode end. It measures "does the memory know how far into the episode
I am", not "does the memory discriminate between actions".

**Fixed by `src/encoders/factory.pin_fp32_conv_precision()`**, called at the top
of `make_encoder` — which is MFEC's and only MFEC's φ factory, so NEC/DQN/DDPG/
A2C conv throughput is untouched. Applied to every arm, not just `resnet`,
because a per-arm numerics regime would make a between-arm read a comparison of
float precisions. Two costs, both intended: PVM arms get slower (TF32 is ~2x on
conv-bound work and φ is the bottleneck), and **checkpoints do not cross the
change** — a QEC written under TF32 holds keys the FP32 process will never
reproduce, so re-run rather than resume. Guarded by
`tests/test_encoder_factory.py::test_building_any_encoder_pins_fp32_convolutions`
and `::test_the_pin_runs_before_the_encoder_is_constructed`.

**`key b/s = 0.000` in the table above is not a sufficient diagnostic.** At
`d = 512` it reads 0.000 for any drift above ~1e-10, so it cannot distinguish
"1e-6, rescue fine" from "1e-2, rescue dead" — the two regimes that decide
whether a run measures anything. `scripts/encoder_diagnostics.py` reports only
the key rate; the number that matters is the **L2 drift between batch shapes
against `3e-5·(1+‖q‖)`**. Measure that before trusting any new conv-carrying
encoder.

`QEC.value_stats` therefore counts near-exact rescues on the *exact* side.
Keying it off `_lookup_exact` (dict hits only) omitted
`eval/exact_minus_knn_value` entirely for DINOv2/ResNet/CLIP — the encoders the
metric exists to compare. Guarded by
`tests/test_mfec_estimator_gap.py::test_the_gap_survives_an_encoder_with_zero_dict_hits`.

If you ever need the hash path back on a neural encoder, the sound fix is a
bucket-and-verify scheme (coarse key for the O(1) lookup, exact L2 check to
reject collisions), not a finer or coarser `key_scale`. That is a real change
to `QEC` and has not been made.

### Which games MFEC can work on at all — MEASURED

MFEC's whole learning signal is Eq. (1) firing on an **exact state
re-encounter**. Games differ enormously in whether that ever happens, and a
game where it does not is a guaranteed null result no matter how good φ is.
Screen before spending GPU time. Measured (random projection, 15 k frames,
`eps=0.005`; frame repetition from a 2.5 k-frame random rollout):

| game | \|A\| | raw frame repeat | `train/exact_hit_rate` | verdict |
|---|---|---|---|---|
| Ms. Pac-Man | 9 | 22 % | 0.53 | works (reaches ~30x random) |
| Q*bert | 6 | 13 % | 0.60 | works |
| Frostbite | 18 | 22 % | 0.41 | works |
| **Assault** | 7 | **1 %** | **0.000** | **null — scores exactly random** |
| BankHeist | 18 | 43 % | ~0 | marginal (3-5x random) |

**Assault essentially never shows the same frame twice** (2,995 distinct out of
3,000), so the QEC is a write-only log: every query falls through to a
k-neighbour mean over a table it can never index, and the policy is noise. This
is not an encoder problem — counting *raw* 84x84 frames gives the identical
number, i.e. φ merges nothing and loses nothing.

Consequences:

* **The five games of Blundell et al. §4.1** (Ms. Pac-Man, Q*bert, River Raid,
  Frostbite, Space Invaders) are not an arbitrary choice.
* **Benchmark subsets selected for deep RL do not transfer.** The Atari-5 /
  Atari-3 methodology fits a regression on gradient-learner scores, for which
  frame repetition is irrelevant; Assault is in the Atari-3 test set and is
  near-worst-case for episodic control.
* **NEC does not share this constraint.** Its DND read path is a
  kernel-weighted sum over the p nearest neighbours, not an exact-match lookup,
  and its encoder is trained — so it degrades gracefully where MFEC collapses.
  Only its *blend* rule needs exact matches, which is why
  `train/dnd_blend_rate` reads ~0 on high-novelty games. Expect that; it is not
  a bug. MFEC is therefore the binding constraint when choosing games for a
  study that runs both.

The recommended set for a joint MFEC/NEC study is the intersection of "MFEC
demonstrably works" and "in the Atari-100k 26": **Ms. Pac-Man, Q*bert,
Frostbite**.

The NEC side of that set is implemented: `experiment/nec/{mspacman,qbert,
frostbite}{,_dinov2,_clip}.yaml` over `{mspacman,qbert,frostbite}_nec_{train,
eval}.yaml`. See "The NEC encoder ablation" below. The MFEC side takes the game
as a variable instead (`experiment=mfec/<encoder> game=<Game>`).

### `game` — one token per Atari game, no per-game config files

`configs/environment/atari_mfec_*.yaml` build `name: ALE/${game}-v5`, so a whole
suite is a sweep rather than a directory of near-duplicates:

```shell
python src/train.py -m experiment=mfec/clip game=Assault,BankHeist,RoadRunner
```

Nothing else is game-specific: `|A|`, the QEC's shape and the random
projection's input width all come off the env spec in `MFECAlgorithm.setup()`
(verified on Assault = 7 actions and BankHeist = 18, no config change).

**There is deliberately no `experiment/mfec/frostbite.yaml`** (nor a Q\*bert or
Ms. Pac-Man one — `mfec/qbert.yaml` is the deprecated pre-`game` file, see its
header). MFEC on Frostbite is:

```shell
python src/train.py experiment=mfec/rp_gray game=Frostbite
python src/train.py -m \
    experiment=mfec/rp_gray,mfec/rp_rgb,mfec/vae,mfec/dinov2,mfec/resnet,mfec/clip,mfec/mae \
    game=MsPacman,Qbert,Frostbite            # the full 7 x 3 grid, 21 runs
```

Verified end to end: all seven arms compose under `game=Frostbite` with
`environment.name == eval_environment.name == ALE/Frostbite-v5` and
`run.name == mfec_Frostbite_<encoder>_seed42`. Guarded by
`test_the_game_variable_reaches_both_envs`, which parametrises over the three
study games as well as the Atari-3 set.

If you are tempted to add a per-game MFEC file because the NEC side has them,
note the asymmetry is forced, not stylistic: NEC's env pair carries
`EndOfLifeTransform` and a `CatFrames` stack and is written per game, while
MFEC's is a single frame with no reward clipping and could therefore be made
generic. Adding `mfec/frostbite*.yaml` would fork the ablation's shared
settings across twice as many files, which is exactly the failure mode
"The encoder ablation" below exists to prevent.

Three things that must not regress:

* **`run.game` follows `${game}`** for the six ablation arms. Hardcoded, every
  game in a multirun would share one `run.name` and overwrite the previous
  game's output directory and W&B run.
* **The env id is `${oc.select:game,MsPacman}`, not `${game}`.**
  `scripts/encoder_diagnostics.py` loads these files with a bare
  `OmegaConf.load` outside Hydra, where a plain `${game}` raises
  `InterpolationKeyError`.
* **`buffer_size` is per action**, so an 18-action game (Frostbite, BankHeist,
  RoadRunner, Jamesbond) allocates twice Ms. Pac-Man's QEC — see the sizing note
  below.  It is pinned equal to `total_frames`, which gives an exact bound
  rather than measured headroom: total insertions cannot exceed `total_frames`,
  so `buffer_size: 100_000` provably never evicts for any `|A|`.

Guarded by `tests/test_encoder_factory.py` section 6.

### The encoder ablation — hold everything but φ equal

Seven arms, **two** env pairs, on whichever `game` is selected.  Every non-φ
knob is identical: 100k decisions,
`num_envs: 4`, `eval_every_n_steps: 10_000`, `buffer_size: 100_000`.  Identical
across **games** too, not just arms — the grid is 7 encoders x
{Ms. Pac-Man, Q\*bert, Frostbite}, and a knob that moved with the game would make
a cross-game read something other than a game comparison.

| experiment | env pair | φ | d |
|---|---|---|---|
| `mfec/rp_gray` | `atari_mfec_*` (84×84 grayscale) | random projection | 64 |
| `mfec/vae` | `atari_mfec_*` | ConvVAE | 64 |
| `mfec/rp_rgb` | `atari_mfec_*_rgb` (210×160 RGB) | random projection | 64 |
| `mfec/dinov2` | `atari_mfec_*_rgb` | DINOv2 ViT-S/14 | 384 |
| `mfec/resnet` | `atari_mfec_*_rgb` | ResNet-18 | 512 |
| `mfec/clip` | `atari_mfec_*_rgb` | CLIP ViT-B-32 | 512 |
| `mfec/mae` | `atari_mfec_*_rgb` | MAE ViT-B/16 | 768 |

`rp_gray` → `rp_rgb` isolates the observation; `rp_rgb` →
the PVM arms isolates the encoder.

**`mae` is what makes the PVM step an experiment rather than a bake-off.**
ImageNet supervision, DINOv2's self-distillation and CLIP's contrastive loss all
optimise for semantic discriminability, so those three vary the backbone but not
the *kind* of objective. MAE's is masked pixel reconstruction — no term in its
loss ever compares two images — so it is the only arm that varies the property
MFEC actually depends on (φ as a metric). MAE scoring below the other PVMs is
the hypothesis under test; do not read it as a broken arm. Its two costs are
real and are documented below: ~4x the CLIP arm's φ time (197 tokens vs 50) and
the widest `state_dim` in the study. **`rp_rgb` is not optional** — the
PVM arms see RGB and the paper baseline sees grayscale, and on Ms. Pac-Man
ghost identity is colour-coded (a blue ghost is edible, worth 200–1600), so a
direct `rp_gray` vs `dinov2` comparison credits the representation
for information the baseline never received.

Three traps this has already fallen into, now pinned by
`tests/test_encoder_factory.py`:

1. **`resnet` ran 12.5M decisions** against everyone else's 1M, with
   `num_envs: 4` — a budget comparison wearing an encoder comparison's clothes.
2. **`vae` ran on `mspacman_train_singleframe`**, a DQN-style stack
   carrying `SignTransform` (reward clipped to `{-1,0,+1}`, so its
   `episode_reward` was a pellet count, not a score — and MFEC argmaxes over
   raw Monte-Carlo returns, so a dot scored the same as a ghost),
   `EndOfLifeTransform`, and a 4,500-step cap instead of 27,000. It never
   needed that file: `atari_mfec_train.yaml` is *already* a single 84×84
   grayscale frame, which is exactly the paper's VAE input (`x ∈ R^7056`).
   `mspacman_{train,eval}_singleframe.yaml` are now unused.
3. **`mspacman_mfec_*_dinov2` duplicated `atari_mfec_*_rgb`** byte for byte.
   Every RGB arm now shares the `_rgb` pair; the `_dinov2` env files are unused.

Note the control keeps its exact-match hash: `RandomProjectionEncoder`'s
float64 accumulation holds even at 100,800 input dims (RGB 210×160), measured
`key b/s = 1.000`. So `rp_rgb` takes the O(1) dict path while the PVM arms on
the *same* observations fall through to the near-exact rescue — a difference in
lookup mechanism, not in the value returned (see the key-stability section).

### QEC memory is sized by `state_dim`, and it is not small

`QEC._init_states` allocates `(num_actions, buffer_size, state_dim)` float32
**eagerly**, so a wider encoder costs GPU memory before a frame is collected.
On Ms. Pac-Man's 9 actions at the `mfec_atari` default `buffer_size: 1_000_000`:

| encoder | `state_dim` | eager allocation |
|---|---|---|
| `random_projection` | 64 | 2.3 GB |
| `dinov2` ViT-S/14 | 384 | 13.8 GB |
| `resnet` resnet18 / `clip` ViT-B-32 | 512 | 18.4 GB |
| `dinov2` ViT-B/14, `clip` ViT-L-14 | 768 | 27.6 GB |

Every ablation arm sets `buffer_size: 100_000`, **pinned equal to
`trainer.total_frames`** (the shared probe budget). That is a structural bound,
not measured headroom: a run inserts at most one QEC entry per decision, so
total insertions cannot exceed `total_frames`, and they spread over `|A|`
per-action tables — the per-action peak therefore *falls* as `|A|` rises, and
LRU eviction can never fire on any game. (The earlier `150_000` was sized off a
measurement against a 1M-decision budget: `train/qec_size`, the mean over
actions, topped out ~40 k with the busiest action ~1.8x that, so a real peak of
~72 k. The pin supersedes that argument.)

**`|A|` is the other multiplier, and it belongs to the game.** At
`buffer_size: 100_000`:

| encoder | d | 9 actions (Ms. Pac-Man) | 6 (Q\*bert) | 18 (Frostbite) |
|---|---|---|---|---|
| `random_projection` / `vae` | 64 | 0.23 GB | 0.15 GB | 0.46 GB |
| `dinov2` ViT-S/14 | 384 | 1.38 GB | 0.92 GB | 2.76 GB |
| `resnet` / `clip` | 512 | 1.84 GB | 1.23 GB | 3.69 GB |
| `mae` ViT-B/16 | 768 | 2.76 GB | 1.84 GB | **5.53 GB** |

**`mae` on Frostbite is the largest allocation in the study** and the one to
check before queueing it: 5.53 GB of QEC states, plus ~0.34 GB of ViT-B/16
weights and ~0.3–1 GB of transient activations (`_EMBED_CHUNK_BYTES` is 64 MB of
raw observation ≈ 158 RGB frames per forward), i.e. **~6.2–6.9 GB** on the same
device — it fits an 8 GB card with little room, comfortably on 11 GB.
**Do not shrink `buffer_size` for the `mae` arm alone** to buy memory back:
it is pinned equal across the whole 7 x 3 grid by
`test_the_budget_is_held_equal_across_games_too`, and an arm-specific value would
turn a between-arm read into a comparison of memory budgets.

**RAISE IT IN LOCKSTEP WITH `total_frames`.** The whole no-eviction argument is
the pin; leaving `buffer_size` at 1e5 against a longer run reintroduces
eviction mid-run, at a different moment in each arm, silently breaking the
ablation. Watch `train/qec_size` either way — it logs the mean over actions, and
the busiest action runs ~1.8x it.

### Adding an encoder keyword is a THREE-file change

Keywords are threaded `experiment YAML → MFECAlgorithm.__init__ →
make_encoder(...)`, and `setup()` passes **every** encoder's keywords on every
call regardless of `encoder_name`. So a keyword added in only some of those
places does not fail "just for that encoder":

| Missed step | Failure |
|---|---|
| Not in `make_encoder`'s signature | `TypeError: make_encoder() got an unexpected keyword argument ...` in `setup()`, for **every** encoder — the whole algorithm is down |
| Encoder class not imported in `factory.py` | `NameError` the first time that branch is selected |
| YAML key ≠ `__init__` parameter name | `TypeError` from Hydra `instantiate()` before setup even runs |

All three shipped together when `resnet` was added (commit `9f9bbc4`) and took
down every MFEC run, including `random_projection`. `tests/test_resnet_encoder.py`
did not catch it because it constructs `ResNetEncoder` directly and never goes
through the factory — the seam is now pinned by `tests/test_encoder_factory.py`,
which checks the signature against the real call site (via `ast`), that every
name the factory references is bound, and that each MFEC experiment's
`algorithm:` keys bind to `MFECAlgorithm.__init__`.

**Keep `ResNetEncoder` in `eval()` mode.** In `train()` mode BatchNorm
normalises with batch statistics, so the same frame embeds differently
depending on batch composition and the QEC exact-hit path never fires. This is
the one way ResNet is more fragile than DINOv2 as a PVM; `load_state()`
re-asserts it, and `tests/test_resnet_encoder.py` covers both.

### Policy chain — φ runs ONCE per frame

```
pixels ──_EmbedModule(φ)──▶ "state_embedding" ──QValueActor(QECPolicy)──▶ action
```

`_EmbedModule` is a separate `TensorDictModule` so the **collector persists
the embedding under `state_embedding`**, and `MFECAlgorithm.step()` reads it
back instead of embedding the same frames a second time. This is exact, not an
approximation: the encoder is frozen, so a cached vector is bit-identical to a
recomputation (verified in
`tests/test_mfec_encoder_refactor.py::test_step_reuses_collector_embedding_and_falls_back_when_absent`).

Consequences for anyone editing this:
- **`QECPolicy.forward()` takes a `(..., d)` embedding, NOT `(..., C, H, W)`
  pixels.** `QECPolicy.embed()` still exists for tests/offline analysis.
- `get_policy()` must return the full chain (`self._policy`), never
  `self.q_actor` alone — the actor reads a key only `_EmbedModule` writes.
- `step()` falls back to `_embed_observations()` if `state_embedding` is
  absent. That is reachable only with `init_random_frames > 0`, which
  `get_collector_config()` pins to 0 for MFEC.
- **Do not copy this to NEC.** NEC's CNN is trained, so it changes between
  collection and `step()`; a cached embedding would be stale. NEC re-embeds
  deliberately (`NECAlgorithm._embed`).

Embedding is chunked by a **byte budget** (`_EMBED_CHUNK_BYTES`), not by
`num_envs`. Sizing it by `num_envs` (as it was until review) means
`frames_per_batch // num_envs` forward passes at batch size `num_envs` — free
for a random-projection matmul, but for a ViT it leaves the GPU idle and pays
that many times the launch overhead.

### MFEC evaluates greedily — `eval_eps: 0.0`, `num_eval_episodes: 1`

**This reverses an earlier decision. Read the whole section before raising
`eval_eps` again.**

`get_policy()` ends in `_EvalEGreedyModule`, but its ε defaults to **0.0**, so
evaluation is a pure argmax rollout.

The module still has to be there. torchrl's `EGreedyModule.forward` is gated on
`exploration_type() in (ExplorationType.RANDOM, None)` and
`BaseTrainer.evaluate()` runs under `ExplorationType.MODE`, so a *stock*
`EGreedyModule` in the eval chain is silently a no-op — raising `eval_eps` would
do nothing. `_EvalEGreedyModule` forces the gate open. Do not "fix" that by
changing `evaluate()`'s `set_exploration_type`: MODE is correct for DQN/A2C.
Guarded by `tests/test_mfec_eval_policy.py`.

#### Why ε at evaluation was wrong

`eval_eps` was 0.005 (Blundell et al. §4.1) so that `num_eval_episodes` produced
more than one distinct sample: MFEC requires `repeat_action_probability=0.0`, so
ALE is deterministic, `NoopResetEnv` was believed not to change Ms. Pac-Man's
score, and a greedy rollout would repeat itself exactly. The `NoopResetEnv` half
is **false** — see "`num_eval_episodes` was 1 for a wrong reason" below — and
the cure was worse than the disease regardless. Measured against one QEC
(Ms. Pac-Man, 51 k frames, 6 episodes each):

| `eval_eps` | returns | mean | min |
|---|---|---|---|
| **0.000** | `[1440, 1440, 1440, 1440, 1440, 1440]` | **1440** | 1440 |
| 0.005 | `[490, 1440, 870, 550, 1440, 1440]` | 1038 | 490 |

A 1000-decision episode takes ~5 forced random actions at ε = 0.005, and MFEC
cannot absorb even one. Off its memorised trajectory ~7 of 9 actions have no
exact QEC entry and lose the argmax to the estimator gap (next section), so the
episode never recovers. Consequences on a real 1 M-decision run:

- `eval/return_mean` **understates the policy by ~30%**;
- `eval/return_max` is the only statistic reporting the true score;
- `eval/return_min` is the worst of N ε-derailments — an extreme order statistic
  of a heavy left tail. It sits near the floor **by construction** and cannot
  improve however good the memory gets. Observed: min pinned at ~200 (below
  random play) for a whole run while max climbed past 4000;
- `eval/return_std` measures ε-sensitivity, not policy variability.

Nothing is lost by evaluating greedily: the ε = 0.005 score the paper reports is
the **collector's**, and it is already logged as `train/episode_reward` — which
this file elsewhere already names as the paper-comparable metric.

#### `num_eval_episodes` was 1 for a wrong reason — MEASURED

Every `configs/experiment/mfec/*.yaml` used to set `num_eval_episodes: 1`,
justified by "eval is deterministic, so N > 1 buys N copies of one number".
**The premise does not hold.** `NoopResetEnv(noops=30, random=True)` draws 1–30
no-ops on every reset, and Ms. Pac-Man does not absorb them. Measured on
`atari_mfec_eval_rgb`:

| probe | result |
|---|---|
| distinct first observations over 8 resets | **7 / 8** |
| return under one *fixed* 600-action sequence, 4 resets | `[380, 170, 180, 340]` |

A closed-loop greedy policy may partly re-converge where an open-loop action
stream cannot — which is probably what the earlier `eval/return_std == 0`
measurement saw — but the **start state genuinely varies**, so `N = 1` was one
sample from a real distribution, not the whole of a degenerate one.

The reported symptom is exactly that: on the `mfec/resnet` Ms. Pac-Man runs
`eval/return_{min,mean,max}` logged as one identical curve swinging several
hundred points between adjacent eval steps, and `eval/return_std` was never
defined at all (`BaseTrainer.evaluate` omits it at `n = 1` rather than
fabricating a 0.0).

**Every MFEC experiment now sets `num_eval_episodes: 5`**, matching all eleven
NEC experiments and `configs/train.yaml`'s default, so every `eval/return_mean`
in the study carries the same sample size and the same standard error. A per-arm
`N` is a real failure mode this repo has already hit once — see the NEC `nature`
baseline, which ran 10 against the PVM arms' 5. Note eval episodes run at batch
size 1, the slowest φ path, so this is not free: budget ~5x the previous eval
cost per run, on top of the FP32 conv pin.

This does **not** reopen `eval_eps`. The 1440-vs-1038 measurement above is about
ε derailing the policy, not about sample count, and stands unchanged.

Raise `eval_eps` only to deliberately measure ε-robustness, and then read
`eval/return_mean` as such. Guarded by `tests/test_mfec_estimator_gap.py`.
`NECAlgorithm` sets `eval_eps: 0.05` for the decorrelation reason above and has
not been re-examined against this measurement.

### Two estimators, one argmax — `eval/exact_minus_knn_value`

Eq. (2) answers a query in one of two ways — the stored value on an exact
match, a k-neighbour mean otherwise — and `argmax` compares the two as if they
were commensurable. They are not. Measured on Ms. Pac-Man (random projection,
`state_dim=64`, 51 k frames, 957 held-out `(s, a)` pairs):

| estimator | bias vs true return-to-go | corr |
|---|---|---|
| exact stored value | **+60** | **1.00** |
| kNN mean | **−411** | 0.63 |

`exact − kNN` on the *same* `(s, a)`: **+540** (median +546). The exact branch
is a max over realised returns (upward biased); the kNN branch is a mean over
neighbours (downward biased). So an action the agent has already taken from a
state beats one it has not by ~540 points of pure estimator bias, regardless of
which is actually better. Measured consequence: the policy picks an exact-hit
action **99.8 %** of the time although only ~1.8 of 9 actions carry one.

This is Eq. (2) as the paper writes it, **not** an implementation defect, and it
has not been changed. But it is why MFEC cannot recover from an ε-derailment,
and why the QEC behaves as a trajectory replay rather than a value function.

`QEC.value_stats()` accumulates both branches and `MFECAlgorithm.eval_metrics()`
reports `eval/exact_minus_knn_value`. It is a per-encoder quantity: a
representation whose neighbourhoods are semantically tighter should shrink the
gap, so it is the natural dependent variable for the encoder ablation
(`random_projection` vs `vae` vs `dinov2` vs `resnet`).

Related: the `+inf` in `QEC.estimate_all` fires only while an action's **whole
buffer** holds `<= k` entries, which stops happening inside episode 1. It is a
warm-up device, not per-`(s, a)` optimism, and it does nothing to offset the gap
above. The module docstring used to claim otherwise.

### Optimistic init must be tie-broken RANDOMLY

`QEC.estimate_all` returns `+inf` for actions whose buffer holds `<= k`
entries, so untried actions are always preferred (Blundell et al. 2016 §2).
`QECPolicy.forward` converts those sentinels to a finite value — `QValueActor`
cannot argmax over `inf` — as
`QECPolicy.OPTIMISTIC_VALUE + U(0, QECPolicy.OPTIMISTIC_JITTER)`, drawn
independently per `(state, action)`.

The jitter is load-bearing. Mapping every `+inf` onto one constant makes the
untried actions **exact ties**, and argmax resolves ties by lowest index.
Measured on a 9-action Ms. Pac-Man spec: an empty QEC emitted action 0 for
499/500 states; once action 0's buffer passed `k`, action 1 for 499/500. The
agent played one fixed action per episode, cycling 0..8, and seeded the QEC
with 9 degenerate single-action trajectories whose max-returns then persist
forever (Eq. 1 never decreases). Guarded by
`tests/test_mfec_optimistic_tiebreak.py`.

Constraints if you touch this:
- **The jitter cannot be small.** `q_values` is float32 and the ULP at 1e9 is
  64.0, so `uniform(0, 1)` jitter rounds straight back to exactly 1e9 and
  restores the tie. `OPTIMISTIC_JITTER` is 1e6.
- **`OPTIMISTIC_VALUE` must exceed any achievable return** (unclipped
  Ms. Pac-Man tops out ~3e4) or a real Q-value would outrank an untried action.
- **Only the `+inf` path is perturbed.** Finite estimates pass through
  `torch.where` untouched, so a QEC that has information about a state is
  exactly as deterministic as before — the guarantee that matters for eval.
  The draw uses the global torch RNG, so seeded runs stay reproducible.
- **NEC has the same fix**, in `DNDPolicy.forward` (`src/algorithms/nec.py`),
  with its own `OPTIMISTIC_VALUE` / `OPTIMISTIC_JITTER` constants — duplicated
  rather than imported so `nec.py` stays free of a dependency on `mfec.py`.
  This note used to say NEC "still has the un-jittered
  `torch.where(isinf, 1e9, ...)` form and the same latent defect", left alone
  because the failure had only been confirmed experimentally for MFEC. It has
  now been confirmed for NEC too, on a 9-action Ms. Pac-Man spec over 500
  states: an empty DND emitted action 0 for **500/500**, and once actions 0-3
  passed `k` it emitted action 4 for **500/500**. Guarded by
  `tests/test_nec_optimistic_tiebreak.py`.
- NEC is less exposed than MFEC only because `init_random_frames` covers part
  of the window in which a table sits at or below `k`. The exposure is worst on
  the game with the most tables to fill (Frostbite, 18 actions), and the window
  is precisely when the DND is first being seeded.
- **Determinism is guaranteed on the finite path only**, for both algorithms.
  A test that asserts a deterministic argmax must populate the memory first —
  `tests/test_nec_eval_policy.py::test_zero_eval_eps_restores_deterministic_argmax`
  used to assert it against an EMPTY DND, which was asserting the collapse
  rather than the fix.

Contract (`Encoder`):
- `embed(obs) -> (B, d) float32` on `obs.device`; **must be deterministic**
  (identical pixels -> identical embedding), or the QEC exact-hit hash path
  (`QEC._make_keys`) never fires.
- Determinism must hold **across batch shapes**, not just across repeats:
  training embeds `num_envs` rows while `BaseTrainer.evaluate` embeds 1, and a
  float32 reduction picks a different kernel per shape.
  `RandomProjectionEncoder` accumulates in float64 for exactly this reason
  (see its docstring); a float32 ViT such as `DINOv2Encoder` has **no such
  guarantee**. Check any new encoder with
  `python scripts/encoder_diagnostics.py --device cuda` before trusting a
  training run — it reports the batch-vs-single key match rate, which must be
  1.000.
- `state() -> dict` / `load_state(dict) -> None` for checkpointing — plugged
  into `MFECAlgorithm._get_training_state()` / `_load_training_state()` as
  `extra["encoder_state"]`.

Two implementations:
- `random_projection` (default) — `RandomProjectionEncoder`
  (`src/encoders/random_projectins.py` — note the filename typo, kept as-is
  to avoid churning existing imports). Fixed `np.random.default_rng(seed)`
  projection matrix, no training needed.
- `vae` — `VAEEncoder` (`src/encoders/vae_encoder.py`), wraps a frozen
  `src/models/conv_vae.py::ConvVAE`. Architecture and training regime match
  Blundell et al. 2016 ("Model-Free Episodic Control"), Appendix D exactly:
  encoder = 4 conv layers ({32,32,64,64} kernels {4,5,5,4}, stride
  {2,2,2,2}, no padding, ReLU) -> 512-unit FC ReLU -> linear (mean, log-std)
  heads, `latent_dim=32`; decoder mirrors this and also outputs (mean,
  log-std) for `p(x|z)` (a full Gaussian NLL reconstruction loss, not MSE).
  `embed(obs)` returns `concat(mean, log-std)` — **64 values, both
  deterministic** (no sampling; only `reparameterize()` samples, and only
  during training) — per the paper: "both the mean and log-standard-deviation
  parameters ... were used as dimensions for computing Euclidean distances
  in the episodic controller." `VAEEncoder`'s `state_dim` is the *exposed*
  width (64); the underlying `ConvVAE.latent_dim` is `state_dim // 2`.

  The paper's VAE input is a **single 84×84 grayscale frame** (`x ∈ R^7056`,
  `in_channels=1`), not the 4-frame stack the rest of this repo's Atari
  configs use (`CatFrames N=4`). MFEC-with-VAE therefore needs a
  single-frame environment variant — see `mspacman_train_singleframe.yaml`
  / `mspacman_eval_singleframe.yaml` (drop `CatFrames` from
  `mspacman_train.yaml` / `mspacman_eval.yaml`) and
  `experiment/mfec/vae.yaml`, which composes them with
  `encoder_name=vae`. This is a deliberate, paper-faithful trade-off: MFEC
  loses temporal (velocity) information when using this encoder, vs. the
  `random_projection` variant which keeps `CatFrames`. Only replicate the
  singleframe/`*_vae.yaml` pattern for another game if you actually want
  paper-exact VAE behaviour there.

  Requires a checkpoint: `algorithm.vae_checkpoint=<path>` pointing at a
  `torch.save(vae.state_dict())` file. Produce one with `src/train_vae.py`
  (defaults also match the paper — 1,000,000 random-policy frames, RMSProp,
  lr=1e-5, batch=100, 400,000 SGD steps):

  ```shell
  python src/train_vae.py
  python src/train.py experiment=mfec/vae \
      algorithm.vae_checkpoint=<checkpoint.save_path printed above>
  ```

  `train_vae.py` rolls out `collect.frames` pixel frames with a random
  policy (`torchrl.collectors.Collector` + `RandomPolicy`) from
  `environment` (defaults to `mspacman_train_singleframe`), trains `ConvVAE`
  with the Gaussian-NLL + KL loss (`src.models.conv_vae.vae_loss`) for
  `train.steps` SGD steps, and saves the state_dict to
  `checkpoint.save_path`. `in_channels` is inferred from the collected
  frames' shape (not a config knob), so it automatically matches whichever
  `environment` you point it at. It is a standalone Hydra entry point
  (`configs/train_vae.yaml`), not an `Algorithm`/`Trainer` — VAE pretraining
  is offline data prep, not part of the RL loop.
  `vae.latent_dim` (32 by default) must be half of the downstream
  `algorithm.state_dim` (64 by default): the exposed embedding is
  `mean ⊕ log-std`.

## NEC — gradients flow through the whole architecture

`NECAlgorithm` (Pritzel et al. 2017) differs from MFEC in ways that require
special care:

### 0. Where NEC's hyperparameters actually come from

The paper has **no hyperparameter table** — its only appendix is
"A. Scores on Atari Games", and everything is prose in §4. Do not cite a
"Table S1"; `configs/algorithm/nec_atari.yaml` used to and it was wrong.

Stated in §4 (safe to cite): `k`/p = 50, `kernel_delta` δ = 1e-3,
`n_step` N = 100, `dnd_capacity` 5e5 per action, replay buffer of the last
1e5 states, `batch_size` 32, one replay update per **16 raw ALE frames**,
`gamma` 0.99, action repeat 4, and no reward clipping.

Explicitly **swept and never reported** (do not claim paper authority):
the SGD learning rate (`lr`), the fast-update rate α (`dnd_lr`), the
embedding dimensionality (`embedding_dim`), and the ε-greedy rate.

The optimizer is **RMSProp**, per §4 ("we used the RMSProp algorithm for
gradient descent training"). It was Adam until a review re-checked the paper,
so `lr` was tuned against Adam — treat it as unvalidated.

One remaining caveat: the exact-match blend rule is **largely inert in
practice** (see §5 below).

### 0b. Frames are raw ALE frames; `frames_per_batch` is agent steps

Every frame-denominated number in the paper counts **raw ALE frames** — the
paper fixes this itself by noting 40M frames "corresponds to 185 hours of
gameplay" (40e6 / 60fps = 185.2 h). Everything in this repo's configs
(`total_frames`, `frames_per_batch`, `eval_every_n_steps`,
`annealing_frames`, `init_random_frames`) counts **agent steps**. With
action repeat 4 the conversion is `raw = agent_steps * 4`.

`gym_kwargs.frame_skip` **is** the ALE action repeat. TorchRL does not stack
a second repeat on top of ALE-v5's default: `GymEnv` forwards it into
`gym.make(..., frameskip=N)` and sets `wrapper_frame_skip = 1`, only falling
back to a wrapper-level repeat if `gym.make` *rejects* the kwarg (ALE-v5 does
not). Check with `GymEnv("ALE/Pong-v5", frame_skip=4).wrapper_frame_skip`.

`mspacman_nec_train.yaml` had `frame_skip: 1` on the mistaken belief that
TorchRL stacked a second repeat — that ran Ms. Pac-Man with no action repeat
at all and put every derived number out by 4x. All three
`experiment/nec/*.yaml` now land on the paper's targets exactly: 40M raw
frames total, eval every 200k raw frames, 1 update / 16 raw frames.

### 0c. `EndOfLifeTransform` is INERT for NEC and MFEC

It writes a separate `end-of-life` key and deliberately does **not** touch
`done` ("isn't registered within the done_spec because it should not instruct
the env to reset"). Nothing in `src/` reads that key, so life losses do not
segment episodes for either episodic-control algorithm — episodes are whole
games or `StepCounter` truncations. Several env-config comments claimed
otherwise; the NEC ones are corrected. Making it real means segmenting on
`end-of-life` in `NECAlgorithm.step()`, which is a behaviour change.

`done == terminated | truncated` does hold, so NEC's truncation test
(`not term_2d[env_idx, raw_end]` → bootstrap the tail from the DND) correctly
identifies `StepCounter` cutoffs versus real game-overs.

### 1. `DND.keys` and `DND.values` are updated by the loss — via sparse SGD

Paper Figure 2: *"Gradients flow through the entire architecture."* The
regression loss updates the stored keys and values, not just the CNN.

They are still plain (non-autograd) tensors. `_gradient_step` gathers the
retrieved neighbours into small leaf tensors, and `DND.apply_gradient`
scatters `-lr * grad` back into only the slots that minibatch read:

```python
nk = self.dnd.keys[a, indices].detach().requires_grad_(True)     # (n_a, k, d)
nv = self.dnd.values[a, indices].detach().requires_grad_(True)   # (n_a, k)
...                                     # after loss.backward():
self.dnd.apply_gradient(a, indices, nk.grad, nv.grad,
                        key_lr=self.dnd_key_lr, value_lr=self.dnd_value_lr)
```

**Do not "simplify" this by making `keys`/`values` autograd leaves.** Indexing
an autograd leaf produces a gradient the size of the *whole table* —
`num_actions x dnd_capacity x embedding_dim` = 1.15 GB at the Atari defaults,
allocated on every one of the 400 updates per collector batch.

**Do not give the DND a stateful optimiser** (Adam/RMSProp). Two reasons:
per-slot moments decay all 5e5 entries per action on every step, and the ring
buffer overwrites slots underneath them, so a freshly inserted entry inherits
the moments of whatever it evicted — that is the bug that previously drove
stored values negative and caused an earlier version to freeze the DND
entirely. Stateless SGD has no state to go stale.

`apply_gradient` restores two invariants; both are pinned by
`tests/test_nec_dnd_gradient.py`:

1. **Unit-norm keys.** A raw step pushes keys off the unit sphere and the
   kernel collapses back into `kernel_delta` domination (see
   `test_nec_kernel_scale.py`). Touched rows are re-projected.
2. **Hash validity.** `_key_to_slot` maps a quantised copy of a stored key to
   its slot; once the key moves, that entry is stale and `write_batch` could
   blend into the wrong slot. Moved slots are recorded and re-hashed in bulk by
   `flush_moved_slots()`, called once per collector batch from `step()` and
   reported as `train/dnd_rehashed`.

Setting `dnd_key_lr=dnd_value_lr=0` reproduces the old frozen-DND behaviour
bit-for-bit, so the change can be A/B'd against earlier runs.

##### Moved slots must be re-hashed, never delisted

`flush_moved_slots()` originally **delisted** moved slots — dropped them from
`_key_to_slot` and never restored them — because re-hashing "would cost a
GPU->CPU sync per touched slot per update". That objection applies to the
per-update location, not to `flush_moved_slots`, which already runs once per
collector batch and already pays one sync per action to build its slot list.

Delisting was quietly fatal, and it is the second reason the Ms. Pac-Man runs
plateaued (the first being encoder drift, above). The kNN retrieves
`num_updates x batch_size x k` neighbours per batch — 640,000 at 400/32/50 —
so within a few batches **essentially every entry had left the exact-match
dict permanently**. Two consequences:

1. **The blend rule stops firing.** `Q_i <- Q_i + alpha(Q^(N) - Q_i)`
   (§3.3, Eq. 4) only applies to a listed key, and it is NEC's headline
   mechanism (§1, "rapidly updated estimates of the value function"). Without
   it the DND degenerates to an append-only log of stale one-sample returns.
   `train/dnd_blend_rate` decaying toward 0 over a run is this, not the
   "expected" reading an earlier note claimed.
2. **Re-encounters duplicate instead of updating**, spending capacity on
   near-identical keys carrying disagreeing values.

Note the old `train/dnd_delisted` *undercounted* badly — it only incremented
when a slot still had a key, so each slot counted at most once ever. The
500-2500/batch it showed against ~198,000 total entries was already enough to
delist the whole table; the true touch count is ~1.1e5-1.9e5 per batch.
`train/dnd_rehashed` reports that real number, so expect it to read far higher.

Cost: 142 ms/batch at `num_updates=100`, 221 ms at 400 — against a kNN budget
of seconds. Getting there needed `_make_keys` to serialise in one pass and
slice, rather than allocating three Python objects per row (5x: 585 -> 108 ms
at 1.8e5 rows), which speeds up the insert path too. Pinned by
`tests/test_nec_dnd_rehash.py`, including the hash-collision case where two
moved keys quantise identically and the slot<->key mapping must stay
consistent.

**Why this mattered:** before this, a stored key was written once and never
refreshed. At `dnd_capacity=5e5` with ~178 inserts per action per collector
batch, an entry survived ~2800 batches — over a million gradient steps of CNN
drift — while the kNN kept retrieving it as if it still lived in the current
embedding space.

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

### 4b. Eviction is LRU (§3.3)

`DND._lru` stamps every slot returned by `knn_action` / `knn_all_actions`, and
every slot written or blended; `_insert_novel` fills any remaining free slots
and then evicts the coldest **live** ones (`topk(_lru[a, :size], largest=False)`).

This was FIFO until the eviction regime became reachable. That decision was
explicitly conditional — "eviction policy has no effect until a table is
full", which at `dnd_capacity=5e5` and ~178 inserts/action/batch meant ~4.5M
agent steps, beyond any run attempted. **Cutting `dnd_capacity` to 5e4 for
throughput voided the precondition without anyone re-opening the decision**:
tables now fill at ~450k agent steps, so a 900k-step Ms. Pac-Man run spent
half its life evicting under the wrong policy. Symptom in the logs:
`train/dnd_size` flattening against capacity while `train/q_loss` turns
around and climbs (1.5e3 → 8.5e3) with `train/q_values` already plateaued.

FIFO is specifically wrong once the blend rule is live (§5): a frequently
re-encountered state occupies ONE slot whose *value* is refreshed by blending
but whose *insertion order* never is, so FIFO discards precisely the
best-estimated, most-reused entries on a fixed timer.

Three things this had to get right, all pinned by `tests/test_nec_dnd_lru.py`:

* **Victims come from `[0, size)` only.** Searching the whole table returns
  the still-unwritten slots (they hold the sentinel `-1`), which are already
  being filled in the same call — duplicate slots would corrupt the
  slot↔key dicts.
* **Newly written slots are stamped.** Otherwise they inherit the evicted
  entry's cold stamp and are evicted again by the very next insert.
* **`_lru` and `_tick` are checkpointed.** Without that a resume restarts
  every stamp at 0 and evicts near-randomly until the table is swept again.
  Pre-LRU checkpoints have no `action_lru`; those entries load as `-1`
  (evicted first), which is the correct fallback — FIFO order is not a usable
  proxy for recency.

Cost on the hot path is negligible: one integer stamp per lookup *call* (not
per neighbour), fused into a single scatter — 0.1–0.7 % of the kNN call it
rides along with. `_write_ptrs` survives only for the fill phase (`ptr ==
size`) and is 0 once full, so `__getstate__` no longer rotates: with LRU the
slot order is not insertion order and there is nothing to rotate.

### 5. The exact-match blend rule is largely inert in practice

`write_batch` blends `Q_i ← Q_i + α(G − Q_i)` only on a bit-level hash match
of the quantised 64-d embedding. The CNN takes `num_updates` (100–400) RMSProp
steps between successive `step()` calls, so a state re-encountered in a later
batch essentially never re-hashes to its stored key. Blends therefore only
happen between duplicate frames embedded *within one* `step()` call, and the
DND behaves close to an insert-only log.

Now that gradients also move stored keys (§1), a hash entry additionally goes
stale the moment its key is updated — `flush_moved_slots()` re-hashes those, so
the blend rate falls further rather than silently blending into a slot whose
key has changed.

This is a consequence of porting MFEC's exact-hash design onto a *moving*
embedding. Making the rule fire across batches would mean matching within a
radius rather than exactly — a design change needing empirical validation,
deliberately **not** done.

`train/dnd_blend_rate` measures the residual. **Do not expect it near 0 on
Atari** (an earlier version of this section said so and it is wrong). Atari
has long runs of bit-identical frames — the opening "ready" freeze, the pause
after each death — and duplicate observations inside one `step()` call are
exactly the case the rule *does* fire on. Measured on a 1500-step Ms. Pac-Man
rollout: 264/1500 = 17.6 % of observations are byte-identical to an earlier
frame, and the quantised embeddings collapse at the same rate (1236 distinct
keys for 1236 distinct frames — the encoder adds no collapse of its own). A
blend rate of 0.1–0.5 on Ms. Pac-Man is the expected reading. Use
`eval/dnd_top_weight` to detect an embedding that has stopped discriminating.

### 5a. Audit against the paper (Pritzel et al. 2017)

Everything §4 actually publishes, checked against the composed config:

| Paper §4 / §3 | Value | Ours | |
|---|---|---|---|
| "store up to 5 x 10^5 memories per action" | 5e5 | `dnd_capacity: 50_000` | deviation — 5e5 never fills inside this repo's budgets; 5e4 reproduces the paper's own ~1.85 turnovers and bounds the exact-kNN scan (§5b). NOTE at `total_frames: 100_000` even 5e4 cannot bind: ~100k inserts spread over `\|A\|` tables peaks at ~11k/action on Ms. Pac-Man, so eviction never fires and the LRU policy below is inert |
| "nearest neighbours p = 50 in all our experiments" | 50 | `k: 50` | ok |
| "horizon of N = 100" | 100 | `n_step: 100` | ok |
| "replay buffer stores the only last 10^5 states" | 1e5 | `LazyTensorStorage(100_000)` | ok |
| "one replay update for every 16 observed frames" | — | `num_updates: 400` per 1600 agent steps = 1 per 4 agent steps = 1 per 16 raw frames | ok |
| "minibatch of size 32" | 32 | `batch_size: 32` | ok |
| "discount rate γ = 0.99" | 0.99 | `gamma: 0.99` | ok |
| "repeating each action four times" | 4 | `gym_kwargs.frame_skip: 4` | ok |
| "RMSProp algorithm" | — | `torch.optim.RMSprop` | ok |
| "NEC and MFEC do not require reward clipping" | — | no `SignTransform` in `mspacman_nec_train` | ok |
| "evaluate MFEC and NEC every 200.000 frames" | 200k raw | `eval_every_n_steps: 10_000` agent steps (40k raw) | deviation — shortened in proportion to the 100k-step probe budget |
| Eq. 3 N-step, bootstrap = `max_a Q(s_{t+N},a)` over **all** memories | — | `_compute_n_step_returns` + `_max_finite_q` | ok |
| §3.3 "earliest such values can be added is N steps after" | — | the sliding window in `step()` | ok |
| Eq. 4 `Q_i <- Q_i + α(Q^(N) - Q_i)` | — | `write_batch` blend | ok |
| §3.4 backprop updates "the keys and values of each action-specific memory ... using a lower learning rate than α" | < α = 0.1 | `dnd_key_lr: 1e-4`, `dnd_value_lr: 1e-5` | ok |
| Fig. 2 "Gradients flow through the entire architecture" | — | `DND.apply_gradient` | ok |
| "We set δ = 10^-3" | 1e-3 | `kernel_delta: 1e-3` | ok |

Every published number is now used as published. One thing worth knowing about
δ: it is a squared distance, so it is only meaningful relative to the embedding
scale, and this repo L2-normalises where the paper and the reference do not.
Measured, δ is ~17 % of the mean squared distance to the k=50 neighbours here
against ~0.003 % in the reference's geometry. Lowering it to 1e-5 was tried and
**reverted**: it lifts an exact re-encounter from 5.6 % to 45 % of the kernel
mass, but retrieval quality barely moved (held-out return-to-go prediction
r = +0.50 -> +0.51) and no score improvement could be demonstrated.

Swept-and-unreported by §4, so no paper value exists to match: the SGD learning
rate, α (`dnd_lr`), the embedding dimensionality, and the ε-greedy rate.

Deviations that are **not** numerical:

* **Eviction.** §3.3: "We overwrite the item that has least recently shown up
  as a neighbour" (LRU). Ours is LRU too — see §4b. Do NOT reason about this
  as "never binds": that was true only at `dnd_capacity=5e5`, and capacity is
  now 5e4, which fills at ~450k agent steps.
* **Approximate NN.** §3.1 uses kd-trees; ours is an exact scan. Same answer,
  much slower — see the fps table in §5b.
* **Warm-up.** `init_random_frames: 4_800` has no counterpart in the paper or
  the reference; both act ε-greedily from step 0. It exists only to keep the
  optimistic-init argmax off an empty DND, and it is sized against
  `trainer.total_frames` — see "Shared probe budget" above for the arithmetic
  and for the 1M-era values (`12_500`, and `50_000` in this file's own
  defaults) that survived the cut to 100k.
* **Optimistic init.** Neither the paper nor the reference returns `+inf` for an
  under-populated action (the reference returns 0.0); we return `1e9` **plus
  per-entry uniform jitter**, so the argmax is uniform over the under-populated
  actions rather than always the lowest index. See "Optimistic init must be
  tie-broken RANDOMLY" above — the un-jittered form played one fixed action for
  a whole rollout.
* **Gradient clipping.** Neither clips, and neither do we: `max_grad_norm:
  null`. (This bullet used to say "we keep `max_grad_norm: 10.0`"; that was
  already stale — see the §5b table row, which records the change as shipped.)
  At the old threshold of 10 it bound on **100 % of updates** (median raw grad
  norm ~1.7e3), making it the de-facto step-size control rather than a safety
  net. It could only be removed together with the reference's
  `rmsprop_eps=0.01`: torch's `eps=1e-8` damps nothing, so dropping both at
  once diverges (`train/q_loss` 1.5e3 -> 1.9e4 within three batches). The two
  shipped together; re-enable the clip only alongside a re-examination of
  `rmsprop_eps`.

### 5b. Diff against the reference implementation

`github.com/EndingCredits/Neural-Episodic-Control` is the implementation the module docstring cites. It is TensorFlow and
its encoder is fixed, but every *numerical* choice below was checked against it
directly (`NECAgent.py`, `knn_dictionary.py`, `networks.py`, `main.py`).
What actually shipped, and what was tried and reverted:

| | reference | ours | status |
|---|---|---|---|
| updates per step | `learn_step=4` -> 1 update / 4 agent steps = **400** per 1600 | `num_updates: 400` (runs had logged 100) | **fixed** — 4x too few |
| gradient clipping | none | `max_grad_norm: null` | **fixed** — was 10.0, binding on 100 % of updates |
| optimiser | `RMSPropOptimizer(1e-5, decay=0.9, epsilon=0.01)` | same: `lr=1e-5, rmsprop_alpha=0.9, rmsprop_eps=0.01` | **fixed** — was torch defaults |
| loss reduction | `reduce_sum` | `.mean()` | **not changed** |
| `kernel_delta` | 1e-3 with an unnormalised, `trunc_normal(0, 0.1)` embedding | 1e-3 with an L2-normalised one | **not changed** |

##### Why the optimiser trio was finally adopted

It was changed and reverted twice before, on the grounds that it "was not shown
to improve the score" and that torch's defaults were what the only ~3000 Ms.
Pac-Man run used. That reasoning missed the mechanism. Measured encoder drift
per collector batch, in units of the mean distance between distinct states:

| `num_updates` | old (1e-4 / .99 / 1e-8 / clip 10) | reference trio (1e-5 / .90 / .01) |
|---|---|---|
| 25 | 4.95x | 1.13x |
| 50 | 4.93x | 1.64x |
| 100 | 4.97x | 1.66x |
| 200 | 5.07x | 2.42x |
| 400 | 4.92x | 1.07x |

Two readings, both load-bearing:

1. **~5x the inter-state distance exceeds the diameter of the unit sphere the
   embeddings live on** (two random 64-d unit vectors sit at 1.41; mean
   inter-state distance is ~0.28). A key written one batch pointed essentially
   nowhere by the next, so the kNN retrieved neighbours whose stored returns
   belonged to unrelated states. This violates NEC's stated premise (§6, "keys
   stored in the DND remain relatively stable") and is the direct cause of the
   signature seen on the 350k-step Ms. Pac-Man run: `eval/dnd_top_weight`
   decaying toward `1/k` (a flat average over all k neighbours),
   `train/dnd_blend_rate` collapsing 0.35 -> 0.05, and `train/q_loss` *rising*
   1000 -> 2100 instead of converging. That run still climbed to ~1300 because
   γ¹⁰⁰ = 0.37 leaves 63 % of Q⁽ᴺ⁾ as observed Monte-Carlo reward — the learning
   came from the returns, not from the memory.
2. **The old drift is flat in `num_updates`** — it saturates within ~25
   updates, because the clip bound on every update and `eps=1e-8` then moved
   each parameter by ~`lr` regardless of gradient size. So lowering
   `num_updates` never addressed staleness, and any earlier note claiming
   "4x the updates is 4x the drift" is wrong.

The clip and `rmsprop_eps` must move together: clipping was only load-bearing
because `eps=1e-8` damps nothing (removing both at once diverges, `q_loss`
1.5e3 -> 1.9e4 in three batches), and `eps=0.01` damps the step on its own.
To restore the previous behaviour for an A/B:

    algorithm.lr=1e-4 algorithm.rmsprop_alpha=0.99 algorithm.rmsprop_eps=1e-8 algorithm.max_grad_norm=10

The loss reduction stays `.mean()`: for RMSProp, scaling the loss by `c` is
equivalent to dividing `eps` by `c`, so `reduce_sum` only transfers together
with a re-derived `(lr, alpha, eps)` — it is not an independent knob.

**Do not copy the reference's N-step return.** `NECAgent.Update` runs
`for i in xrange(start_t-1, t, -1)`, which stops at `t+1` and therefore drops
the immediate reward `r[t]`, shifting every discount exponent by one. Our
`lfilter`-based `_compute_n_step_returns` is correct; the reference is not.

Differences **left in place** — deliberate, but the first is the one to A/B first:

* **Gradients into `DND.keys`/`.values`.** The reference explicitly does not do
  this (its README lists it as a deviation). We do, per paper Fig. 2. Setting
  `dnd_key_lr=0 dnd_value_lr=0` is a bit-exact no-op by construction, so it is
  a free A/B.
* **Optimistic init.** Reference returns `Q = 0.0` for an action whose dict
  holds `<= k` entries; we return `+inf -> 1e9 + U(0, 1e6)`, so argmax chases
  whichever action is under-populated, uniformly among them rather than by
  lowest index (see "Optimistic init must be tie-broken RANDOMLY"). Reference
  also gates *training* on **all** actions being queryable.
* **`embedding_dim`** 64 vs the reference's 128.
* **Exploration.** Reference default is a *constant* `epsilon=0.1` (no anneal);
  we anneal 1.0 -> `eps_end` over `annealing_frames`, which is sized against
  `trainer.total_frames` (see "Shared probe budget").

#### fps: exact kNN vs the reference's ANNOY index

We do an exact scan of the whole table; the reference uses an ANNOY tree. Our
cost is therefore linear in DND size, so **throughput decays as training
proceeds** (CPU, 9 actions, k=50, per 1600-frame batch):

| entries/action | `estimate_all` | `knn_action` | per batch | fps |
|---|---|---|---|---|
| 25,000 | 21.6 ms | 1.27 ms | 8.9 s | 180 |
| 100,000 | 94.4 ms | 5.30 ms | 38.0 s | 42 |
| 500,000 | 408 ms | 24.8 ms | 171 s | 9 |

`_gradient_step` issues `num_updates x num_actions` `knn_action` calls per batch
(400 x 9 = 3600 full scans) against the reference's 32 tree lookups, and the
policy's `estimate_all` scans every action's table once per env step. The two
are roughly equal shares of the cost.

Three fixes, all applied, measured together at **31 fps -> 127 fps (4.1x)** for
a 1M-agent-step Ms. Pac-Man run:

1. **`dnd_capacity` 5e5 -> 50k.** Capacity does nothing until a table is full,
   and at ~0.093 inserts/action/step a 1M-step run only reaches ~93k — so 5e5
   never evicted anything and the DND held keys from the randomly-initialised
   encoder alongside current ones. 50k reproduces the paper's own turnover
   ratio (~1.85 cycles) and bounds the scan.
2. **`num_updates` 400 -> 100** for Ms. Pac-Man (see the experiment config):
   cuts the training-side kNN 4x.
3. **Epsilon anneal in closed form.** `EGreedyModule.step(frames)` loops
   `frames` times in Python — 1600 tensor ops per batch (26.8 ms) to evaluate a
   linear ramp. `NECAlgorithm.step` computes `max(eps_end, eps - n*delta)`
   directly: **125x faster**, and *more* accurate — torchrl's float32
   accumulation drifts ~2e-5 per batch and ~4e-4 over forty, always leaving
   epsilon above the configured schedule. Guarded by
   `tests/test_nec_epsilon_anneal.py`.

4. **`DND._block_topk`: rank on similarity, not distance.** `knn_all_actions`
   used to materialise the full `(A, b, n)` matrix and walk it four extra
   times — `2 - 2*sim` into a fresh tensor, then `clamp_min_`, then `sqrt_`,
   then `masked_fill` into *another* tensor of the same size (266 MB at A=9,
   b=74, n=1e5). Since `‖q-h‖² = 2 - 2·q·h` is monotone decreasing in `q·h`,
   ranking on the similarity matrix picks the same neighbours in the same
   order; the conversion to distance then runs over `(A, b, k)` instead of
   `(A, b, n)`, i.e. `n/k` = 2000x fewer elements. One in-place mask replaces
   two full-size allocations.

   Exact, not approximate — pinned against brute-force `cdist` in
   `tests/test_nec_knn_all_actions.py` across ragged, empty, below-`k`,
   full, chunked and non-unit-norm tables. Measured on `estimate_all`:

   | entries/action | before | after | speedup |
   |---|---|---|---|
   | 10,000 | 78.6 ms | 32.7 ms | 2.4x |
   | 50,000 | 395 ms | 154 ms | 2.6x |
   | 100,000 | 677 ms | 371 ms | 1.8x |

   This halves the *policy-side* share of the kNN cost, which fixes 1-3 above
   leave untouched (they only reduce the training-side scans).

Two smaller costs remain, both absent from the reference: the exact-match hash
bookkeeping in `_make_keys` (a GPU->CPU sync per write, ~6 s per 1M frames),
which the reference disabled outright as "cleaner (and faster without)"; and
`_embed` re-embedding the retained window every `step()`.

**On an approximate index.** `faiss` 1.15.0 is installed and currently unused.
It is a smaller win than it looks and has not been adopted: benchmarked at
100k entries/action, `IndexFlatL2` is only 1.9x faster than the exact torch
path, and `IndexHNSWFlat` (M=32, efSearch=128) is 7.9x but returns
**recall@50 = 0.745** and costs ~3.3 s per action to build — ~30 s per
collector batch across 9 actions if rebuilt each time. `dnd_key_lr` moves keys
on every update, so any prebuilt index goes stale within a batch. Adopting one
means committing to a rebuild cadence and accepting sub-unity recall; do that
only if the scan is still the binding constraint after 1-4 above.

### 6. NEC evaluation diagnostics (`eval/*`)

`NECAlgorithm` implements `reset_eval_metrics()` / `eval_metrics()`, the same
`BaseAlgorithm` hooks MFEC uses for `eval/exact_hit_rate`. NEC has no
exact-match concept on its read path (`estimate_all` has no shortcut — see §1),
so it reports the kernel's *shape* instead. `DND.reset_lookup_stats()` /
`lookup_stats()` are the read-path counters; accumulation is **off by default**
because `estimate_all` is the ε-greedy hot path, and only `reset_eval_metrics()`
switches it on.

| Metric | Reads |
|---|---|
| `eval/epsilon` | the exploration rate evaluation actually ran at |
| `eval/dnd_top_weight` | mean `max_i w_i / Σ_i w_i`. **Do not compare this against `1/k`** — see the calibration note below. Useful as a *trend*: it should lift when stored states are queried back. |
| `eval/dnd_nn_dist` | mean L2 from the query to its nearest stored key. Embeddings are unit-norm, so this is bounded by 2 and comparable across runs; drifting upward means stored keys are going stale faster than `dnd_key_lr` refreshes them. |
| `eval/dnd_optimistic_rate` | fraction of *(state, action)* pairs still answered with the `+inf` sentinel (`size <= k`). Above 0 late in a run means an action is starved and argmax is chasing the sentinel. |

Denominator is every *(state, action)* pair (all `\|A\|` per frame), matching
MFEC's eval-side convention and **not** comparable to any `train/*` counter.

#### Calibrating `eval/dnd_top_weight`

An earlier version of this section claimed that `top_weight` near `1/k` meant
the kernel had degenerated and the memory was unused. **That is wrong.** With
`k` = 50 in 64 dimensions the 1st and 50th neighbour sit at nearly the same
distance, so the inverse-distance weights are near-uniform even for a good
retriever. Measured on 6000 real Ms. Pac-Man frames, predicting held-out
discounted return-to-go:

| retriever | Pearson r | rmse vs predict-the-mean | `top_weight` |
|---|---|---|---|
| NEC embedding (shipped) | +0.50 | -14.7 % | 0.025 |
| raw pixels (k-NN reference) | +0.60 | -20.3 % | 0.029 |
| centre-then-normalise | +0.40 | -7.1 % | 0.033 |

So near-`1/k` is the normal reading, and a *higher* `top_weight` is not
automatically better — centring the embedding raises `top_weight` while making
retrieval measurably **worse**. Judge retrieval by the returns, and use
`top_weight` only for the exact-re-encounter signal it was added for.

Related, and also measured: `kernel_delta` = 1e-3 is ~20 % of the typical
squared neighbour distance under the shipped L2-normalised embedding
(d^2 ~ 4e-3), so the paper's division-by-zero *guard* is not negligible the way
the paper assumes. Dropping it to 1e-5 is a small, consistent improvement
(r +0.50 -> +0.51) and stays 22x above the float32 error floor of the
`2 - 2*sim` fast path in `_topk_l2_unit` (measured 4.5e-7). It is **not**
changed by default — the gain is marginal and 1e-3 is the published value.

### 7. `eval_eps` and `eps_end` must not drift apart

They are independent knobs in different files — `eval_eps` in
`configs/algorithm/nec_atari.yaml`, `eps_end` in the per-experiment override —
and nothing couples them. When they diverge the run **trains one policy and
scores another**, which shows up as:

* `eval/return_mean` well below `train/episode_reward` *at the same episode
  length* (a scoring-rate gap, not a survival gap);
* `eval/return_min` pinned near random play and not improving, while
  `eval/return_max` tracks `train/episode_reward`;
* every learning curve — `train/q_loss`, `train/q_values`, `train/dnd_size` —
  looking perfectly healthy.

Nothing else in the logs makes this recoverable after the fact, which is why
`setup()` warns when the two differ by more than 10x and `eval_metrics()`
reports `eval/epsilon` next to the returns. Guarded by
`tests/test_nec_eval_metrics.py`.

Note this is *not* the CPU/CUDA observation-drift failure MFEC hits
(`env_worker_device`): both env configs and `BaseTrainer.evaluate`'s manual
rollout loop were verified byte-identical to `env.rollout` on Ms. Pac-Man.

~~`NoopResetEnv` alone produces **exactly zero** return variance there — so a
non-zero `eval/return_std` is proof that eval ran with a real ε, not proof that
episodes were decorrelated by the noop reset.~~ **Retracted.** That was measured
from one closed-loop policy's returns, and it does not generalise: the no-op
draw *does* move the start state (7 of 8 resets give a different first
observation on `atari_mfec_eval_rgb`, and one fixed action sequence returns
`[380, 170, 180, 340]` — see "`num_eval_episodes` was 1 for a wrong reason").
A non-zero `eval/return_std` is therefore **not** proof that ε was on. Read
`eval/epsilon`, which `eval_metrics()` logs for exactly this purpose.

### 2. N-step returns (per-env, sliding window)

NEC uses bootstrapped N-step returns:

    Q^(N)(s_t, a_t) = Σ_{j=0}^{N-1} γ^j r_{t+j} + γ^N max_{a'} Q(s_{t+N}, a')

**Returns are finalised on a sliding window, not at episode end.** That
formula needs only `r_t..r_{t+n-1}` and the bootstrap state `s_{t+n}` — the
episode end is irrelevant except for the last `n_step` steps. So `step()`
finalises step *t* as soon as `t + n_step` arrives and retains at most
`n_step` raw frames per env.

This is a memory fix, and a large one: with `StepCounter.max_steps=27_000`
the old whole-episode carry was 3.05 GB per env — **24.4 GB across 8 envs, in
a 32 GB container** — versus ~11 MB now (measured flat at exactly `n_step`
frames over a 2,000-step done-free stream).

Two invariants the implementation must keep, both pinned by
`tests/test_nec_sliding_window.py`:

- **DND writes stay at episode end** (paper §3.3). Finalised
  `(h, action, return)` triples are buffered in `_carry["pending"]` at ~264 B
  per step and flushed by `_flush_pending_to_dnd()` when the done arrives.
  The window changes when returns are *computed*, not when they are written.
- **Chunking must not change the returns.** Feeding one episode whole vs in
  slivers of 1/2/3/5/7/11 steps must give identical returns; the test asserts
  that against a closed form (the fixture fills every DND table with a single
  constant so `max_a Q` is known exactly).

Note the returns are *not* bit-identical to the pre-window implementation:
step *t*'s bootstrap Q and embedding are now taken `n_step` steps after *t*
rather than at episode end, so both are fresher. Same formula, better-
conditioned inputs.

The bootstrap correction itself is applied after a full MC pass over the
window slice:

```python
# mc: full Monte Carlo returns via lfilter
gamma_n = gamma ** n_step
correction = gamma_n * (q_max_at_t_plus_n - mc[n_step:])
n_step_G[:T - n_step] = where(valid, mc[:T - n_step] + correction, mc[:T - n_step])
```

Bootstrapping uses the CURRENT DND state (written by previous episodes in
the same batch, or prior batches).  The window stores RAW observations (not
pre-computed embeddings) so they are re-embedded with the current network
at the start of each step() call, via `NECAlgorithm._embed()` — the single
place the L2 normalisation lives, chunked at `_EMBED_CHUNK` frames.

`q_max` comes from `_max_finite_q()`, **not** a plain `.max(dim=-1)`.
`DND.estimate_all` returns +inf for actions holding `<= k` entries, so a
plain max lets a single under-populated action poison `max_a Q` for every
state in the episode and silently drop the whole episode back to plain Monte
Carlo, with no metric revealing it. On H.E.R.O. (18 actions, k=50) one rarely
chosen action can sit below 51 entries for millions of frames once ε has
annealed, which would disable N-step bootstrapping for an entire run.

### 2a. Metrics: never report a skipped update as a zero

`NECAlgorithm._gradient_step()` returns **`None`**, not `(0.0, 0.0)`, when it
skips — replay buffer below `batch_size`, or every sampled action's DND table
still at/below `k`. `step()` averages only the updates that actually ran and
emits `train/q_loss` / `train/q_values` **only if there were any**, plus
`train/updates` (how many of `num_updates` ran) on every batch.

A skipped step is not a zero-loss step. Folding it in reported a
`train/q_loss` of exactly 0.0, which reads as "converged" rather than "never
ran" — and that is guaranteed for the first batches after *every* checkpoint
resume, because the replay buffer is not checkpointed. Watch `train/updates`;
if it sits below `num_updates`, the gradient path is starving.

Also watch the *magnitude* of `train/q_loss`. Every replayed state was also
written into `DND[a]` at episode end with a value equal to its own target, so
it is its own nearest neighbour at distance ~0 (weight 1/δ = 1000) and `Q̂`
reproduces the target almost exactly. A curve pinned around 1e-6 means the
network is not being pushed anywhere. Inherent to the paper's design, not a
defect in this implementation, but it makes the loss a poor progress signal.

### 2b. `estimate_all` has no exact-match shortcut (unlike QEC)

`DND.estimate_all` computes the kernel sum and nothing else. It does **not**
consult `_key_to_slot` to short-circuit an exact re-encounter to its stored
value, and does not special-case a near-zero nearest distance. `QEC` does
both, because MFEC's Eq. (2) genuinely defines an exact hit as a separate
case; NEC's Q is always the kernel sum.

The reason this matters is `_gradient_step`: it cannot reproduce a shortcut
(returning a stored constant kills the gradient), so any shortcut in
`estimate_all` makes the network **act** under a different Q-function than
the one it is **trained** on. Measured divergence when the shortcut was
present: ~0.3% on exact hits. Do not re-add it. The kernel needs no
shortcut — an exact re-encounter has distance 0 and therefore weight
1/δ = 1000, already dominating every genuine neighbour.

`_key_to_slot` / `_slot_to_key` remain write-path structures, used only by
`write_batch`'s blend rule and ring-buffer eviction.

### 3. Embedding networks — a config group, and it is TRAINABLE

NEC's φ is selected from the Hydra config group
`configs/algorithm/embedding_network/`, listed in the defaults of both
`configs/algorithm/nec.yaml` and `nec_atari.yaml`:

```yaml
defaults:
  - embedding_network: nature
  - _self_
```

so it swaps on the CLI without editing nested YAML:

```shell
python src/train.py experiment=nec/pong algorithm/embedding_network=nature
```

Options:

| option | factory | status |
|---|---|---|
| `nature` (default) | `src.networks.NatureEmbedding` | the paper's network; used by every `experiment/nec/*.yaml` |
| `dinov2_finetune` | `src.networks.DINOv2Embedding` | finetuned DINOv2 ViT-S/14; bundled as `experiment/nec/{mspacman,qbert,frostbite}_dinov2.yaml` — see below |
| `clip_finetune` | `src.networks.CLIPEmbedding` | finetuned CLIP ViT-B-32; bundled as `experiment/nec/{mspacman,qbert,frostbite}_clip.yaml`. Needs the optional `open_clip_torch` extra — see below |
| `mae_finetune` | `src.networks.MAEEmbedding` | finetuned MAE ViT-B/16 — the **reconstruction** arm, the only non-similarity objective; bundled as `experiment/nec/{mspacman,qbert,frostbite}_mae.yaml`. Needs the optional `timm` extra — see below |

#### The NEC encoder ablation — 3 games x 4 encoders

Twelve bundled arms: `experiment=nec/{mspacman,qbert,frostbite}` (nature),
`..._dinov2`, `..._clip`, `..._mae`.

The three PVM arms are not three flavours of one idea, and the writeup should
not present them as one. `dinov2_finetune` and `clip_finetune` both optimise a
**similarity** objective; `mae_finetune` optimises masked **pixel
reconstruction**, whose loss never compares two images. MAE is the arm that
varies the causal variable, and its expected position is *below* the other two.

| game | env pair | \|A\| | raw-frame repeat | MFEC `exact_hit_rate` |
|---|---|---|---|---|
| Ms. Pac-Man | `mspacman_nec_{train,eval}` | 9 | 22 % | 0.53 |
| Q*bert | `qbert_nec_{train,eval}` | 6 | 13 % | 0.60 |
| Frostbite | `frostbite_nec_{train,eval}` | 18 | 22 % | 0.41 |

The game set is the recommendation derived in "Which games MFEC can work on at
all" above — the intersection of "MFEC demonstrably works" and "in the
Atari-100k 26" — so the same three serve an MFEC comparison. NEC does not share
MFEC's exact-match constraint, so it is safe on all three by construction.

**Held identical across all twelve, by design:** `total_frames: 100_000` agent
steps (400k raw frames — the shared probe budget), `num_updates: 100`,
`eps_end: 0.001`, `annealing_frames: 10_000`, `init_random_frames: 4_800`,
`eval_eps: 0.005`, `num_envs: 8`, `num_eval_episodes: 5`,
`log_every_n_steps: 5_000`, `eval_every_n_steps: 10_000`,
`seed: 42`, and the six env configs (all unclipped,
no VecNorm, sticky actions off, 27,000-step cap). The measurements justifying
each value live in `experiment/nec/mspacman.yaml`; the other eleven files repeat
them with a pointer rather than re-arguing them.

**`num_envs` is NOT a free resource knob** — this is the correction that matters
most here. This section used to say "only resource knobs vary by arm: the ViT
arms use `num_envs: 8` and `num_eval_episodes: 5` where the nature arms use 16
and 10, because `num_envs` is also the ViT's collector-side inference batch."
The inference-batch part is true; the "only resource knobs" conclusion is not.
`num_envs` moves the learning curve for episodic control, and the MFEC side had
already measured how (canonical note in `configs/experiment/mfec/rp_gray.yaml`):

* NEC and MFEC write to memory only at EPISODE END, so frames sitting in a
  trailing partial episode when the run stops are **never written at all**. The
  loss is ~`num_envs x (mean episode length / 2)` — ABSOLUTE, so it bites
  hardest at a short budget. At this budget, with Ms. Pac-Man's measured
  498-step random-policy episodes, that is ~4.0k frames (4.0%) at 16 envs
  against ~2.0k (2.0%) at 8.
* 16 envs kept **39% fewer unique states** than 4 at equal frames: the envs
  share one memory and a low epsilon, so they largely retread one trajectory.

So the three `nature` arms at 16 were handicapped against the nine PVM arms at
8 that they are plotted against — a data-collection difference wearing an
encoder comparison's clothes, the same trap the MFEC ablation lists as failure
mode 1. All twelve now run `num_envs: 8`: 8 rather than MFEC's 4 because the
PVM arms' ViT inference batch IS `num_envs`, and 16 additionally overruns the
32 GB devcontainer limit for NEC (a preallocated ~11.3 GB replay buffer plus a
per-env raw-pixel carry, on top of the DND).

`num_eval_episodes` was 10 against the PVM arms' 5, which gave the baseline's
`eval/return_mean` a different standard error from the arms it is compared to.
Now 5 everywhere.

Parity is guarded by `tests/test_nec_ablation_parity.py`, which checks every
algorithm- and trainer-side knob for all twelve arms against their `nature`
baseline **and** across games.
`tests/test_nec_mae_finetune.py::test_the_mae_arm_holds_every_learning_knob_identical`
is the older, narrower version of the same check for the three `_mae` files.

Three failure modes to avoid when editing these:

* **Tuning one arm.** Any change to a learning-relevant knob has to land in all
  twelve or the comparison is void. Hydra experiment configs do not compose with
  each other, so there is no inheritance to lean on — it is twelve edits. This
  includes `kernel_delta`, which the CLIP and MAE arms both have a measured
  reason to want lowered (see their config comments) — lower it for all twelve
  and declare it, or not at all.
* **Dropping `run.game` / `run.encoder`.** Both are set explicitly in every file.
  `run.encoder` defaults to `${oc.select:algorithm.encoder_name,none}`, an
  MFEC-only key, so without the override every encoder arm of a game resolves to
  `nec_<game>_none_seed42` and they overwrite each other.
* **Reaching for `qbert_train.yaml`.** It clips rewards, caps episodes at 4,500
  agent steps and leaves v5's 0.25 sticky actions on. Use `qbert_nec_train.yaml`.
  (`frostbite` has no clipped variant at all, so that mistake is unreachable
  there.)

```shell
# One arm:
python src/train.py experiment=nec/frostbite

# Game sweep for one encoder, five seeds — run.group makes these one mean curve:
python src/train.py -m experiment=nec/mspacman,nec/qbert,nec/frostbite \
    trainer.seed=42,43,44,45,46
```

**Do not confuse this with MFEC's encoders** (`src/encoders/`, selected by the
`algorithm.encoder_name` *string*, not a config group). The two systems are
deliberately separate and must not be merged:

| | MFEC `Encoder` | NEC embedding network |
|---|---|---|
| type | plain object (`embed`/`state`/`load_state` protocol) | `nn.Module` |
| trained? | **frozen** — `DINOv2Encoder` calls `requires_grad_(False)` | **trained end-to-end** by `NECAlgorithm`'s Adam |
| determinism | bit-exact required (QEC hash) | not required (embeddings shift as the CNN learns) |
| configured by | `algorithm.encoder_name` string + `make_encoder()` | `algorithm/embedding_network` config group + `_partial_` factory |
| checkpointed via | `extra["encoder_state"]` | `policy_state_dict` (`embedding_net.state_dict()`) |

#### The contract: `src.networks.NECEmbeddingNetwork`

A lightweight `Protocol` next to `NatureEmbedding` — documentation, not
machinery (nothing type-checks against it; factories need not inherit it).
It states:

- callable as `factory(obs_shape: Sequence[int], embedding_dim: int, **kwargs)
  -> nn.Module`, everything after the two positional args keyword-only so a
  Hydra `_partial_` can pre-bind design kwargs;
- module `forward(obs: (B, *obs_shape)) -> (B, embedding_dim)` float32 (`B`
  is flat — callers reshape away leading `(E, T)` dims);
- **all parameters trainable by default** (`requires_grad=True`). `setup()`
  hands `embedding_net.parameters()` straight to Adam, so a frozen parameter
  is a dead parameter. An opt-in freeze kwarg is allowed (see
  `DINOv2Embedding.freeze_backbone`) but must default to unfrozen and must
  leave at least one trainable parameter;
- output must survive NEC's `F.normalize(h, dim=-1)`: no all-zero rows, and
  do **not** pre-normalise inside the module — see
  `tests/test_nec_kernel_scale.py` for why that normalisation exists;
- no state beyond `state_dict()`, unless you also extend
  `_get_training_state` / `_load_training_state`;
- **optionally** `param_groups(base_lr) -> list[dict]`. When present,
  `NECAlgorithm._build_optimizer` hands its result to RMSProp instead of a
  flat `parameters()` list — that is how a module splits itself across
  learning rates (see `DINOv2Embedding`). Modules without the attribute are
  unaffected. The groups must cover exactly the trainable parameters, and
  the grouping must be reproducible from the constructor args alone, since
  resume rebuilds the optimizer through the same method.

`NatureEmbedding(obs_shape, embedding_dim, *, ...)` is the standard
implementation: a CNN trunk + single dense layer — the same convolutional
body as NatureDQN with the MLP Q-head replaced.

Guarded by `tests/test_nec_embedding_network.py` (shape/dtype, gradient
flow, Hydra-composition architecture regression, config-swap end-to-end).

### 3a. Adding a new NEC embedding network

1. **Write the factory** in `src/networks.py`, satisfying
   `NECEmbeddingNetwork` above. Either a function returning an `nn.Module`
   (like `NatureEmbedding`) or an `nn.Module` subclass whose `__init__` takes
   `(obs_shape, embedding_dim, *, ...)` (like `DINOv2Embedding`). Keep every
   design kwarg keyword-only.
   - If the input needs adapting (Atari gives 4 stacked 84×84 grayscale
     frames), do it *inside* the module — a 1×1 conv channel adapter, a
     resize, a normalisation. The algorithm passes raw `obs_shape` and does
     not adapt anything.
   - End with an unconstrained projection to `embedding_dim` (e.g.
     `nn.Linear`) so NEC's downstream L2 normalisation still does work.
   - Do **not** call `requires_grad_(False)` unconditionally. That is the
     MFEC-encoder pattern and it silently produces a network that never
     learns.
2. **Add `configs/algorithm/embedding_network/<name>.yaml`** with
   `_partial_: true`, `_target_: src.networks.<Factory>`, and the design
   kwargs. Leave `obs_shape` / `embedding_dim` unbound — `setup()` supplies
   them positionally. For `activation_class`-style kwargs use
   `hydra.utils.get_class` (see the "Activation class in YAML" note above).
   Mark a required path as `???` so a run without it fails at instantiation.
3. **Run it** with `algorithm/embedding_network=<name>`. Nothing in
   `nec.yaml` / `nec_atari.yaml` / `experiment/nec/*.yaml` needs to change.
4. **Set `run.encoder`** in the experiment config (or on the CLI) if you want
   the run directory to distinguish encoders. `configs/train.yaml` computes
   `run.encoder` from `${oc.select:algorithm.encoder_name,none}`, which is an
   **MFEC-only** key — NEC runs are named `nec_<game>_none_seed<n>` regardless
   of which embedding network they use. Left as-is deliberately so the config
   group refactor did not rename existing NEC run/checkpoint directories.
5. **Extra checkpoint state.** `NECAlgorithm._get_training_state()` saves
   `embedding_net.state_dict()` as `policy_state_dict` and
   `_load_training_state()` restores it, then **rebuilds the Adam optimizer**
   over `embedding_net.parameters()` before loading the optimizer state. That
   covers any network whose entire state is its `state_dict()`. If yours has
   more (e.g. an external tokenizer, a separate pretrained-weights file whose
   path must be recorded, or non-persistent buffers), add it to
   `TrainingState.extra` in `_get_training_state()` and restore it in
   `_load_training_state()` — the same way MFEC threads `extra["encoder_state"]`
   through for its frozen encoders.

   Three checkpoint rules that are easy to get wrong (all were, and were
   fixed after a review):

   - **`greedy_state` must be saved.** `EGreedyModule` keeps the live ε in
     its own buffer, which is *not* part of `embedding_net.state_dict()`.
     Restoring `collected_frames` without it makes a resumed run restart
     exploration at `eps_init` and re-anneal from scratch. Both NEC and MFEC
     now save `extra["greedy_state"]`.
   - **Re-pin the DND/QEC device on load.** `torch.load`'s `map_location`
     does not rewrite the pickled `torch.device` inside `dnd_state`, so
     `_load_training_state` overwrites `dnd_state["device"]` with
     `self._buffer_device` before calling `__setstate__`. Without it a
     checkpoint written on `cuda:0` rebuilds the DND on `cuda:0` regardless
     of what the resuming run resolved to.
   - **The replay buffer and `_carry` are deliberately NOT checkpointed.**
     1e5 float32 pixel transitions is 11.3 GB, and `_carry` holds raw pixels
     for the in-flight episode (up to 508 MB/env, ~4 GB across 8 envs).
     Dropping `_carry` is nearly free: returns are computed *backwards* from
     the episode end, so a partial episode starting mid-stream still yields
     correct return-to-go for every step it contains — the cost is at most
     `num_envs` partial episodes per process restart. Dropping the replay
     buffer means the gradient path no-ops until it refills; `step()` reports
     **`train/updates`** so that gap is visible.
6. **Add tests** to `tests/test_nec_embedding_network.py` for the *config-group
   seam* (shape/dtype, all-params-trainable, YAML composes + instantiates), and
   put behaviour specific to your network in its own file —
   `tests/test_nec_dinov2_finetune.py` is the worked example. If the network
   needs downloaded weights, monkeypatch the loader to a stub backbone so CI
   stays offline (`tests/test_dinov2_encoder.py` and the `_StubViT` fixtures
   show the pattern), and put anything a stub structurally cannot check —
   real state_dict loading, real patch grids, real gradient paths — behind an
   opt-in env var as a second tier.
7. **Update `README.md` and `AGENTS.md`** (this table).

#### `dinov2_finetune` — finetunable DINOv2 ViT

`src.networks.DINOv2Embedding` +
`configs/algorithm/embedding_network/dinov2_finetune.yaml`. Pipeline:
torch.hub architecture load + local `.pth` state_dict (same pattern as
`src/encoders/dino_v2_encoder.py`), a 1×1 conv channel adapter, bilinear
resize to `image_size`, ImageNet normalisation, ViT, then
`nn.Linear(embed_dim, embedding_dim)`. Unlike the MFEC encoder it does
**not** freeze the backbone — `freeze_backbone: false` is the default and
the reason a NEC-specific class exists.

Run it via the bundled experiment, or as a group override on any NEC game:

```shell
python src/train.py experiment=nec/mspacman_dinov2 \
    algorithm.embedding_network.weights_path=/path/dinov2_vits14_pretrain.pth

python src/train.py experiment=nec/pong \
    algorithm/embedding_network=dinov2_finetune \
    algorithm.embedding_network.weights_path=/path/dinov2_vits14_pretrain.pth \
    run.encoder=dinov2
```

**Three things about it that are load-bearing, not incidental:**

1. **The channel adapter is initialised, not random.** `weight = 1/C`,
   `bias = 0`, so the 4-frame Atari stack enters the ViT as its mean
   replicated to R=G=B — a plain grayscale image in `[0, 1]`, which is what
   the ImageNet normalisation downstream assumes. A default `nn.Conv2d`
   init instead feeds the ViT out-of-distribution channels, and a
   pretrained representation that is only useful *after* the adapter has
   been learned is not a pretrained representation. The symmetry breaks on
   the first gradient (the 3 output channels get different gradients
   through the patch-embed conv), so the adapter can still learn a
   motion-encoding assignment.
2. **It uses `param_groups`, an opt-in extension of the contract.**
   `DINOv2Embedding.param_groups(base_lr)` puts the pretrained backbone at
   `base_lr * backbone_lr_scale` (default 0.1) and the fresh adapter + head
   at `base_lr`; `NECAlgorithm._build_optimizer` picks it up via `hasattr`.
   This matters because NEC's RMSProp trio (lr=1e-5, α=0.9, ε=0.01) was
   calibrated on `NatureEmbedding`, where every parameter is random init.
   `backbone_lr_scale: 1.0` restores a uniform rate. **`_build_optimizer` is
   shared by `setup()` and `_load_training_state()` on purpose** — resume
   rebuilds the optimizer before `load_state_dict`, which raises on a
   group-count mismatch.
3. **It keeps the standard 4×84×84 NEC env**, unlike the MFEC DINOv2 arm,
   which needs `mspacman_mfec_*_dinov2` (RGB, no GrayScale/Resize) because
   its frozen ViT cannot adapt channels. Same env as `nec/mspacman.yaml`
   means the encoder is the only variable in the ablation, and the frame
   stack — NEC's only velocity signal — is preserved.

**Cost.** The ViT dominates every phase (policy inference at
batch=`num_envs`, episode re-embedding, and `num_updates × batch_size`
gradient updates), and cost is quadratic in the token count, so
`image_size` is the throughput knob: 224 → 16×16 patches, 112 → 8×8, 98 →
7×7 (the smallest that does not downsample an 84×84 frame). Checkpoints are
~177 MB rather than a few MB — 22M ViT params plus RMSProp's `square_avg`.

**Verified** (`tests/test_nec_dinov2_finetune.py`, plus
`tests/test_smoke.py::test_smoke_nec_pong_dinov2_finetune`): with a stub
backbone — adapter init, param-group split and its use by `setup()`,
finetuning through NEC's real `step()`, gradient arrival at the backbone via
the DND kernel term, checkpoint round-trip with groups intact, and the full
`_train()` path from the CLI override. With the **real** `dinov2_vits14`
(opt-in, `NEC_DINOV2_REAL=1`, `NEC_DINOV2_REPO_DIR=` for offline hosts) —
forward shape at every documented `image_size`, a real state_dict loading
with `strict=True` and changing the output, gradients reaching
`blocks.0.attn.qkv` / `patch_embed.proj` / `cls_token`, and NEC
`setup()` → `step()` end-to-end. Also run manually against
`experiment=nec/mspacman_dinov2` on ALE with the real ViT: DND writes,
gradient updates, checkpoint save, and resume all confirmed.

**Not verified**: whether NEC *scores* better with DINOv2 than with
`nature`. No full training run has been completed — that is the experiment,
not a precondition for it. `backbone_lr_scale` is likewise untuned.

#### `clip_finetune` — finetunable CLIP vision tower

`src.networks.CLIPEmbedding` +
`configs/algorithm/embedding_network/clip_finetune.yaml`, bundled as
`experiment/nec/mspacman_clip.yaml`. The NEC counterpart to MFEC's
`encoder_name=clip`. Structurally it is `DINOv2Embedding` with an open_clip
backbone: same mean-replicate channel adapter, same `param_groups` split,
same `freeze_backbone` / `backbone_lr_scale` knobs. Four things are
CLIP-specific and all four are load-bearing:

1. **`open_clip_torch` is an OPTIONAL dependency** (`uv sync --extra clip`).
   `CLIPEmbedding` imports it **lazily, inside `__init__`**. This matters far
   more than it does for MFEC's `clip_encoder.py`: `src/networks.py` is
   imported by `src/algorithms/nec.py` and by every DQN/DDPG/A2C config, so a
   module-scope `import open_clip` would take the *entire repo* down on a
   machine without the extra. Pinned by
   `test_open_clip_is_not_a_module_level_import_of_networks`, which parses the
   AST rather than string-matching.
2. **Only `model.visual` is kept.** The ~63M-param text tower is not merely
   dead weight here the way it is for MFEC — every parameter of an
   `nn.Module` embedding network goes into RMSProp *and into every
   checkpoint*. Vision tower 87.8M, text tower 63.4M (measured, open_clip
   3.3.0).
3. **QuickGELU pairing is a hard error**, exactly as in `CLIPEncoder`:
   `pretrained_tag=openai` requires a `-quickgelu` model name. Verified
   against open_clip 3.3.0 that plain `ViT-B-32` really does use `nn.GELU`
   and `ViT-B-32-quickgelu` really does use `QuickGELU`; a real test asserts
   this so the rule cannot rot silently.
4. **`image_size` must be divisible by the patch size (32).** open_clip does
   NOT check this. Measured: `force_image_size=112` builds and runs, but the
   patch conv yields a 3×3 grid covering 96 of 112 pixels and **silently
   discards 16 px of each axis** — on Ms. Pac-Man that is the score/lives row.
   `_assert_patch_grid_covers_the_image` rejects it and names the valid sizes.
   Skipped for non-ViT towers, which have no patch grid.

Plus one warning rather than an error: **CLIP's RN\* towers carry BatchNorm**
(measured: `RN50` has 55 modules; every ViT-\* has none). NEC keeps the
embedding network in `train()` mode because it is being optimised, so
BatchNorm would use *batch* statistics — and NEC batches differently in each
phase (`num_envs` collecting, `batch_size` in gradient steps, **1** in
`BaseTrainer.evaluate`). The same frame would embed differently in each,
destabilising every DND key. Prefer a ViT tower.

**`normalize_features` does NOT mean what `clip_normalize` means for MFEC.**
There φ *is* the key, so L2-normalising makes MFEC's Euclidean kNN exactly
the cosine kNN CLIP was trained under. Here a learned `Linear(512,
embedding_dim)` head sits in between and NEC normalises the *head's* output,
so the DND metric is cosine in the head's space, not CLIP's. Do not repeat
the MFEC claim for this arm. What it does buy is a unit-norm head input
(pretrained ViT-B-32 emits norm ~10.7).

**Measured at initialisation** (60 real Ms. Pac-Man frames, embeddings as
`NECAlgorithm._embed` produces them, i.e. what the DND kernel sees):

| encoder | mean pairwise L2 | `kernel_delta`=1e-3 as % of mean sq. dist |
|---|---|---|
| `NatureEmbedding` (baseline) | 0.031 | 86% |
| CLIP pretrained | 0.007 | 809% |
| CLIP randomly initialised | 0.001 | ~1.4e5% |

CLIP starts ~4.4× more tightly clustered than the paper's ConvNet, so the
inverse-distance kernel is even closer to a flat average at step 0 than it
already is for the baseline (see `NECAlgorithm._embed`). The cause is
upstream and is not NEC's env: raw pairwise cosine in CLIP space is 0.9935
on the 4×84×84 NEC frames and 0.9949 on the MFEC arm's RGB 210×160 ones.
What the pretraining buys is the residual after the common component is
removed — 7.4% of the norm pretrained against 0.75% random, a 10× gap. If
`train/q_loss` will not descend and `eval/dnd_top_weight` sits at 1/k, try
lowering `kernel_delta` for this arm before concluding CLIP is a bad encoder
— and declare it, because it is a learning-relevant knob.

**Cost.** ViT-B-32's 7×7=49 patch grid at 224 makes the forward comparable to
DINOv2 ViT-S/14's 16×16=256 despite the wider model, but the parameter count
is 4× (87.8M vs 22M). Checkpoints measured at 703 MB without the DND, ~820 MB
with the production `dnd_capacity`; at `checkpoint.save_every_n_steps=50_000`
a 1M-step run writes 20 of them (~16 GB).

**Verified** (`tests/test_nec_clip_finetune.py`, plus
`tests/test_smoke.py::test_smoke_nec_pong_clip_finetune`): with a stub tower —
lazy import, QuickGELU guard, patch-grid guard, BatchNorm warning, text-tower
removal, CLIP-vs-ImageNet stats, whole-frame resize, adapter init, param
groups, finetuning through NEC's `step()`, gradient arrival at the tower, and
checkpoint round-trip. With the **real** `ViT-B-32-quickgelu` (opt-in,
`NEC_CLIP_REAL=1`) — tower size, GELU/QuickGELU pairing, CLIP stats off the
real checkpoint metadata, patch-aligned sizes accepted and 112 rejected,
gradients reaching `conv1` / `positional_embedding` / `class_embedding` /
`transformer.resblocks.0.*` / `proj`, and NEC end-to-end. With the **real
OpenAI pretrained checkpoint** (`CLIP_WEIGHTS=`) — loads and embeds. Also run
manually against `experiment=nec/mspacman_clip` on ALE with the genuine
pretrained tower: DND writes, gradient updates, checkpoint save and resume.

**Not verified**: whether NEC scores better with CLIP than with `nature` or
`dinov2_finetune`. That is the experiment.

#### `mae_finetune` — finetunable MAE ViT

`src.networks.MAEEmbedding` +
`configs/algorithm/embedding_network/mae_finetune.yaml`, bundled as
`experiment/nec/{mspacman,qbert,frostbite}_mae.yaml`. The NEC counterpart to
MFEC's `encoder_name=mae`. Structurally it is `DINOv2Embedding` with a timm
backbone: same mean-replicate channel adapter, same ImageNet stats, same
bilinear whole-frame resize, same `param_groups` split, same `freeze_backbone` /
`backbone_lr_scale` knobs. Four things are MAE-specific:

1. **`timm` is an OPTIONAL dependency** (`uv sync --extra mae`), imported
   **lazily inside `__init__`** for the same reason `CLIPEmbedding` does it:
   `src/networks.py` is imported by `src/algorithms/nec.py` and every
   DQN/DDPG/A2C config, so a module-scope `import timm` would take the entire
   repo down on a machine without the extra. Pinned by
   `test_timm_is_not_a_module_level_import_of_networks`, which parses the AST.
2. **Pooling is over the PATCH tokens, and that is load-bearing.** MAE's CLS
   token is never directly supervised by the reconstruction loss, so
   `pooling: mean` averages the patch tokens with the model's
   `num_prefix_tokens` prefix entries dropped (read, not hardcoded — a ViT with
   register tokens reports more). `pooling: cls` exists to ablate it and is
   expected to score worse. timm's own default for the `.mae` tag is
   `global_pool='token'`, i.e. CLS, and `global_pool='avg'` is **not** the fix —
   it makes `norm` an `nn.Identity` and adds a fresh `fc_norm` the MAE
   checkpoint does not contain. So the module pools from `forward_features`
   itself, exactly as `MAEEncoder` does.
3. **`image_size` defaults to 112, not `MAEEncoder`'s 224.** Measured cost and
   granularity:

   | image_size | tokens | ≈GFLOPs/frame | source px per patch (84×84) | fwd, B=8 |
   |---|---|---|---|---|
   | 224 | 14×14 = 196 | 17.6 | 6 | 1300 ms |
   | 112 | 7×7 = 49 | 4.4 | 12 | 408 ms |

   (float32 CPU, B=8 = `num_envs`. The CLIP arm's ViT-B/32 at 224 measures
   371 ms on the same box: 112 is within 10% of it, 224 would be 3.5×.)

   At 112 this arm has the CLIP arm's token count, patch granularity and
   parameter count (85.7 M vs 87.8 M), so a difference between the two PVM arms
   is attributable to the pretraining objective rather than to compute. Parity
   with the frozen MFEC arm's 224 is not an argument here — the NEC arms already
   see a different input (4×84×84 grayscale vs RGB 210×160) — and the pos-embed
   resample is far cheaper for a finetuned backbone than a frozen one.
   **Verified against timm 1.0.28** that the resample really happens, for the
   hub path *and* a local file: `_builder.load_pretrained` applies `filter_fn`
   after every source branch, and `vision_transformer.checkpoint_filter_fn`
   calls `resample_abs_pos_embed` on a shape mismatch. A local file with the
   upstream `{"model": ...}` wrapper produces a state_dict **identical** to the
   hub's, with `pos_embed` (1, 50, 768) and the CLS position token preserved.
4. **`image_size` must be divisible by 16.** timm does NOT check. Measured:
   `img_size=100` builds and runs, but `patch_embed.grid_size` is 6×6, covering
   96 of 100 pixels and **silently discarding 4 px of each axis** — on Ms.
   Pac-Man the score/lives row. `_assert_patch_grid_covers_the_image` rejects it
   and names the valid sizes, the same guard `CLIPEmbedding` carries for
   open_clip.

No BatchNorm warning is needed (unlike `CLIPEmbedding`): timm ViTs are
LayerNorm-only and `vit_base_patch16_224` is built with `drop_rate=0` /
`drop_path_rate=0`, so the forward is batch-independent even in `train()` mode.
`test_real_vit_has_no_batch_dependent_layers` asserts batched and single-row
forwards agree.

**Measured at initialisation** (60 real Ms. Pac-Man frames from
`mspacman_nec_train`, embeddings as `NECAlgorithm._embed` produces them, mean
over 3 init seeds):

| encoder | mean pairwise L2 | `kernel_delta`=1e-3 as % of mean sq. dist |
|---|---|---|
| `NatureEmbedding` (baseline) | 0.038 | 46% |
| MAE pretrained | 0.008 | 392% |
| MAE randomly initialised | 0.005 | 1299% |

MAE starts ~4.8× more tightly clustered than the paper's ConvNet. Raw pairwise
cosine of the pooled features is **0.9999** at 112 (1.0000 at 224), and the
residual after removing that common component is **0.66 % of the embedding norm
pretrained against 0.44 % at random init — a 1.5× gap, where CLIP's is 10×**.
That is the arm's result at initialisation, before any training: a
reconstruction objective does not organise Atari frames the way a similarity
objective does. `kernel_delta` is deliberately unchanged; see the ablation
section's "tuning one arm".

**Comparability caveat**: re-measuring the CLIP arm with the same script gave
0.030, not the 0.007 recorded in `clip_finetune.yaml` — the tower's own numbers
reproduce exactly (raw norm 10.76 vs the documented ~10.7), so the difference is
downstream of it. Re-measure the CLIP row before putting the two tables side by
side.

**Cost.** 85.7 M parameters. Checkpoints measured at **686 MB** (weights +
RMSProp `square_avg`) and ~801 MB once the DND fills on a 9-action game (the DND
serialises only its filled entries: 9 × 50 000 × 64 f32 = 115 MB at capacity,
230 MB on 18-action Frostbite). ~16 GB per 1M-step run at
`checkpoint.save_every_n_steps=50_000`.

**Verified** (`tests/test_nec_mae_finetune.py`, plus
`tests/test_smoke.py::test_smoke_nec_pong_mae_finetune`): with a stub `timm` —
lazy import, build kwargs (`num_classes=0`, `img_size` forwarded, `global_pool`
never passed), the `pretrained_cfg_overlay` file route, patch-token pooling
against a poisoned-prefix stub, prefix count read not assumed, the patch-grid
guard, ImageNet-vs-CLIP stats, non-persistent normalisation buffers, whole-frame
resize, adapter init, param groups and `setup()`'s use of them, finetuning
through NEC's `step()`, gradient arrival at the backbone through the DND kernel,
checkpoint round-trip with groups intact, `state_dict()` completeness, and the
learning-knob parity of all three experiment configs. With the **real** ViT-B/16
(opt-in, `NEC_MAE_REAL=1`, `pretrained=False` so no network) — 85.7 M-param
backbone, the real 7×7 grid and 50 tokens at 112, batch independence, gradients
reaching `blocks.0.attn.qkv` / `patch_embed.proj` / `cls_token` / `pos_embed`,
and NEC end-to-end. With **real pretrained weights** (`MAE_WEIGHTS=` or
`NEC_MAE_DOWNLOAD=1`) — pos-embed resampling at 112 and 224, and a local file
loading identically to the hub.

**Not verified**: whether NEC scores better with MAE than with `nature`,
`dinov2_finetune` or `clip_finetune`. That is the experiment. No training run
has been completed with it, and `backbone_lr_scale` is untuned like the others.

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
python src/train.py experiment=mfec/rp_gray       # MFEC on Ms. Pac-Man (40M frames, GPU)
python src/train.py experiment=mfec/vae algorithm.vae_checkpoint=<path>  # + paper-exact VAE encoder
python src/train_vae.py                            # pretrain the MFEC "vae" encoder (see above)
pytest tests/test_smoke.py -v
```