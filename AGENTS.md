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
| MFEC      | ALE/MsPacman-v5  | `experiment=mfec/mspacman`     |
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

Set `trainer.eval_every_n_steps` (opt-in; `null` by default in
`configs/train.yaml`) to also run evaluation periodically during training and
have `eval/return_mean` land in the normal training logs (TensorBoard/W&B)
next to `train/*`, with no separate `src/eval.py` invocation needed:

```yaml
# configs/experiment/nec/pong.yaml
trainer:
  eval_every_n_steps: 200_000   # paper §4 cadence for MFEC/NEC; baselines use 1_000_000
  num_eval_episodes: 5
```

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
                              no Q-head; default), DINOv2Embedding (finetunable
                              ViT; SCAFFOLDING, see "Adding a new NEC embedding
                              network")
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
  models/
    conv_vae.py              — ConvVAE (Blundell et al. 2016 App. D architecture);
                              vae_loss/gaussian_nll/kl_diag_gaussian used by train_vae.py
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: gymnasium + transforms list + gym_kwargs/gym_backend
  trainers/
    BaseTrainer.py          — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    StepTrainer.py          — StepTrainer (Collector-driven loop)
  callbacks/                — ProgressCallback, CheckpointCallback, EvalCallback,
                              WandBLogger, TensorBoardLogger
  utils/                    — device resolution, seeding, callback builders
configs/
  algorithm/dqn.yaml        — DQN HPs (CartPole defaults); _partial_ replay_buffer + network
  algorithm/dqn_atari.yaml  — DQN HPs (Atari/NatureDQN defaults; pixel obs)
  algorithm/ddpg.yaml       — DDPG HPs (HalfCheetah defaults); _partial_ actor/critic/noise
  algorithm/a2c.yaml        — A2C HPs (HalfCheetah/MuJoCo defaults); _partial_ actor/value
  algorithm/mfec_atari.yaml — MFEC HPs (Blundell et al. 2016 §4.1 Atari defaults: buffer_size=1M,
                              k=11, state_dim=64, gamma=1.0, constant eps=0.005;
                              encoder_name=random_projection, vae_checkpoint=null, seed=null).
                              NOTE §4.2's Labyrinth settings differ (k=50, gamma=0.99)
  algorithm/nec.yaml        — NEC HPs (base defaults); defaults-lists the
                              embedding_network config group + _partial_ replay_buffer
  algorithm/nec_atari.yaml  — NEC HPs (Atari defaults per Pritzel et al. 2017 §4;
                              the paper has NO hyperparameter table — see "NEC" below
                              for which values it states vs. swept-and-unreported)
  algorithm/embedding_network/  — NEC encoder config group (swap with
                              `algorithm/embedding_network=<name>`)
    nature.yaml             — NatureDQN trunk + dense layer (DEFAULT; the paper's net)
    dinov2_finetune.yaml    — finetunable DINOv2 ViT (SCAFFOLDING, unvalidated)
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
  environment/mspacman_mfec_train.yaml — ALE/MsPacman-v5, paper-faithful MFEC stack: single frame,
                              no SignTransform, repeat_action_probability=0.0 (see "MFEC on Atari" below)
  environment/mspacman_mfec_eval.yaml  — eval counterpart (identical by design; nothing to strip)
  environment/mspacman_mfec_train_dinov2.yaml — the mspacman_mfec_train stack with GrayScale/Resize
                              dropped (DINOv2 needs 3 channels and resizes internally); task
                              settings held identical so encoder is the only variable
  environment/mspacman_mfec_eval_dinov2.yaml  — eval counterpart (identical by design)
  environment/mspacman_mfec_train_rgb.yaml — the generalised version of the two above, for ANY
                              RGB PVM (dinov2, resnet, ...); byte-equivalent to the _dinov2
                              pair, which is kept only so existing runs stay reproducible.
                              Prefer this one for new encoder arms.
  environment/mspacman_mfec_eval_rgb.yaml  — eval counterpart (identical by design)
  environment/mspacman_nec_train.yaml — ALE/MsPacman-v5 for NEC: action-repeat 4, NO
                              SignTransform (paper §4 names Ms. Pac-Man as a game where
                              NEC's lack of reward clipping is what produces the result)
  environment/mspacman_nec_eval.yaml  — eval counterpart (drops EndOfLife)
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
  experiment/mfec/mspacman.yaml — MFEC on Ms. Pac-Man, paper-comparable (12.5M decisions =
                              50M emulator frames = Figure 1's full x-range; num_envs=4)
  experiment/mfec/mspacman_vae.yaml — same, with encoder_name=vae + singleframe env
                              (vae_checkpoint is required, no default — see "Encoders" above)
  experiment/mfec/mspacman_dinov2.yaml — same, with encoder_name=dinov2 (frozen ViT-S/14,
                              state_dim=384) + the mspacman_mfec_*_dinov2 env pair.
                              dinov2_weights is required and has no usable default — the
                              checked-in path is cluster-local.  NOTE: this config used the
                              DQN-style mspacman_train_dinov2 env pair until Aug 2026; runs
                              logged before that carry reward clipping, sticky actions and a
                              4,500-step cap, and are NOT comparable to mspacman.yaml.
  experiment/mfec/mspacman_resnet.yaml — same, with encoder_name=resnet (frozen ImageNet
                              resnet18, state_dim=512) + the mspacman_mfec_*_rgb env pair.
                              resnet_weights_path may be null (torchvision downloads
                              IMAGENET1K_V1); set a path on an offline cluster.
  experiment/nec/pong.yaml     — NEC on Pong (10M agent steps = 40M raw frames, num_envs=16);
                              keeps the clipped env — Pong's rewards are already in [-1, 1]
                              so SignTransform is a no-op there, not a deviation
  experiment/nec/hero.yaml     — NEC on H.E.R.O. (10M agent steps = 40M raw frames);
                              uses hero_nec_train (unclipped)
  experiment/nec/mspacman.yaml — NEC on Ms. Pac-Man (10M agent steps = 40M raw frames,
                              num_envs=8); uses mspacman_nec_train (unclipped)
  logger/{wandb,tensorboard}.yaml
  paths/default.yaml
  train.yaml, eval.yaml, train_vae.yaml
tests/
  test_smoke.py             — DQN-on-CartPole, DQN-on-Pong, DDPG-on-HalfCheetah, A2C-on-HalfCheetah, MFEC-on-Pong, NEC-on-Pong smoke tests
  test_mfec_encoder_refactor.py — encoder-abstraction transparency: setup() wiring, embed()
                              shape/determinism, forward(), deepcopy sharing, checkpoint round-trip
  test_nec_embedding_network.py — NEC embedding-network config group: shape/dtype contract,
                              gradient flow (proves the encoder is genuinely trainable),
                              Hydra-composition architecture regression, config-swap
                              setup()+step() end-to-end, DINOv2 scaffolding (stub backbone)
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
| `mspacman_mfec_train.yaml` | ✗ | MFEC Ms. Pac-Man (paper-faithful; see "MFEC on Atari" below) |
| `mspacman_mfec_train_dinov2.yaml` | ✗ | MFEC Ms. Pac-Man + frozen DINOv2 encoder (paper-faithful) |

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

The exception is the `mspacman_mfec_*` pair, which has no `SignTransform` at
all (see below): there `train/episode_reward` and `eval/return_mean` measure
the same thing, so a gap between them is a real train/eval discrepancy rather
than a units mismatch.

## MFEC on Atari — what a DQN-style env config gets wrong

Reusing a DQN Atari transform stack for MFEC quietly breaks the algorithm.
`mspacman_mfec_train.yaml` / `mspacman_mfec_eval.yaml` are the corrected
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
approximately 200,000 frames"). So **paper frames = 4 × logged step**, and
`total_frames: 12_500_000` covers Figure 1's full 50M-frame range.

### QEC eviction is LRU, not FIFO

§2: "we limit the size of the table by removing the least recently *updated*
entry". `QEC._key_to_slot[a]` is an `OrderedDict` kept in
least-recently-updated-first order, so it doubles as the LRU queue:
`popitem(last=False)` picks the victim and `QEC.touch()` refreshes an entry
after an Eq. (1) max-update. A kNN **read** deliberately does not refresh
recency. A FIFO ring buffer would evict the oldest *insertions*, which on
Atari are exactly the early-level states re-visited every episode.

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
legal — torchvision downloads `IMAGENET1K_V1`). The RGB backbones share the
`mspacman_mfec_train_rgb` / `mspacman_mfec_eval_rgb` env pair.

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

### Evaluation must keep ε — a deterministic eval policy is a bug

`get_policy()` ends in `_EvalEGreedyModule` (constant `eval_eps`, default 0.005
= Blundell et al. §4.1), **not** a bare argmax. Two independent reasons:

1. **`num_eval_episodes` silently collapses to one sample.** MFEC requires
   `repeat_action_probability=0.0` (Eq. 1 is max-over-returns — footnote 1), so
   ALE is deterministic, and Ms. Pac-Man's opening is insensitive to
   `NoopResetEnv`. A deterministic policy therefore replays the same trajectory
   every episode: `eval/return_std` is identically 0 and
   `eval/return_min == eval/return_mean == eval/return_max`. That signature in a
   run's charts means this bug is back.
2. **Pure argmax is not the policy being trained.** QEC values are optimistic
   (Eq. 1 never decreases), so exploiting them with no ε walks into states whose
   stored value came from one lucky ε-greedy trajectory. The paper reports the
   ε = 0.005 policy's score; it never evaluates a pure argmax.

Measured on Ms. Pac-Man against one identical QEC:

| policy | exploration type | mean | std |
|---|---|---|---|
| `get_policy()` (old, bare argmax) | MODE | 380.0 | **0.000** (5/5 identical) |
| `get_explore_policy()` | RANDOM | 448.0 | 248.4 |
| `get_explore_policy()` | MODE | 380.0 | **0.000** |

**The third row is the trap.** torchrl's `EGreedyModule.forward` is gated on
`exploration_type() in (ExplorationType.RANDOM, None)`, and
`BaseTrainer.evaluate()` runs under `ExplorationType.MODE` — so dropping a stock
`EGreedyModule` into the eval chain does *nothing*. `_EvalEGreedyModule`
subclasses it and forces `ExplorationType.RANDOM` inside `forward`.

Do not "fix" this by changing `BaseTrainer.evaluate()`'s `set_exploration_type`:
MODE is correct for DQN/A2C, whose eval policies genuinely are deterministic.
Exploration is an algorithm concern, so the opt-out lives in the algorithm.
`eval_eps: 0.0` restores the old deterministic behaviour. `NECAlgorithm` returns
a bare greedy chain from `get_policy()` and has the same latent issue; it has not
been changed. Guarded by `tests/test_mfec_eval_policy.py`.

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
- `NECPolicy` (`src/algorithms/nec.py`) still has the un-jittered
  `torch.where(isinf, 1e9, ...)` form and the same latent defect. It has not
  been changed, because the failure was only confirmed experimentally for MFEC.

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
  `experiment/mfec/mspacman_vae.yaml`, which composes them with
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
  python src/train.py experiment=mfec/mspacman_vae \
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
   blend into the wrong slot. Moved slots are recorded and delisted in bulk by
   `flush_moved_slots()`, called once per collector batch from `step()` and
   reported as `train/dnd_delisted`.

Setting `dnd_key_lr=dnd_value_lr=0` reproduces the old frozen-DND behaviour
bit-for-bit, so the change can be A/B'd against earlier runs.

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

### 4b. Eviction is still FIFO, and that is now a cost decision

§3.1 specifies LRU. The old rationale for FIFO ("keys go stale, so oldest ==
stalest") died with §1 — keys are refreshed now. It is left as FIFO because
**eviction policy has no effect until a table is full**: at
`dnd_capacity=5e5` and ~178 inserts per action per batch that is ~2800
batches (~4.5M agent steps), and below that the two policies are
bit-identical. Switching means reworking the ring-buffer serialisation
(`__getstate__`/`__setstate__` rotate by `_write_ptrs`) plus recency
bookkeeping on the hottest path. Do it before a full 40M-frame run.

### 5. The exact-match blend rule is largely inert in practice

`write_batch` blends `Q_i ← Q_i + α(G − Q_i)` only on a bit-level hash match
of the quantised 64-d embedding. The CNN takes `num_updates` (100–400) RMSProp
steps between successive `step()` calls, so a state re-encountered in a later
batch essentially never re-hashes to its stored key. Blends therefore only
happen between duplicate frames embedded *within one* `step()` call, and the
DND behaves close to an insert-only log.

Now that gradients also move stored keys (§1), a hash entry additionally goes
stale the moment its key is updated — `flush_moved_slots()` delists those, so
the blend rate falls further rather than silently blending into a slot whose
key has changed.

`train/dnd_blend_rate` measures this directly; expect it near 0. This is a
consequence of porting MFEC's exact-hash design onto a *moving* embedding.
Making the rule fire would mean matching within a radius rather than exactly
— a design change needing empirical validation, deliberately **not** done.

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
at the start of each step() call, via `NECAlgorithm._embed()` — the single
place the L2 normalisation lives, chunked at `_EMBED_CHUNK` frames so a
4500-step episode does not go through the CNN in one pass.

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
| `dinov2_finetune` | `src.networks.DINOv2Embedding` | **scaffolding, unvalidated** — see below |

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
  `_get_training_state` / `_load_training_state`.

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
6. **Add tests** to `tests/test_nec_embedding_network.py` (shape/dtype,
   all-params-trainable, YAML composes + instantiates). If the network needs
   downloaded weights, monkeypatch the loader to a stub backbone so CI stays
   offline — `tests/test_dinov2_encoder.py` and the `_StubViT` fixture in
   `test_nec_embedding_network.py` show the pattern.
7. **Update `README.md` and `AGENTS.md`** (this table).

#### `dinov2_finetune` — scaffolding, not a finished feature

`src.networks.DINOv2Embedding` +
`configs/algorithm/embedding_network/dinov2_finetune.yaml` scaffold a
finetunable DINOv2 ViT: torch.hub architecture load + local `.pth`
state_dict (same pattern as `src/encoders/dino_v2_encoder.py`), a 1×1 conv
channel adapter, bilinear resize to `image_size`, ImageNet normalisation,
ViT, then `nn.Linear(embed_dim, embedding_dim)`. Unlike the MFEC encoder it
does **not** freeze the backbone; `freeze_backbone: bool = False` is exposed
to toggle later.

**Verified** (against a stub backbone, no downloads): the YAML composes, the
factory builds, forward shape/dtype, and `freeze_backbone` gating.
**Not verified**: that real `dinov2_vits14` weights load through this path,
that 84×84 → 224 upsampled Atari frames are sensible ViT input, or that NEC
learns anything with it. No training run has used it. Validate before
drawing any conclusion from a run that selects it.

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
python src/train.py experiment=mfec/mspacman       # MFEC on Ms. Pac-Man (40M frames, GPU)
python src/train.py experiment=mfec/mspacman_vae algorithm.vae_checkpoint=<path>  # + paper-exact VAE encoder
python src/train_vae.py                            # pretrain the MFEC "vae" encoder (see above)
pytest tests/test_smoke.py -v
```