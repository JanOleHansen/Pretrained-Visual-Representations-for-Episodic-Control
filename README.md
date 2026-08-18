<div align="center">

# TorchRL Hydra Template

A clean, modular template for deep reinforcement learning research.<br>
Click on [<kbd>Use this template</kbd>](https://github.com/raphaelschwinger/torchrl-hydra-template/generate) to initialize a new repository.

_Suggestions are always welcome!_

</div>

## Philosophy

Reinforcement learning code tends to become monolithic — training loop, environment
setup, network construction, replay buffer, and update rule all tangled together.
This template enforces a hard split into three components, inspired by how
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) structures
deep learning code:

| Component       | Owns                                                                  | Lightning analogy        |
|-----------------|-----------------------------------------------------------------------|--------------------------|
| **Algorithm**   | Everything that affects learning: network, replay buffer, loss, optimiser, exploration, target-net schedule, collector config. **All hyperparameters live here.** | `LightningModule`        |
| **Trainer**     | The loop. Device placement, data collection, logging, callbacks, checkpointing. **No knobs that affect reward.** | `Trainer`                |
| **Environment** | Fixed task definition: env name + transform list. Independent of algorithm. | `LightningDataModule`    |

Two derived rules:

1. **RL algorithm code reads like the paper.** `step()` is short and corresponds to
   the update equations. The DQN file looks like Mnih et al. (2015)'s pseudocode,
   not framework glue.
2. **Anything that influences reward or sample efficiency lives in the algorithm.**
   If a knob shifts the learning curve, it goes on `__init__`. The trainer cannot
   silently change behaviour.

**Implemented experiments:**

| Algorithm | Environment      | Config                          |
|-----------|------------------|---------------------------------|
| DQN       | CartPole-v1      | `experiment=dqn/cartpole`       |
| DQN       | ALE/Pong-v5      | `experiment=dqn/pong`           |
| DDPG      | HalfCheetah-v4   | `experiment=ddpg/halfcheetah`   |
| A2C       | HalfCheetah-v4   | `experiment=a2c/halfcheetah`    |
| MFEC      | ALE/Pong-v5      | `experiment=mfec/pong`          |
| MFEC      | ALE/Breakout-v5  | `experiment=mfec/breakout`      |
| MFEC      | ALE/Qbert-v5     | `experiment=mfec/qbert`         |
| MFEC      | any ALE game     | `experiment=mfec/<encoder> game=<Game>` (see the encoder ablation) |
| NEC       | ALE/Pong-v5      | `experiment=nec/pong`           |
| NEC       | ALE/Hero-v5      | `experiment=nec/hero`           |
| NEC       | ALE/MsPacman-v5  | `experiment=nec/mspacman`       |
| NEC       | ALE/MsPacman-v5  | `experiment=nec/mspacman_dinov2` (finetuned DINOv2 ViT-S/14) |
| NEC       | ALE/MsPacman-v5  | `experiment=nec/mspacman_clip` (finetuned CLIP ViT-B-32) |
| NEC       | ALE/MsPacman-v5  | `experiment=nec/mspacman_mae` (finetuned MAE ViT-B/16) |
| NEC       | ALE/Qbert-v5     | `experiment=nec/qbert{,_dinov2,_clip,_mae}` |
| NEC       | ALE/Frostbite-v5 | `experiment=nec/frostbite{,_dinov2,_clip,_mae}` |

The last four rows are the **NEC encoder ablation**: three games x four
encoders, twelve arms, every learning-relevant setting held identical so that
game and encoder are the only variables. See "The NEC encoder ablation" below.

## Main technologies

**[TorchRL](https://github.com/pytorch/rl)** — A PyTorch-native library for
reinforcement learning that provides modular primitives for environments, replay
buffers, data collectors, and loss modules. It uses
[`TensorDict`](https://github.com/pytorch/tensordict) as a universal data carrier,
making it easy to swap components without rewriting glue code.

**[Hydra](https://github.com/facebookresearch/hydra)** — A configuration framework
that lets you compose hierarchical configs from multiple YAML files and override
any parameter from the command line. Trivial to launch hyperparameter sweeps and
keep every experiment setting version-controlled.

## Quick start

```shell
git clone https://github.com/raphaelschwinger/torchrl-hydra-template
cd torchrl-hydra-template

uv sync
source .venv/bin/activate

python src/train.py experiment=dqn/cartpole
```

`uv sync` covers everything except the `clip` and `mae` arms of the two encoder
ablations, whose backbone packages are opt-in extras: `uv sync --extra clip`
(open_clip — MFEC's `encoder_name=clip` and NEC's `clip_finetune`) and
`uv sync --extra mae` (timm — MFEC's `encoder_name=mae` and NEC's
`mae_finetune`). See "MFEC encoders" and "NEC embedding networks".

A full training run (500k frames, ~7 minutes on CPU) reproduces the torchrl SOTA
reference for DQN-CartPole.

For Atari Pong (mirrors the torchrl SOTA `dqn_atari.py` reference, 40M frames on
GPU):

```shell
python src/train.py experiment=dqn/pong
```

## Architecture

```
train.py  ->  Trainer(algorithm, environment)
                ├── owns: device, env lifecycle, Collector, eval, callbacks, checkpoints
                └── calls: algorithm.step(batch) -> metrics

Algorithm    ->  owns: network, replay buffer, loss, optimiser, exploration,
                       collector config (frames_per_batch, init_random_frames, ...)
               ├── setup(make_env)        — read env specs, build everything
               ├── step(batch)            — anneal eps, store, sample, update
               ├── get_policy()           — greedy policy (eval)
               ├── get_explore_policy()   — eps-greedy policy (collection)
               └── get_collector_config() — frames_per_batch + init_random_frames

Environment  ->  factory: env name + transforms list
               └── make_env(num_envs, device, seed) -> TransformedEnv
```

### Algorithm

The `BaseAlgorithm` API is small:

| Method                    | Purpose                                                           |
|---------------------------|-------------------------------------------------------------------|
| `setup(make_env)`         | Build network, replay buffer, loss, optimiser. Read env specs by calling `make_env()`. |
| `step(batch)`             | Process one batch and return metrics. Where the learning happens. |
| `get_policy()`            | Greedy policy used by `trainer.evaluate()`.                       |
| `get_explore_policy()`    | Exploration policy used by the data collector.                    |
| `get_collector_config()`  | Tells the trainer how to size the `Collector`.                    |
| `reset_eval_metrics()` / `eval_metrics()` | *Optional* (default no-op / `{}`). Called by `evaluate()` around the rollout; whatever `eval_metrics()` returns is merged into the `eval/*` dict. MFEC reports its episodic-memory hit rate here; NEC reports `eval/epsilon` and the shape of its DND kernel (see below). |

`step()` is intentionally unconstrained — the algorithm decides what to do with the
batch. For DQN that means: anneal epsilon, store, skip during warm-up, otherwise
loop `num_updates` of (sample → loss → backward → optimiser → target update).

### Algorithm hyperparameters

Hyperparameters live as **explicit keyword arguments on `__init__`**, not in a
config dataclass:

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        replay_buffer: Callable[[], ReplayBuffer] = default_replay_buffer,
        network: Callable[[tuple[int, ...], int], nn.Module] = default_network,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        num_updates: int = 100,
        hard_update_freq: int = 50,
        ...
    ): ...
```

This buys three things:

1. **Typed defaults** — every hyperparameter has an explicit Python default so the
   algorithm is runnable without any YAML.
2. **Inline documentation** — IDE hover shows you the parameter and its default.
3. **Discoverability** — opening `dqn.py` shows every knob without YAML lookups.

`replay_buffer` and `network` are `Callable` factories rather than scalars because
they encode design decisions (which storage backend, what MLP shape). Their bodies
sit at the top of `dqn.py` as `default_replay_buffer` and `default_network`. To
swap them, edit those functions or pass a different factory in code.

`train.py` unpacks `cfg.algorithm` as `**kwargs`, so YAML values override defaults
and CLI overrides override YAML:

```python
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(device=None, **alg_kwargs)
```

### Environment

Just an env name plus an explicit transforms list:

```yaml
# configs/environment/cartpole.yaml
name: CartPole-v1
transforms:
  - _target_: torchrl.envs.transforms.StepCounter
```

For envs that need extra `GymEnv` constructor arguments (e.g. `frame_skip`,
`from_pixels` for pixel-based Atari), pass them via `gym_kwargs`, and pin the
gym backend with `gym_backend`:

```yaml
# configs/environment/pong_train.yaml
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
  # ...
```

`make_env` in `src/environments/factory.py` instantiates each transform fresh per
call (so stateful transforms like `CatFrames` get independent state), composes
them on top of `GymEnv(name, **gym_kwargs)`, and wraps in `ParallelEnv` when
`num_envs > 1`.

Backends supported: **gymnasium**.

#### Separate evaluation environment

For tasks where training-time observations differ from what evaluation should
see (e.g. Atari, where the SOTA reference clips rewards and ends episodes on
life loss during training but not during eval), declare a second env via the
Hydra package override:

```yaml
# configs/experiment/dqn/pong.yaml
defaults:
  - override /environment: pong_train
  - override /environment@eval_environment: pong_eval
```

When `eval_environment` is set, `BaseTrainer.evaluate()` uses it; otherwise it
falls back to `environment`.

By default `evaluate()` only runs when you invoke `src/eval.py` against a
checkpoint. Set `trainer.eval_every_n_steps` (see "Trainer" below) to also
run it periodically *during* training and get `eval/return_mean` logged
alongside `train/*` in the same run — no separate eval pass needed.

### Environment seeding

`Environment.make_env(num_envs, device, seed)` takes a **master seed for the
env group** and derives one stream per worker from it. The trainer supplies it;
you never pass it by hand. `seed=None` reverts to entropy seeding.

Why this is not just `seed_everything`: `ParallelEnv` uses
`mp_start_method="spawn"`, so each worker is a fresh interpreter whose
`torch`/`numpy`/`random` RNGs start from OS entropy. Two things need seeding
there, and only one is reachable from the parent:

| what | seeded by | reachable from parent? |
|---|---|---|
| the emulator | `env.set_seed(seed)` | yes |
| the worker's **global** RNGs | `seed_everything(seed)` *inside* the worker | no — `Transform` has no `_set_seed` hook |

The second is the load-bearing one. With `repeat_action_probability=0.0` an ALE
reset is fully deterministic, so `NoopResetEnv`'s `torch.randint` draw is the
*only* thing that distinguishes one episode's start state from another's — and
it happens in the worker. `make_env` therefore builds **one `env_fn` per
worker** (a list, not a shared callable) and each seeds its own process.

Consequences worth knowing:

- **The eval env is keyed on the step count**, not a constant
  (`derive_seed(seed, 1, step)`). A fixed seed would make every evaluation
  replay the same episodes, collapsing `eval/return_mean` to one sample.
- **A `num_envs=1` env never re-seeds the parent process.** `evaluate()` builds
  one per eval interval; re-seeding there would reset the trainer's exploration
  stream on every evaluation and make training exploration periodic.
- **`derive_seed` hashes rather than adding an offset.** `base + i` aliases
  across runs — with seeds `42..46` and 4 workers, `(44, worker 1)` and
  `(45, worker 0)` both give 45, so two "independent" seeds replay each other.

Guarded by `tests/test_env_seeding.py`.

### Trainer

`StepTrainer` creates a `torchrl.collectors.Collector` from the algorithm's
collector config and the trainer-level `total_frames`, then iterates:

```python
for batch in self.collector:
    self._step += batch.numel()
    metrics = self.algorithm.step(batch)
    if self._should_log(...):
        fire_callbacks(ON_STEP_END, self.callbacks, metrics=metrics, step=self._step)
```

`BaseTrainer` owns:
- **Device** — resolves `accelerator` + `devices` to `torch.device`, and
  `_env_device` (via `env_worker_device`) for the env itself: `ParallelEnv`
  workers are CPU-only, so with `num_envs > 1` the two differ.
- **Env lifecycle** — creates train/eval envs via `Environment.make_env()`.
  Both are built on `_env_device` so training and evaluation observations go
  through the *same* preprocessing kernels — see "Reproducing MFEC on Atari".
- **Seeding** — every env gets a stream derived from `trainer.seed` via
  `derive_seed` (`src/utils/seeding.py`): training worker *i* from
  `derive_seed(seed, 0, i)`, the eval env from `derive_seed(seed, 1, step)`.
  `seed_everything` alone is not enough — `ParallelEnv` spawns its workers, so
  a worker's global RNGs (which `NoopResetEnv` draws from) are seeded inside
  the worker by `make_env(..., seed=...)`. See "Environment seeding" below.
- **Eval** — `evaluate(num_episodes)` runs the greedy policy.
- **Callbacks** — fires `ON_TRAIN_START`, `ON_STEP_END`, `ON_TRAIN_END` events.
- **Checkpoints** — orchestrates save/load of algorithm state.

Trainer config knobs (`total_frames`, `seed`, `accelerator`, `devices`,
`num_envs`, `log_every_n_steps`, `eval_every_n_steps`, `num_eval_episodes`)
only control how training runs, never what is learned. Set `eval_every_n_steps`
to `null` to disable periodic evaluation (see `EvalCallback` below).

**Shared probe budget.** `configs/train.yaml` and *every* config under
`configs/experiment/` hold these three identical, so any two runs are read off
the same x-axis at the same sample points:

| knob | value |
|------|-------|
| `trainer.total_frames`      | `100_000` agent steps (= 400k raw ALE frames at action repeat 4) |
| `trainer.log_every_n_steps` | `5_000` |
| `trainer.eval_every_n_steps`| `10_000` (10 eval points per run) |

Memory capacity is **pinned equal to `total_frames`** alongside it —
`algorithm.buffer_size: 100_000` for MFEC, `algorithm.replay_buffer.storage.max_size:
100_000` for DQN / DDPG / NEC. That makes the no-eviction bound structural
rather than empirical: a run inserts at most one entry per decision, so
capacity can never bind. **Change the four as a set** — raising `total_frames`
without raising capacity reintroduces eviction mid-run.

This budget is an Atari-100k-style probe, far short of the papers' 40M-frame
runs; scores off it are not comparable to published tables. Override per run:

```bash
python src/train.py experiment=nec/mspacman \
  trainer.total_frames=2_500_000 algorithm.replay_buffer.storage.max_size=2_500_000
```

## Configuration

```
configs/
├── train.yaml              <- top-level defaults (trainer, checkpoint)
├── eval.yaml               <- evaluation defaults
├── train_vae.yaml          <- VAE pretraining defaults (MFEC "vae" encoder)
├── algorithm/
│   ├── dqn.yaml            <- DQN HPs (CartPole defaults)
│   ├── dqn_atari.yaml      <- DQN HPs (Atari/NatureDQN defaults)
│   ├── ddpg.yaml           <- DDPG HPs (HalfCheetah defaults)
│   ├── a2c.yaml            <- A2C HPs (HalfCheetah/MuJoCo defaults)
│   ├── mfec_atari.yaml     <- MFEC HPs (Atari defaults; random projection, QEC)
│   ├── nec.yaml            <- NEC HPs (base defaults; trainable CNN, DND)
│   ├── nec_atari.yaml      <- NEC HPs (Atari defaults per Pritzel et al. 2017 §4;
│   │                          the paper has no hyperparameter table — several
│   │                          values there were swept and never published)
│   └── embedding_network/  <- NEC encoder config group (see "NEC embedding networks")
│       ├── nature.yaml     <- NatureDQN trunk + dense layer (default)
│       ├── dinov2_finetune.yaml <- finetunable DINOv2 ViT (weights_path required)
│       ├── clip_finetune.yaml   <- finetunable CLIP ViT-B-32 (needs the `clip` extra)
│       └── mae_finetune.yaml    <- finetunable MAE ViT-B/16 (needs the `mae` extra)
├── environment/
│   ├── cartpole.yaml       <- env name + transforms
│   ├── pong_train.yaml     <- Pong with EndOfLife + Sign + VecNorm (training)
│   ├── pong_eval.yaml      <- Pong without those transforms (evaluation)
│   ├── breakout_train.yaml <- Breakout training transforms
│   ├── breakout_eval.yaml  <- Breakout eval transforms
│   ├── qbert_train.yaml    <- Q*Bert training transforms
│   ├── qbert_eval.yaml     <- Q*Bert eval transforms
│   ├── mspacman_train.yaml <- Ms. Pac-Man training transforms
│   ├── mspacman_eval.yaml  <- Ms. Pac-Man eval transforms
│   ├── mspacman_train_singleframe.yaml <- same, no CatFrames (paper-exact VAE encoder)
│   ├── mspacman_eval_singleframe.yaml  <- eval counterpart, no CatFrames
│   └── halfcheetah.yaml    <- HalfCheetah-v4 (DoubleToFloat + InitTracker)
├── logger/
│   ├── wandb.yaml
│   └── tensorboard.yaml
├── paths/default.yaml
└── experiment/
    ├── dqn/
    │   ├── cartpole.yaml   <- composed: algorithm + environment + trainer overrides
    │   └── pong.yaml       <- composed Atari Pong experiment
    ├── ddpg/
    │   └── halfcheetah.yaml <- composed DDPG HalfCheetah experiment
    ├── a2c/
    │   └── halfcheetah.yaml <- composed A2C HalfCheetah experiment
    ├── mfec/
    │   ├── pong.yaml       <- MFEC on Pong (40M frames)
    │   ├── breakout.yaml   <- MFEC on Breakout (1M frames)
    │   ├── qbert.yaml      <- MFEC on Q*Bert (40M frames)
    │   ├── mspacman.yaml   <- MFEC on Ms. Pac-Man (paper-faithful; 12.5M decisions = 50M frames)
    │   ├── mspacman_vae.yaml <- same, with the paper-exact VAE encoder
    │   └── mspacman_dinov2.yaml <- same, with a frozen DINOv2 ViT-S/14 as the encoder
    └── nec/
        ├── pong.yaml       <- NEC on Pong (40M raw frames)
        ├── hero.yaml       <- NEC on H.E.R.O. (40M raw frames; unclipped rewards)
        ├── mspacman{,_dinov2,_clip,_mae}.yaml  <- encoder ablation, Ms. Pac-Man (400k raw frames)
        ├── qbert{,_dinov2,_clip,_mae}.yaml     <- encoder ablation, Q*bert (400k raw frames)
        └── frostbite{,_dinov2,_clip,_mae}.yaml <- encoder ablation, Frostbite (400k raw frames)
```

> **Frames vs. agent steps.** Every count in these configs (`total_frames`,
> `frames_per_batch`, `eval_every_n_steps`, `annealing_frames`) is in **agent
> steps**, while the papers count **raw ALE frames**. With action repeat 4
> (`gym_kwargs.frame_skip: 4`) the conversion is `raw = agent_steps * 4`, so
> `total_frames: 100_000` (the shared probe budget) is 400k raw frames, and the
> paper's 40M-frame budget would be `total_frames: 10_000_000`. Note that
> `frame_skip` here *is* the ALE action repeat — TorchRL forwards it into
> `gym.make(..., frameskip=N)` rather than stacking a second repeat on top.

### Override hierarchy

```
Python __init__ defaults  <-  configs/algorithm/dqn.yaml  <-  experiment config  <-  CLI overrides
```

```shell
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3 trainer.total_frames=200_000
```

## Logging

Defaults: plain CLI runs log to **tensorboard**; runs launched via
`experiment=...` log to **wandb**. Override with any combination of `wandb` and
`tensorboard`:

```shell
python src/train.py experiment=dqn/cartpole 'logger=[wandb,tensorboard]'
python src/train.py experiment=dqn/cartpole 'logger=[tensorboard]'
python src/train.py experiment=dqn/cartpole logger=[]
```

### Training metrics are interval means

`train/episode_reward`, `train/episode_length` and `train/q_values` are
averaged over **every** episode finished since the previous logging boundary,
not over the single batch that happened to cross it. `train/episodes_completed`
reports how many episodes each point averages over.

This matters whenever `log_every_n_steps` is much larger than
`frames_per_batch`: at 10,000 vs. 1,024 a single batch contains only ~1-2
finished Ms. Pac-Man episodes, so a per-batch mean is a 1-2 sample estimate of
a quantity with several hundred points of spread — and it throws away ~90% of
the episodes played. If an interval finished no episodes, nothing is emitted
for it (a real gap in the chart) rather than a stale or zero value.

`train/q_values` drops executed-action values above 1e8, which are
optimistic-initialisation sentinels from episodic-control policies (MFEC/NEC
substitute ~1e9 for state-actions their memory cannot yet evaluate) rather than
real returns.

### Paper-ready metrics

These are logged **live, per run** — no post-processing — so the presentation
figures come straight off W&B. Every one is a plain scalar the W&B UI can plot
or put in a run-summary table.

| metric | where | what it is |
|---|---|---|
| `eval/hns` | every eval | **Human-normalised score** `(return_mean − random)/(human − random)`, the primary Atari 100k metric. Puts every game on one axis. Only logged for the 26 Atari 100k games (baselines in `src/utils/atari_scores.py`, from Wang et al. 2016 / SPR Table 3); silently absent otherwise. |
| `eval/value_return_corr` | every eval | **kNN retrieval quality**: Pearson correlation between the value the memory assigns the taken action and the discounted return that action actually earned (over `eval/value_return_n` pooled eval steps). The dependent variable of the encoder comparison — a better representation scores higher. Uses the algorithm's own `gamma`; sentinels and non-finite pairs dropped. |
| `sys/gpu_mem_peak_gb` | every log step | **Peak GPU allocation** (`torch.cuda.max_memory_allocated`, reset at train start). Its last value is the run's peak — the compute-cost axis for the larger PVR encoders. CUDA only. |
| `time/elapsed_min` | every log step | Cumulative wall-clock; its last value is total training time. Alongside the existing `time/speed` (frames/s). |
| `eval/num_episodes` | every eval | Sample size behind each eval point. `eval/return_std` is **omitted** (not `0.0`) at `num_eval_episodes=1`, so a single-episode eval does not read as a zero-variance policy. |

The one thing that is *not* a per-run number is the **cross-game aggregate**
(mean / median / IQM human-normalised score with confidence intervals): each run
is a single game, so aggregating across games is inherently multi-run. Two ways
to get it, cheapest first:

- **In W&B, no script:** group runs by `run.encoder` (or `run.group`) and plot
  the mean of the `eval/hns` summary — a grouped bar of mean HNS per encoder
  across games and seeds. Enough for the talk.
- **Publication-grade:** `scripts/aggregate_results.py` pulls the runs via the
  W&B API and emits mean/median/IQM HNS with 95 % stratified-bootstrap CIs
  (Agarwal et al. 2021), a LaTeX table, a per-encoder cost table, and optional
  learning-curve CSVs. It recomputes `eval/hns` from `eval/return_mean` for runs
  that predate the metric, so **runs finished before this change need no
  re-running**.

  ```shell
  python scripts/aggregate_results.py --entity <you> --project <proj> --plot --curves
  ```

## Callbacks

The trainer fires events at key points:

| Event             | When                  | Receives                            |
|-------------------|-----------------------|-------------------------------------|
| `ON_TRAIN_START`  | Before the loop       | `state: {"cfg": cfg}`               |
| `ON_STEP_END`     | After each logged step| `metrics: dict, step: int`          |
| `ON_TRAIN_END`    | After the loop        | `state: {"cfg": cfg}`               |

Built-in callbacks: `ProgressCallback` (tqdm bar), `CheckpointCallback`,
`EvalCallback` (periodic greedy-policy eval via `eval_environment`, opt-in
with `trainer.eval_every_n_steps`; merges `eval/*` into the same `metrics`
dict the loggers see), `WandBLogger`, `TensorBoardLogger`.

## Reproducing MFEC on Atari

`experiment=mfec/rp_gray` is set up to be read directly against Figure 1 of
Blundell et al. (2016). Getting a comparable curve depends on four things that
a DQN-style Atari config gets wrong for episodic control:

| Setting | DQN-style default | MFEC needs | Why |
|---|---|---|---|
| `repeat_action_probability` | `0.25` (the `ALE/*-v5` default) | `0.0` | Eq. (1) is max-over-returns and never decreases; footnote 1 calls it "not suited to rational action selection in stochastic environments". Sticky actions also collapse `train/exact_hit_rate`, which §4.1 measures at ~50% on Ms. Pac-Man. |
| `SignTransform` | present | **absent** | MFEC stores raw Monte-Carlo returns. Clipping to `{-1,0,+1}` makes a dot (10 pts) worth as much as a ghost (200–1600) or fruit (100–5000), so the policy has no reason to ever use a power pill. |
| `gamma` | `0.99` | `1.0` | §4.1: "The discount rate was set to γ = 1." The 0.99 in §4.2 is the *Labyrinth* setting. |
| `eps_end` | `0.05` | `0.005` | §4.1: "We found that higher exploration rates were not as beneficial, as more exploration makes exploiting what is known harder." |

Plus: the observation is a **single** 84×84 frame (§3, D = 7056), not a
4-frame stack, and eviction is **LRU** (least recently *updated*), not FIFO.

**Units.** `trainer.total_frames` counts agent *decisions*, but the paper's
x-axis counts ALE emulator frames at 4 per decision (§4.1: "An hour of game
play corresponds to approximately 200,000 frames"). So **paper frames = 4 ×
the logged step count**: the shared probe budget of `total_frames: 100_000` is
400k emulator frames, and `total_frames: 12_500_000` covers Figure 1's full
50M-frame range.

`frame_skip` is a trap worth knowing about: `GymEnv._build_env` forwards it to
ALE as `frameskip` and *overrides* the `ALE/*-v5` registry default, so omitting
it yields 1 emulator frame per decision rather than falling back to v5's 4.
Keep `frame_skip: 4` in every Atari config.

The remaining Atari MFEC experiments (`pong`, `breakout`, `qbert`) still use
the DQN-style env configs and have **not** been given the same treatment; they
pick up the corrected `gamma`/`eps` defaults from `mfec_atari.yaml` but still
clip rewards, stack 4 frames and run with sticky actions.

**Evaluation is greedy: `algorithm.eval_eps: 0.0`, `trainer.num_eval_episodes: 1`.**
This reverses an earlier default of ε = 0.005 (Blundell et al. §4.1), which was
there so that `num_eval_episodes` produced more than one distinct sample — MFEC
needs `repeat_action_probability=0.0`, so ALE is deterministic and a greedy
rollout repeats itself. Measured cost of doing it that way, against one QEC
(Ms. Pac-Man, 6 episodes each):

| `eval_eps` | returns | mean | min |
|---|---|---|---|
| **0.000** | `[1440, 1440, 1440, 1440, 1440, 1440]` | **1440** | 1440 |
| 0.005 | `[490, 1440, 870, 550, 1440, 1440]` | 1038 | 490 |

A 1000-decision episode takes ~5 forced random actions at ε = 0.005 and MFEC
cannot absorb even one, so ε at evaluation understates the policy by ~30 %,
leaves `eval/return_max` as the only honest statistic, and pins
`eval/return_min` near the floor **permanently** — it becomes the worst of N
ε-derailments rather than anything about the memory. Nothing is lost by
evaluating greedily: the ε = 0.005 score the paper reports is the *collector's*,
already logged as `train/episode_reward`. `eval/return_std == 0` is therefore
now expected, not a symptom. See `AGENTS.md` for the full argument.

**Evaluation preprocesses observations on the training env's device.** With
`trainer.num_envs > 1` the training envs live in `ParallelEnv` workers, which
are CPU-only (a CUDA context does not survive the spawn) — so the whole
`ToTensorImage → GrayScale → Resize` stack runs on CPU even under
`accelerator: gpu`. `BaseTrainer.evaluate()` therefore builds its eval env on
`self._env_device` (from `env_worker_device`), not on the accelerator, and
moves the tensordict onto the policy device for the forward pass only.

This is not a micro-optimisation. Bilinear-antialias `Resize` is not
bit-identical on CPU and CUDA, and MFEC keys its memory on
`round(embedding * key_scale)`: **1e-7 of drift per pixel already changes ~40 %
of hash keys, 1e-6 changes all of them.** Every lookup then degrades to a
k-neighbour mean, which is near-constant across actions, so the argmax becomes
noise. Measured on Ms. Pac-Man after 60 k decisions (training at 1 858):
no drift → `eval/return_mean` 2 076; 1e-7 → 390; 1e-6 → 292. Building the eval
env on the accelerator produced exactly that collapse — a training curve
climbing past 3 000 next to a flat `eval/return_mean` around 350.

**`eval/exact_hit_rate` is the metric to check first** when `eval/return_mean`
disagrees with `train/episode_reward`. It reports how often the evaluation
policy actually found the query state in the QEC; near 0 means the memory is
not being reached at all (an observation problem), rather than holding a bad
policy. `eval/episode_length` is logged alongside so a low return can be told
apart from an episode that ends early. Note the eval rate counts every
*(state, action)* pair queried while `train/exact_hit_rate` counts only the
action actually taken, so the two differ by roughly `|A|`.

**`eval/exact_minus_knn_value` measures how locked-in the policy is.** Eq. (2)
answers a query with a stored value on an exact match (a max over realised
returns) and a k-neighbour mean otherwise, then `argmax` compares the two. They
are not on the same scale: measured on Ms. Pac-Man over 957 held-out `(s, a)`
pairs, the exact branch is biased **+60** against the true return-to-go
(corr 1.00) while the kNN branch is biased **−411** (corr 0.63) — a **+540** gap
for the *same* `(s, a)`. An action the agent has already taken therefore beats
one it has not on estimator bias alone, and the policy picks an exact-hit action
**99.8 %** of the time although only ~1.8 of 9 actions carry one. That is Eq. (2)
as the paper writes it and has not been changed, but it is why MFEC cannot
recover from an off-trajectory action. The metric is the natural dependent
variable for the encoder ablation: a better representation should shrink it.

**Optimistic initialisation is tie-broken randomly.** The QEC reports `+inf`
for any action **whose whole buffer** holds `<= k` entries — a warm-up device
that stops firing inside the first episode, *not* per-`(state, action)`
optimism, and no help against the gap above.
`QECPolicy.forward` turns those into a large finite value *plus
independent random jitter*, because argmax breaks exact ties by lowest index:
with one shared constant the agent played a single fixed action for whole
episodes (action 0 for 499/500 states on an empty QEC, then action 1 once
action 0 filled) and seeded the memory with degenerate single-action
trajectories. Real Q-values are never perturbed, so a populated QEC evaluates
deterministically. See "Optimistic init must be tie-broken RANDOMLY" in
AGENTS.md.

## Choosing the game: the `game` variable

The MFEC Atari environment configs (`configs/environment/atari_mfec_*.yaml`)
build their env id as `ALE/${game}-v5`, so a game is **one token** and a whole
suite is a sweep rather than a directory of near-duplicate files:

```shell
python src/train.py experiment=mfec/clip game=Assault
python src/train.py -m experiment=mfec/clip game=Assault,BankHeist,RoadRunner
```

**MFEC on Frostbite is this and nothing else** — there is no
`experiment=mfec/frostbite`, and there should not be one. The seven arms below are
game-generic, so the Frostbite ablation is the same seven configs with the token
swapped:

```shell
python src/train.py experiment=mfec/rp_gray game=Frostbite
# the whole 7-arm ablation on the three recommended games (21 runs):
python src/train.py -m \
    experiment=mfec/rp_gray,mfec/rp_rgb,mfec/vae,mfec/dinov2,mfec/resnet,mfec/clip,mfec/mae \
    game=MsPacman,Qbert,Frostbite
```

`run.name` comes out as `mfec_Frostbite_<encoder>_seed42`, so the arms and games
do not collide on disk or in W&B. Note this is the *opposite* layout from the
NEC ablation, whose twelve arms are twelve files (`experiment=nec/frostbite`, …)
because its env configs are per-game — NEC needs `EndOfLifeTransform` and a
frame stack, MFEC does not, so only MFEC's env pair could be made generic.

Nothing else is game-specific: `|A|`, the QEC's shape and the random
projection's input width are all read from the env spec in
`MFECAlgorithm.setup()`. Verified on Assault (7 actions) and BankHeist (18)
without touching a config.

Two consequences worth knowing:

- `run.game` follows `${game}` for these arms, so `run.name` / `run.group`
  stay distinct across a sweep. Without that, every game in a multirun would
  write to the same directory and the same W&B run.
- `buffer_size` is **per action**, so an 18-action game (Frostbite, BankHeist,
  RoadRunner, Jamesbond) allocates twice Ms. Pac-Man's QEC: at the ablation's
  `100_000`, 18 × 100k × 512 × 4 = 3.7 GB. Eviction cannot fire, by an exact
  rule rather than by headroom — capacity is pinned equal to `total_frames`, you
  cannot insert more than `total_frames` entries in total, and they spread over
  `|A|` tables, so the per-action peak *falls* as `|A|` rises. (It was `150_000`
  against a 1M-decision budget, sized off a measured ~72k peak on the busiest
  action; the pin supersedes that.) Raise it in lockstep if you raise
  `total_frames`.

The default is `MsPacman`, and it is written as `${oc.select:game,MsPacman}`
rather than `${game}` on purpose: `scripts/encoder_diagnostics.py` loads these
env configs with a bare `OmegaConf.load` outside Hydra, where a plain `${game}`
would raise.

## The encoder ablation

Seven arms over **two** env pairs, on whichever `game` you select. Everything that is not φ is held equal —
100k decisions, `num_envs: 4`, `eval_every: 10_000`, `buffer_size: 100_000` —
so a between-arm difference is an encoder difference and nothing else.
Pinned by `tests/test_encoder_factory.py`.

The same seven arms cover every game; the recommended set for a joint MFEC/NEC
study is **Ms. Pac-Man, Q\*bert, Frostbite** (see "Choosing the game" above for
the sweep command and why no `mfec/frostbite.yaml` exists).

| experiment | observations | φ | d | role |
|---|---|---|---|---|
| `mfec/rp_gray` | 84×84 **grayscale** | random projection | 64 | **paper baseline** — Blundell et al. §3's exact φ, readable against Figure 1 |
| `mfec/vae` | 84×84 grayscale | frozen ConvVAE | 64 | the paper's *other* φ; first half of C1 |
| `mfec/rp_rgb` | 210×160 **RGB** | random projection | 64 | **encoder control** for the PVM arms |
| `mfec/dinov2` | 210×160 RGB | DINOv2 ViT-S/14 | 384 | self-supervised PVM |
| `mfec/resnet` | 210×160 RGB | ImageNet ResNet-18 | 512 | supervised PVM |
| `mfec/clip` | 210×160 RGB | CLIP ViT-B-32 | 512 | contrastive PVM |
| `mfec/mae` | 210×160 RGB | MAE ViT-B/16 | 768 | **reconstruction** PVM — the only non-similarity objective |

Read it as two independent steps:

- `rp_gray` → `rp_rgb` isolates the **observation** (colour + resolution),
- `rp_rgb` → `dinov2`/`resnet`/`clip`/`mae` isolates the **encoder**.

Within that second step, `mae` is what turns a list of backbones into a study.
ImageNet supervision, DINOv2's self-distillation and CLIP's contrastive loss all
optimise for **semantic discriminability** — as a test of "what does the
pretraining objective do to the geometry of a memory key" they are three
flavours of one answer. MAE's objective is masked *pixel reconstruction*;
nothing in its loss ever compares two images. It is the only arm that varies
that causal variable, and MAE scoring *below* the other PVMs is the hypothesis,
not a defect.

That second step is what C1 actually asks, and it needs the control arm: on
Ms. Pac-Man ghost identity is colour-coded and a *blue* ghost is edible and
worth 200–1600, so at 84×84 grayscale the baseline structurally cannot see what
the PVMs see. Comparing `mspacman` directly against `mspacman_dinov2` would
credit the representation for information the baseline was never given.

The control keeps `state_dim: 64` deliberately. Matching the PVMs' 384/512
would trade one confound for another (encoder vs embedding width);
dimensionality is its own ablation, better done by projecting the PVM features
*down* to 64-d.

## MFEC encoders

MFEC's state embedding is pluggable (`algorithm.encoder_name`):

- `random_projection` (default) — fixed random projection, no pretraining needed.
- `vae` — a frozen convolutional VAE (`src/models/conv_vae.py`), matching
  Blundell et al. 2016 ("Model-Free Episodic Control"), Appendix D exactly:
  a single 84×84 grayscale frame in, embedding = `mean ⊕ log-std` of a
  32-dim latent (64 values total). Because the input is a single frame (not
  this repo's usual 4-frame stack), it needs a "singleframe" environment
  variant — see `experiment/mfec/vae.yaml`. Pretrain a checkpoint
  with `src/train_vae.py` (defaults: 1M random-policy frames, RMSProp,
  lr=1e-5, batch=100, 400,000 steps — also matching the paper), then run:

  ```shell
  python src/train_vae.py
  python src/train.py experiment=mfec/vae \
      algorithm.vae_checkpoint=<checkpoint.save_path printed above>
  ```
- `dinov2` — a frozen DINOv2 ViT (`src/encoders/dino_v2_encoder.py`); see
  `experiment/mfec/dinov2.yaml`.
- `resnet` — a frozen ImageNet ResNet (`src/encoders/resnet_encoder.py`), the
  supervised counterpart to `dinov2`. `state_dim` is fixed by the backbone
  (512 for resnet18/34, 2048 for resnet50+) and read off `fc.in_features`;
  `fc` is replaced with `nn.Identity` so `embed()` returns pooled features.
  Unlike `dinov2_weights_path`, `resnet_weights_path` may be `null` —
  torchvision then downloads `IMAGENET1K_V1` into `~/.cache/torch`, so set a
  path on an offline cluster. See `experiment/mfec/resnet.yaml`.

  The backbone is kept in `eval()` mode, which matters more here than for
  DINOv2: in `train()` mode BatchNorm normalises with *batch* statistics, so
  the same frame would embed differently depending on what else is in the
  batch and the QEC exact-hit path would never fire.
- `clip` — a frozen CLIP vision tower (`src/encoders/clip_encoder.py`), the
  contrastive counterpart to `dinov2` and `resnet`. See
  `experiment/mfec/clip.yaml`.

  **Needs the optional `open_clip_torch` package** — it is a
  `[project.optional-dependencies]` extra, so install it with
  `uv sync --extra clip`. It is imported lazily inside `CLIPEncoder.__init__`,
  so leaving it uninstalled costs nothing to the other encoders. `clip_weights_path` may be
  `null`, in which case open_clip resolves `clip_pretrained_tag` from its hub
  over the network — pass a local checkpoint on an offline cluster, as
  `dinov2_weights` requires.

  Three deliberate differences from the other PVM arms, all following from the
  fact that MFEC uses φ as a **metric**:
  1. It returns the **projected** embedding (`model.visual`, 512-d for
     ViT-B-32/ViT-B-16), not the 768-d pre-projection token `timm`'s CLIP
     entries expose. The projection is the space the contrastive loss was
     computed in — and CLIP is the only one of the three foundation models
     whose pretraining objective *is* a metric on the embedding space.
  2. It **L2-normalises by default** (`clip_normalize: true`). On the unit
     sphere `‖a−b‖² = 2 − 2·cos(a,b)`, so MFEC's Euclidean kNN becomes exactly
     the cosine kNN CLIP was trained under. Set `false` to ablate.
  3. It uses **CLIP's normalisation constants, not ImageNet's** (they are
     different numbers), and **no centre crop** — CLIP's reference pipeline
     would crop a 210×160 Atari frame down to its middle and throw away the
     maze edges and the score row. Distorting the aspect ratio is much cheaper
     than losing pixels when the embedding is a memory key.

  Model names are open_clip's hyphenated spelling (`ViT-B-32`), not OpenAI's
  `ViT-B/32`. `ViT-B-32` is the cheap default: 49 patch tokens against
  `ViT-B-16`'s 196, i.e. ~4× less attention for the same parameter count.

  **The `-quickgelu` suffix is mandatory with the `openai` tag.** OpenAI's CLIP
  was trained with QuickGELU activations; open_clip's plain `ViT-B-32` config
  uses standard GELU. Mismatched, open_clip loads the weights anyway, emits
  only a `UserWarning`, and returns subtly wrong features — the worst possible
  failure for an encoder whose entire value here is its geometry. `CLIPEncoder`
  raises instead. The rule:

  | pretrained tag | model name |
  |---|---|
  | `openai` | `ViT-B-32-quickgelu` |
  | `laion2b_*`, `datacomp_*` | `ViT-B-32` (plain GELU) |

  It applies to a locally-cached OpenAI checkpoint too — the file is right, the
  architecture it is loaded into is what must match.
- `mae` — a frozen MAE ViT-B/16 (`src/encoders/mae_encoder.py`), the
  **reconstruction** counterpart to the three similarity-trained arms and the
  reason the ablation is a study rather than a leaderboard. See
  `experiment/mfec/mae.yaml`.

  **Needs the optional `timm` package** — `uv sync --extra mae`. Imported
  lazily inside `MAEEncoder.__init__`, so leaving it uninstalled costs the
  other encoders nothing. `mae_weights_path` may be `null` (timm pulls
  `vit_base_patch16_224.mae` from the HuggingFace hub); on an offline cluster
  set a path, which is handed to timm as `pretrained_cfg_overlay=dict(file=…)`
  so the original `mae_pretrain_vit_base.pth` release loads as well as a
  timm-format file.

  **`mae_pooling` is the load-bearing knob, and its default is not timm's.**
  MAE's CLS token is never directly supervised — the reconstruction loss is
  computed on patch tokens — so MAE feature evaluation conventionally uses
  global average pooling over the **patch** tokens. `mae_pooling: mean`
  (default) does that, dropping the model's `num_prefix_tokens` prefix entries;
  `cls` ablates it and is expected to score worse. Two traps this avoids:
  timm's own default for the `.mae` tag is `global_pool='token'`, i.e. CLS; and
  "just pass `global_pool='avg'`" would flip timm's `use_fc_norm`, replacing the
  pretrained final LayerNorm with a freshly initialised `fc_norm` the MAE
  checkpoint does not contain. `MAEEncoder` therefore pools from
  `forward_features` itself and never delegates. Pinned by
  `tests/test_mae_encoder.py`.

  **It does not L2-normalise**, unlike `clip`. CLIP normalises because cosine
  is the metric its contrastive loss was computed under; MAE has no metric
  objective, and `dinov2`/`resnet` leave their output raw too — so `clip` stays
  the only arm on the unit sphere and the other three differ only in backbone.

  Two costs to plan for. It is **the slowest arm**: ViT-B/16 at 224 px is 197
  tokens against the CLIP arm's ViT-B/32 50, same depth and width, so ≈4× the φ
  cost per frame, and there is no ViT-B/32 MAE checkpoint to trade down to. And
  at `state_dim: 768` it is the **widest** φ, so the eager QEC allocation is
  0.31 GB per action at `buffer_size: 100_000` — 2.76 GB on 9-action
  Ms. Pac-Man and **5.53 GB on 18-action Frostbite**, alongside the backbone on
  the same device. Do not shrink `buffer_size` for this arm alone; it is held
  equal across the whole grid on purpose.

The RGB encoders (`dinov2`, `resnet`, `clip`, `mae`) share one env pair,
`atari_mfec_train_rgb` / `atari_mfec_eval_rgb`: single RGB frame, no
GrayScale/Resize (each encoder resizes and ImageNet-normalises inside
`embed()`), and otherwise identical to the paper-faithful MFEC stack so the
encoder is the only variable across arms.

> Encoder keyword arguments are threaded `experiment YAML → MFECAlgorithm.__init__
> → make_encoder(...)`, and `MFECAlgorithm.setup()` passes **every** encoder's
> keywords on every call regardless of `encoder_name`. A keyword added for one
> encoder must therefore land in all three places at once, or *every* encoder
> breaks with a `TypeError`, not just the new one.
> `tests/test_encoder_factory.py` pins that contract.

### Vet a new encoder before you train with it

```shell
python scripts/encoder_diagnostics.py --device cuda \
    --dinov2-weights /path/dinov2_vits14_pretrain.pth --resnet --clip --mae
```

Screen encoders here **before** committing GPU-days to them: this needs no
training run and takes minutes, whereas the equivalent evidence from a full
sweep costs days per arm.

MFEC's learning signal depends on two properties of φ that are cheap to check
and expensive to discover after a multi-day run:

1. **Key stability** — Eq. (1)'s max-update fires only on an exact hash match.
   The script reports the match rate between embedding a batch and embedding
   one row at a time; **this must be 1.000**, or training (which embeds
   `num_envs` rows) and `evaluate()` (1 row) disagree about what "the same
   state" is and evaluation silently never takes the exact-match path.
   `RandomProjectionEncoder` accumulates in float64 to guarantee this; a
   float32 ViT does not. Run it on `--device cuda`: the failure is a
   cuBLAS shape-dependent-kernel effect and does **not** reproduce on CPU.
2. **Discriminability** — the kNN averages the k nearest stored returns, so an
   encoder that maps visually different game states to nearly the same point
   makes Q(s, a) state-independent. The script reports an adjacency AUC
   (can φ tell near-identical frames from unrelated ones?) and the relative
   contrast of nearest-neighbour vs. mean distance.

The random-projection baseline scores AUC ≈ 0.92 with perfect key stability;
use it as the reference when judging another encoder's numbers.

**Measured result (CUDA, 150 Ms. Pac-Man frames):** every float32 encoder fails
key stability, and it is not fixable by tuning:

| encoder | dim | key b/s | adj AUC | rel contr |
|---|---|---|---|---|
| `random_projection` | 64 | **1.000** | 0.955 | 0.627 |
| `dinov2` ViT-S/14 | 384 | **0.000** | 0.991 | 0.697 |
| `resnet` resnet18 | 512 | **0.000** | 0.991 | 0.690 |
| `clip` ViT-B-32-quickgelu | 512 | **0.000** | 0.990 | 0.728 |

(`mae` is not in this table because it was added after the measurement and has
not been run on a GPU yet — run `--mae` on the training card before trusting it.
Expect `key b/s = 0.000` for the same reason as the other float32 ViTs; that is
documented, not a defect.)

cuBLAS picks float32 GEMM kernels by batch size, so `φ(x)` in a 16-row batch
differs from `φ(x)` alone in the last bits, and a key survives only if all `d`
coordinates escape rounding — probability `(1 − 2·drift·key_scale)^d`, i.e.
~1e-18 at `d=384`. Training is unaffected (one batch shape throughout); at
evaluation the exact-hash path is replaced by the near-exact rescue, which
resolves to the same entry and returns the same value. Read
`eval/memory_hit_rate`, not `eval/exact_hit_rate`, when comparing encoders.
Details and the reason lowering `key_scale` is *not* the fix are in AGENTS.md.

Note the flip side of the same table: all three PVMs beat the random projection
on both discriminability columns, with CLIP best on relative contrast.

## NEC embedding networks

MFEC's encoder is **frozen**; NEC's is **trained end-to-end**, so the two have
separate systems that should not be merged. NEC's φ is a Hydra config group,
`configs/algorithm/embedding_network/`, listed in the defaults of both
`configs/algorithm/nec.yaml` and `nec_atari.yaml`:

| option | factory | status |
|---|---|---|
| `nature` (default) | `src.networks.NatureEmbedding` — NatureDQN conv trunk + one dense layer to `embedding_dim` | the paper's network; used by every `experiment/nec/*.yaml` |
| `dinov2_finetune` | `src.networks.DINOv2Embedding` — DINOv2 ViT-S/14, backbone **not** frozen | the self-supervised PVM arm; bundled as `experiment/nec/{mspacman,qbert,frostbite}_dinov2.yaml` (see below) |
| `clip_finetune` | `src.networks.CLIPEmbedding` — CLIP ViT-B-32 vision tower, **not** frozen | the contrastive PVM arm; bundled as `experiment/nec/{mspacman,qbert,frostbite}_clip.yaml`. Needs `uv sync --extra clip` (see below) |
| `mae_finetune` | `src.networks.MAEEmbedding` — MAE ViT-B/16, **not** frozen | the **reconstruction** PVM arm — the only one whose objective is not a similarity objective; bundled as `experiment/nec/{mspacman,qbert,frostbite}_mae.yaml`. Needs `uv sync --extra mae` (see below) |

```shell
# Standard encoder (this is what every experiment/nec/*.yaml already does):
python src/train.py experiment=nec/pong
python src/train.py experiment=nec/pong algorithm/embedding_network=nature   # explicit

# Swap in another one — no YAML editing:
python src/train.py experiment=nec/pong algorithm/embedding_network=<name>
```

### The NEC encoder ablation — three games x four encoders

The twelve bundled arms are `experiment=nec/{mspacman,qbert,frostbite}` (the
paper's ConvNet), `..._dinov2`, `..._clip` and `..._mae`. Each one is a complete
config; they do not compose with each other, which is why the shared settings
are repeated verbatim in all twelve files.

The three PVM arms are not three flavours of one idea. `dinov2_finetune` and
`clip_finetune` both optimise a **similarity** objective (self-distilled view
agreement, image-text cosine); `mae_finetune`'s optimises masked **pixel
reconstruction**, and nothing in that loss ever compares two images. It is the
arm that turns a list of backbones into a test of what the pretraining
objective does to the geometry of an episodic-memory key.

| game | env pair | \|A\| | raw-frame repeat | MFEC `exact_hit_rate` |
|---|---|---|---|---|
| Ms. Pac-Man | `mspacman_nec_{train,eval}` | 9 | 22 % | 0.53 |
| Q*bert | `qbert_nec_{train,eval}` | 6 | 13 % | 0.60 |
| Frostbite | `frostbite_nec_{train,eval}` | 18 | 22 % | 0.41 |

The three games are the intersection of "MFEC demonstrably works" (measured — see
AGENTS.md, *Which games MFEC can work on at all*) and "in the Atari-100k 26", so
the same set also serves an MFEC comparison. All six `*_nec_*` env configs drop
`SignTransform` (NEC does not clip rewards), drop `VecNorm`, disable v5 sticky
actions, and cap episodes at the full 27,000 agent steps (30 min).

Held identical across all twelve: `total_frames: 100_000` agent steps (= 400k
raw frames, the shared probe budget), `num_updates: 100`, `eps_end: 0.001`,
`annealing_frames: 10_000`, `init_random_frames: 4_800`, `eval_eps: 0.005`,
`num_envs: 8`, `num_eval_episodes: 5`, `log_every_n_steps: 5_000`,
`eval_every_n_steps: 10_000`, `seed: 42`. **Do not tune a single arm in
isolation**; a change that belongs to the comparison has to land in all twelve.
`tests/test_nec_ablation_parity.py` enforces that.

Two of those were wrong until recently, and both are worth knowing about if you
are reading older runs:

* **The exploration schedule did not follow the budget down.**
  `annealing_frames: 50_000` and `init_random_frames: 12_500` were sized for a
  1M-step run and survived the cut to 100k, where they became 50% and 12.5% of
  it — **26.6% of every collected frame took a uniform random action**, and the
  policy was only near-greedy for the second half of the run. Worse,
  `nec/pong.yaml` and `nec/hero.yaml` carry no `algorithm:` block and inherited
  `nec_atari.yaml`'s paper-scale defaults (`annealing_frames: 4_000_000`,
  `init_random_frames: 50_000`), so ε reached only 0.975 by step 100k — those
  two were essentially random-play runs. Both are now sized against the probe
  budget; see AGENTS.md, *Shared probe budget*, for the arithmetic and the
  measured random-policy episode lengths behind the floor.
* **`num_envs` is not a free resource knob.** The ConvNet arms ran 16 against
  the ViT arms' 8. NEC writes to the DND only at episode end, so frames left in
  a trailing partial episode are never written — a loss proportional to
  `num_envs` — and MFEC measured that 16 envs keep 39% fewer unique states than
  4 at equal frames. The baseline was being handicapped against the encoders it
  is compared to. All twelve now run 8 (and `num_eval_episodes: 5`, which was
  10 on the ConvNet arms and gave them a different standard error).

```shell
# One arm:
python src/train.py experiment=nec/frostbite

# The whole game sweep for one encoder, five seeds each:
python src/train.py -m experiment=nec/mspacman,nec/qbert,nec/frostbite \
    trainer.seed=42,43,44,45,46
```

Run directories are `nec_{game}_{encoder}_seed{n}`, and `run.game` / `run.encoder`
are set explicitly in every file so a multirun cannot collide.

### The contract

A new embedding network is any callable

```python
factory(obs_shape: Sequence[int], embedding_dim: int, **kwargs) -> nn.Module
```

(everything after the two positional args keyword-only, so a Hydra
`_partial_` can pre-bind design kwargs) whose module maps
`(B, *obs_shape) -> (B, embedding_dim)` float32. **All parameters must be
trainable by default** — `NECAlgorithm.setup()` hands
`embedding_net.parameters()` straight to the optimizer, so a frozen
parameter is a dead one. The output must not be pre-normalised: NEC
L2-normalises before every DND read/write, and that normalisation is what
keeps the inverse-distance kernel from collapsing (see
`tests/test_nec_kernel_scale.py`).

Optionally, a module may define `param_groups(base_lr) -> list[dict]`;
`NECAlgorithm._build_optimizer` then passes those to RMSProp instead of the
flat parameter list, which is how a module splits itself across learning
rates (`DINOv2Embedding` uses it for its pretrained trunk). Modules without
the attribute are unaffected.

The contract is written up as `src.networks.NECEmbeddingNetwork` (a
documentation `Protocol`); AGENTS.md § "Adding a new NEC embedding network"
has the step-by-step, including how to persist extra checkpoint state via
`_get_training_state` / `_load_training_state`.

### `dinov2_finetune` — a finetuned DINOv2 ViT

`DINOv2Embedding` runs a pretrained DINOv2 ViT-S/14 as NEC's φ and trains it
with the DND: torch.hub architecture + a local `.pth`, a 1×1 conv channel
adapter, bilinear resize to `image_size`, ImageNet normalisation, then a
linear head to `embedding_dim`. Deliberately **without** the
`requires_grad_(False)` freeze MFEC's `DINOv2Encoder` applies — that freeze
exists because MFEC's QEC hash needs a bit-exact φ, and NEC has no such
requirement.

```shell
# Bundled experiment (Ms. Pac-Man, same env pair as experiment=nec/mspacman):
python src/train.py experiment=nec/mspacman_dinov2 \
    algorithm.embedding_network.weights_path=/path/dinov2_vits14_pretrain.pth

# Or as a group override on any NEC game:
python src/train.py experiment=nec/pong \
    algorithm/embedding_network=dinov2_finetune \
    algorithm.embedding_network.weights_path=/path/dinov2_vits14_pretrain.pth \
    run.encoder=dinov2          # otherwise the run dir collides with the nature arm
```

Three things worth knowing before running it:

- **The channel adapter starts as grayscale→RGB, not random.** Atari gives 4
  stacked grayscale frames and the ViT wants 3; the adapter is initialised to
  `weight = 1/C, bias = 0`, so its output is the frame-stack mean replicated
  to R=G=B — a valid image in `[0, 1]`, which is what the ImageNet
  normalisation assumes. With a default `nn.Conv2d` init the ViT's first
  forward sees out-of-distribution input and the pretrained weights buy
  nothing until the adapter has itself been learned. It is still trainable,
  so it can learn to encode motion across channels; it just does not *start*
  from noise.
- **The backbone gets its own learning rate.** `backbone_lr_scale` (default
  0.1) scales the pretrained trunk relative to the freshly-initialised
  adapter + head, which stay at `algorithm.lr`. NEC's RMSProp settings were
  calibrated on `NatureEmbedding`, where every parameter is random init.
  Set it to `1.0` for a single uniform rate. **It has not been tuned.**
- **`image_size` is the throughput knob.** Cost is quadratic in the token
  count: 224 → 16×16 patches, 112 → 8×8, 98 → 7×7 (the smallest multiple of
  14 that does not downsample an 84×84 frame). The default is 224 for parity
  with MFEC's frozen arm. Checkpoints are ~177 MB rather than a few MB.

Unlike the MFEC DINOv2 arm — which needs the RGB `mspacman_mfec_*_dinov2`
env pair because a frozen ViT cannot adapt channels — this keeps the standard
4×84×84 NEC env, so the encoder is the only variable against
`experiment=nec/mspacman` and NEC keeps the frame stack it gets velocity from.

Tested in `tests/test_nec_dinov2_finetune.py` against a stub backbone
(adapter init, param groups, finetuning through NEC's `step()`, gradient
arrival at the backbone, checkpoint round-trip) and, opt-in via
`NEC_DINOV2_REAL=1`, against the genuine ViT-S/14 (real state_dict loading,
every documented `image_size`, gradients into the transformer blocks, NEC
end-to-end). **What is not tested is whether it scores better than `nature`**
— no full training run has been completed. That is the experiment.

### `clip_finetune` — a finetuned CLIP vision tower

`CLIPEmbedding` runs a pretrained CLIP ViT-B-32 as NEC's φ and trains it with
the DND. Structurally it is `DINOv2Embedding` with an open_clip backbone: the
same mean-replicate channel adapter, the same `param_groups` split, the same
`freeze_backbone` / `backbone_lr_scale` knobs.

**Needs the optional `open_clip_torch` extra:**

```shell
uv sync --extra clip

# Bundled experiment (Ms. Pac-Man, same env pair as experiment=nec/mspacman):
python src/train.py experiment=nec/mspacman_clip

# Or as a group override on any NEC game:
python src/train.py experiment=nec/pong \
    algorithm/embedding_network=clip_finetune run.encoder=clip
```

`open_clip` is imported **lazily**, inside `CLIPEmbedding.__init__` — without
the extra every other arm still runs untouched.

Four CLIP-specific things, all of which will bite silently if ignored:

- **QuickGELU pairing is a hard error.** OpenAI's CLIP was trained with
  QuickGELU; open_clip's plain `ViT-B-32` config uses standard GELU. open_clip
  loads the mismatch with only a `UserWarning` and then returns subtly wrong
  features. Use `ViT-B-32-quickgelu` with `pretrained_tag: openai`, and the
  plain name with `laion2b_*` / `datacomp_*`.
- **`image_size` must be divisible by the patch size (32).** open_clip does not
  check. `force_image_size=112` builds and runs, but the patch conv tiles only
  96 pixels and **silently discards 16 px of each axis** — on Ms. Pac-Man that
  is the score/lives row. `CLIPEmbedding` rejects it and names the valid sizes.
- **Prefer a ViT tower.** CLIP's `RN*` towers carry BatchNorm (RN50 has 55
  modules). NEC trains the encoder, so it stays in `train()` mode and BatchNorm
  would use batch statistics — and NEC batches differently when collecting
  (`num_envs`), when updating (`batch_size`) and when evaluating (1). The
  module warns.
- **`normalize_features` is not MFEC's `clip_normalize`.** For MFEC, φ *is* the
  memory key, so normalising makes its Euclidean kNN exactly the cosine kNN
  CLIP was trained under. Here a learned `Linear(512, embedding_dim)` head sits
  in between and NEC normalises the head's output, so the DND's metric is
  cosine in the *head's* space. Do not repeat the MFEC claim for this arm.

**Expect a slow start.** Measured on 60 real Ms. Pac-Man frames, as the DND
kernel sees them at initialisation:

| encoder | mean pairwise L2 | `kernel_delta`=1e-3 as % of mean sq. dist |
|---|---|---|
| `NatureEmbedding` (baseline) | 0.031 | 86% |
| CLIP pretrained | 0.007 | 809% |
| CLIP randomly initialised | 0.001 | ~1.4e5% |

Every Atari frame is "a screenshot of Pac-Man" to a contrastive image-text
model, so raw pairwise cosine in CLIP space is 0.9935 (and 0.9949 on the MFEC
arm's RGB frames — this is CLIP, not NEC's grayscale env). The pretraining's
contribution lives in the residual after that common component: 7.4% of the
embedding norm, against 0.75% at random init — a 10× gap, so the weights are
doing real work. But the kernel starts flatter than the baseline's. If
`train/q_loss` will not descend and `eval/dnd_top_weight` sits at `1/k`, try
lowering `kernel_delta` for this arm — and declare it, since it changes
learning.

Checkpoints are ~820 MB (87.8M-param tower, 4× DINOv2 ViT-S/14, plus RMSProp
state and the DND), so ~16 GB per 1M-step run at the default checkpoint
interval.

Tested in `tests/test_nec_clip_finetune.py` against a stub tower, opt-in via
`NEC_CLIP_REAL=1` against the genuine `ViT-B-32-quickgelu`, and via
`CLIP_WEIGHTS=` against the real OpenAI checkpoint. **Whether it beats
`nature` or `dinov2_finetune` is untested** — that is the experiment.

### `mae_finetune` — a finetuned MAE ViT

`MAEEmbedding` runs a pretrained MAE ViT-B/16 as NEC's φ and trains it with the
DND. Structurally it is `DINOv2Embedding` with a timm backbone: the same
mean-replicate channel adapter, the same ImageNet statistics and bilinear
whole-frame resize, the same `param_groups` split, the same `freeze_backbone` /
`backbone_lr_scale` knobs.

**Why this arm exists.** `dinov2_finetune` and `clip_finetune` both optimise a
*similarity* objective, so as a study of what pretraining does to the geometry
of a memory key they give two flavours of one answer. MAE (He et al. 2022) masks
~75 % of the patches and regresses the missing **pixels** — nothing in that loss
ever compares two images. It is the control that says whether NEC's DND needs a
representation trained to place similar things nearby, or merely one that
retains information. **MAE scoring below the other PVMs is the hypothesis, not
a defect.**

**Needs the optional `timm` extra:**

```shell
uv sync --extra mae

# Bundled experiment (Ms. Pac-Man, same env pair as experiment=nec/mspacman):
python src/train.py experiment=nec/mspacman_mae

# Or as a group override on any NEC game:
python src/train.py experiment=nec/pong \
    algorithm/embedding_network=mae_finetune run.encoder=mae
```

`timm` is imported **lazily**, inside `MAEEmbedding.__init__` — without the
extra every other arm still runs untouched. `weights_path: null` pulls
`vit_base_patch16_224.mae` from the HuggingFace hub; on an offline node point it
at a local file, which timm's `checkpoint_filter_fn` accepts in either the
upstream `mae_pretrain_vit_base.pth` layout or its own `.safetensors`.

Three MAE-specific things:

- **Pooling is the load-bearing knob, and its default is not timm's.** MAE's CLS
  token is never directly supervised by the reconstruction loss, so MAE feature
  evaluation conventionally averages the **patch** tokens. `pooling: mean` does
  that (prefix tokens dropped via `num_prefix_tokens`); `pooling: cls` ablates
  it and is expected to score worse. timm's own default for the `.mae` tag is
  CLS, and `global_pool='avg'` is *not* the fix — it replaces the pretrained
  final LayerNorm with a fresh `fc_norm` the checkpoint does not contain, so the
  module pools from `forward_features` itself.
- **`image_size` defaults to 112, not 224.** Token count is
  `(image_size/16)²`, so 112 gives a 7×7 = 49-token grid: the same token count,
  the same 12-source-pixels-per-patch granularity and near-identical parameter
  count (85.7 M) as the CLIP arm's ViT-B/32 at 224. At 224 (196 tokens) MAE
  would get ~4× the CLIP arm's compute and any win would be confounded with
  that. timm resamples `pos_embed` for the smaller grid — verified for the hub
  *and* the local-file path — and a finetuned backbone adapts to it. The cost is
  resolution: a patch covers 12 source pixels rather than 6.
- **It must be divisible by 16.** timm does not check: `img_size=100` builds and
  runs, but the patch conv tiles only 96 pixels and **silently discards 4 px of
  each axis**. `MAEEmbedding` rejects it and names the valid sizes.

**Expect a slow start — more so than CLIP.** Measured on 60 real Ms. Pac-Man
frames, embeddings as `NECAlgorithm._embed` produces them, averaged over 3 init
seeds:

| encoder | mean pairwise L2 | `kernel_delta`=1e-3 as % of mean sq. dist |
|---|---|---|
| `NatureEmbedding` (baseline) | 0.038 | 46% |
| MAE pretrained | 0.008 | 392% |
| MAE randomly initialised | 0.005 | 1299% |

MAE starts ~4.8× more tightly clustered than the paper's ConvNet. Raw pairwise
cosine of the pooled features is **0.9999** at `image_size=112` (1.0000 at 224),
and the residual left after removing that common component is 0.66 % of the
embedding norm pretrained against 0.44 % at random init — a 1.5× gap, where
CLIP's is 10×. That is this arm's headline result at initialisation: masked
pixel reconstruction does not organise Atari frames the way a similarity
objective does. `kernel_delta` is deliberately **not** changed for this arm; if
it needs to move it moves for all twelve, and gets declared.

Checkpoints are 686 MB (85.7 M params plus RMSProp state) and ~801 MB once the
DND fills, so ~16 GB per 1M-step run at the default checkpoint interval.

Tested in `tests/test_nec_mae_finetune.py` against a stub `timm` (pooling,
patch-grid guard, adapter init, param groups, finetuning through NEC's `step()`,
gradient arrival at the backbone, checkpoint round-trip) and, opt-in via
`NEC_MAE_REAL=1`, against the genuine ViT-B/16 (real patch grid, batch
independence, gradients into the transformer blocks, NEC end-to-end), plus a
pretrained-weights tier (`MAE_WEIGHTS=` or `NEC_MAE_DOWNLOAD=1`) that pins the
`pos_embed` resampling and the local-file path. **Whether it beats `nature`,
`dinov2_finetune` or `clip_finetune` is untested** — that is the experiment.

## Reading a NEC run

When `eval/return_mean` disagrees with `train/episode_reward`, check these
before touching a hyperparameter. NEC has no exact-match shortcut on its read
path (unlike MFEC), so it reports the *shape* of the DND kernel instead of a
hit rate.

| Metric | What a bad value means |
|---|---|
| `eval/epsilon` | Compare with `train/epsilon`. `eval_eps` (in `nec_atari.yaml`) and `eps_end` (per-experiment) are independent knobs, so a run can train one policy and score another. `setup()` warns when they differ by more than 10x. |
| `eval/dnd_top_weight` | Share of kernel mass on the nearest neighbour. **Do not read this against `1/k`**: with k=50 in 64-d even a strong retriever sits at ~0.03 (a raw-pixel k-NN reference measures 0.029). Watch the trend, not the level; a *higher* value is not automatically better. |
| `eval/dnd_nn_dist` | Mean L2 to the nearest stored key. Embeddings are unit-norm so this is bounded by 2; drifting upward means stored keys go stale faster than `dnd_key_lr` refreshes them. |
| `eval/dnd_optimistic_rate` | Fraction of *(state, action)* pairs still answered with the `+inf` sentinel. Above 0 late in a run means a starved action is capturing the argmax. |
| `train/dnd_blend_rate` | **Not** expected near 0 on Atari. Duplicate frames (the opening freeze, the pause after each death) are 17.6 % of a Ms. Pac-Man rollout, and they blend legitimately. 0.1–0.5 is normal here. |
| `train/updates` | Should equal `num_updates` (400 for action-repeat-4 Atari: one update per 16 raw frames, per paper §4). A lower flat line means the run was launched with an override and is under-trained per frame. |

A gap where `eval/return_mean` is well below `train/episode_reward` **at the
same `eval/episode_length`** is a scoring-rate gap, not a survival gap, and
points at the policy being evaluated rather than at the environment. The two
Ms. Pac-Man env configs and `BaseTrainer.evaluate`'s rollout loop are verified
byte-identical to a plain `env.rollout`, and `NoopResetEnv` alone produces
exactly zero return variance on this game — so non-zero `eval/return_std` is
proof that evaluation ran with a real ε.

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py`. Define `default_*` factories for design
   choices (network, buffer) and put scalar HPs as keyword args on `__init__`.
2. Implement `setup(make_env)`, `step(batch)`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Add `configs/algorithm/my_algo.yaml` mirroring scalar defaults from `__init__`.
4. Add `configs/experiment/my_algo/<env>.yaml` composing your algorithm + env.
5. Add a smoke test in `tests/test_smoke.py`.
6. Update `README.md` and `AGENTS.md`.

## Smoke test

```shell
pytest tests/test_smoke.py -v
```

Loads the experiment config, applies minimal-frame overrides, and asserts that
one full training cycle runs without error.

## Acknowledgements

This project builds on the ideas pioneered by
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by
@ashleve and further refined in
[yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
by @gorodnitskiy. Their work on combining structured Hydra configs with clean
training pipelines served as the foundation; this template adapts that philosophy
to the reinforcement learning setting with TorchRL.

The DQN reference implementation in `src/algorithms/dqn.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/dqn/dqn_cartpole.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_cartpole.py).
The DDPG reference implementation in `src/algorithms/ddpg.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/ddpg/ddpg.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py).
The A2C reference implementation in `src/algorithms/a2c.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/a2c/a2c_mujoco.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/a2c/a2c_mujoco.py).