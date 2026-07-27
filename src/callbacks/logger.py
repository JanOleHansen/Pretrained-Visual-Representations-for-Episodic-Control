from __future__ import annotations

from typing import Any


class WandBLogger:
    """Logs training metrics to Weights & Biases.

    Args:
        project: W&B project name
        entity: W&B entity (team/user). None uses the default from wandb login.
        name: run name. None lets W&B generate one.
        tags: list of tags to attach to the run
        mode: "online", "offline", or "disabled"
    """

    def __init__(
        self,
        project: str = "torchrl-hydra-template",
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        save_dir: str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.mode = mode
        self.save_dir = save_dir
        self._run = None

    def on_train_start(self, state: dict[str, Any]) -> None:
        import wandb
        from omegaconf import OmegaConf

        if self.save_dir is not None:
            from pathlib import Path
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        cfg = state.get("cfg")
        config_dict = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            mode=self.mode,
            config=config_dict,
            dir=self.save_dir,
        )

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._run is not None:
            import wandb
            wandb.finish()
            self._run = None


class TensorBoardLogger:
    """Logs training metrics, the full config, and hyperparameters to TensorBoard.

    Three surfaces:
      * SCALARS  -- per-step metrics from ``on_step_end``.
      * TEXT     -- the complete resolved config as YAML, written once at start
                    under the ``config`` tag. Nothing is filtered out here.
      * HPARAMS  -- a flattened, scalar-only view of the config plus the final
                    metric values, written once at the end so the table can be
                    sorted by result. Point TensorBoard at the parent directory
                    (``logs/train/runs``) to compare runs in one table.

    Args:
        log_dir: directory where TensorBoard event files are written
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer = None
        self._config_dict: dict | None = None
        self._last_metrics: dict[str, float] = {}
        self._last_step: int = 0

    def on_train_start(self, state: dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter
        from omegaconf import OmegaConf

        # Run directories are deterministic (see hydra.run.dir in configs/train.yaml),
        # so on resume this dir already holds an event file. purge_step tells
        # TensorBoard to drop events at or beyond the restart step, so the interval
        # replayed between the last checkpoint and the crash isn't drawn twice.
        resume_step = int(state.get("step") or 0)
        self._writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=resume_step or None,
        )
        self._last_step = resume_step

        cfg = state.get("cfg")
        if cfg is None:
            return

        # TEXT tab: the entire config, verbatim. Written at start so it survives
        # a crashed run. resolve=True expands ${...} interpolations to real values.
        self._config_dict = OmegaConf.to_container(cfg, resolve=True)
        self._writer.add_text(
            "config",
            _as_markdown_code(OmegaConf.to_yaml(cfg, resolve=True)),
            global_step=resume_step,
        )

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=step)
                self._last_metrics[key] = float(value)
        self._last_step = step

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._writer is None:
            return

        # HPARAMS tab: written last so the table carries the run's final metrics
        # and can be sorted by them. run_name="." keeps torch's internal hparams
        # SummaryWriter in *this* log_dir; the default (str(time.time())) opens a
        # second writer in a nested subdir, which TensorBoard lists as a duplicate run.
        if self._config_dict is not None:
            hparams = _hparams(self._config_dict)
            metrics = {
                k: v for k, v in self._last_metrics.items()
                if k.startswith(("train/", "eval/"))
            }
            self._writer.add_hparams(
                hparams,
                metrics,
                run_name=".",
                global_step=self._last_step,
            )

        self._writer.flush()
        self._writer.close()
        self._writer = None


# Keys whose flattened value is a stringified list/target and therefore useless
# as an HParams column. They remain fully visible in the TEXT tab.
_HPARAM_SKIP = ("transforms", "_target_", "_partial_")


def _hparams(config_dict: dict) -> dict:
    """Flatten the config to the scalar subset worth showing as HParams columns."""
    return {
        key: value
        for key, value in _flatten(config_dict).items()
        if not any(skip in key for skip in _HPARAM_SKIP)
    }


def _as_markdown_code(text: str) -> str:
    """Indent every line by 4 spaces -> a Markdown code block.

    TensorBoard's text plugin renders Markdown with only the ``tables`` extension
    enabled, so triple-backtick fences are not recognised and would show up
    literally. A 4-space indented block is core Markdown and renders verbatim,
    preserving the YAML indentation.
    """
    return "\n".join("    " + line for line in text.splitlines())


def _flatten(d, prefix="") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, f"{key}."))
            elif isinstance(v, (int, float, bool, str)):
                out[key] = v
            else:                      # lists (num_cells), None, etc. -> stringify
                out[key] = str(v)
        return out