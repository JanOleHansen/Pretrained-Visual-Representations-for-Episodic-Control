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
    """Logs training metrics to TensorBoard.

    Args:
        log_dir: directory where TensorBoard event files are written
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer = None

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

        cfg = state.get("cfg")
        if cfg is None:
            return
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        hparams = _flatten(config_dict)          # -> {"trainer.total_frames": 500000, ...}
        # run_name="." keeps torch's internal hparams SummaryWriter in *this* log_dir.
        # The default (str(time.time())) opens a second writer in a nested subdir with
        # its own event file, which TensorBoard lists as a duplicate run.
        self._writer.add_hparams(hparams, {}, run_name=".")

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=step)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


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