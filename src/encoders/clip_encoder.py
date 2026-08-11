"""Frozen CLIP vision tower as MFEC's φ (pretrained visual representation).

Why CLIP is the most interesting PVR for episodic control
---------------------------------------------------------
MFEC's entire decision rule is "which stored key is nearest" — φ is used as a
*metric*, not as network input.  Of the three foundation models in the encoder
ablation, CLIP is the only one whose **pretraining objective is itself a metric
on the embedding space**: the contrastive image-text loss directly optimises
cosine similarity between the projected image embedding and its caption.
DINOv2's self-distillation and MAE's pixel reconstruction are not metric
objectives.  So CLIP is the a-priori strongest candidate, and this encoder is
deliberately built to hand MFEC *that* space rather than an intermediate one:

* it returns the **projected** embedding (``model.visual`` output, 512-d for
  ViT-B/16 and ViT-B/32), not the 768-d pre-projection pooled token that
  ``timm``'s CLIP entries expose — the projection is what the contrastive loss
  was computed on;
* it **L2-normalises by default** (``normalize=True``).  On the unit sphere
  ``||a - b||^2 = 2 - 2 cos(a, b)``, so MFEC's Euclidean kNN becomes exactly
  cosine kNN — the similarity CLIP was trained with.  Turn it off
  (``clip_normalize: false``) to ablate that choice.

Preprocessing notes (both deliberate, both differ from stock CLIP)
------------------------------------------------------------------
* **CLIP's own normalisation constants, not ImageNet's.**  They are different
  numbers and stock CLIP inference uses the former; ``ResNetEncoder`` and
  ``DINOv2Encoder`` correctly use ImageNet's because those backbones were
  trained with them.  Resolved from the model when open_clip exposes it, else
  from the OpenAI constants below.
* **No centre crop.**  CLIP's reference pipeline is Resize(short side) +
  CenterCrop(224), which on a 210x160 Atari frame would cut away the left and
  right of the maze — and, on Ms. Pac-Man, the score row that makes states
  distinguishable at all.  This resizes the whole frame to a square instead,
  accepting the aspect-ratio distortion.  Losing pixels is much worse than
  distorting them when the embedding is a memory key.

Key stability
-------------
Like DINOv2 this is a **float32 ViT**, so it carries no guarantee that the
QEC hash key survives a change of batch shape (training embeds ``num_envs``
rows, ``BaseTrainer.evaluate`` embeds 1).  ``RandomProjectionEncoder``
accumulates in float64 precisely to avoid this; a ViT cannot.  Run

    python scripts/encoder_diagnostics.py --clip --device cuda

**on the GPU you will train on** before trusting a run: the ``key b/s`` column
must read 1.000.  See "Exact-match keys must be invariant to batch shape" in
AGENTS.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Encoder

#: OpenAI CLIP's normalisation statistics — NOT ImageNet's.  Used only when the
#: installed open_clip does not expose them on the model (older versions).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_INSTALL_HINT = (
    "CLIPEncoder needs the `open_clip_torch` package, which is not installed.\n"
    "    uv add open_clip_torch        (or: pip install open_clip_torch)\n"
    "open_clip is used rather than timm because it exposes the *projected* "
    "image embedding — the space CLIP's contrastive objective actually "
    "operates on, which is the whole reason to try CLIP as an episodic-memory "
    "key."
)


class CLIPEncoder(Encoder):
    """Frozen CLIP vision tower. ``embed(obs) -> (B, d)`` projected features.

    Parameters
    ----------
    weights_path
        Local open_clip checkpoint (``open_clip_pytorch_model.bin`` or a
        ``.pt``).  ``None`` lets open_clip resolve ``pretrained_tag`` from its
        hub, which **requires network access** — pass a path on an offline
        cluster, exactly as ``DINOv2Encoder`` does.
    model_name
        An open_clip architecture name, e.g. ``"ViT-B-32"``, ``"ViT-B-16"``,
        ``"ViT-L-14"``.  Note open_clip spells these with hyphens, unlike
        OpenAI's ``"ViT-B/32"``.
    pretrained_tag
        open_clip tag used only when ``weights_path`` is ``None`` (e.g.
        ``"openai"``, ``"laion2b_s34b_b79k"``).  Ignored otherwise.
    image_size
        ``None`` (default) uses the backbone's native size — the safe choice.
        Any other value is passed to open_clip as ``force_image_size``, which
        interpolates the positional embeddings; it is a deviation from the
        pretraining resolution, so only set it deliberately.
    normalize
        L2-normalise the embedding (default ``True``). See module docstring.
    interpolation
        Resize mode. Defaults to ``"bicubic"``, which is what CLIP's own
        preprocessing uses. Set ``"bilinear"`` to match ``DINOv2Encoder`` and
        ``ResNetEncoder`` exactly if you want resizing held constant across
        every arm of the ablation.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        model_name: str = "ViT-B-32",
        pretrained_tag: str | None = "openai",
        image_size: int | None = None,
        device: torch.device | None = None,
        normalize: bool = True,
        interpolation: str = "bicubic",
        image_mean: tuple[float, float, float] | None = None,
        image_std: tuple[float, float, float] | None = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(_INSTALL_HINT) from exc

        self.model_name = model_name
        self.normalize = bool(normalize)
        self.interpolation = interpolation

        # A local file wins over the hub tag; that is the offline-cluster path.
        pretrained = weights_path if weights_path is not None else pretrained_tag

        build_kwargs: dict = {"pretrained": pretrained, "precision": "fp32"}
        if image_size is not None:
            # Only pass it when asked: older open_clip releases lack the kwarg,
            # and passing it needlessly would break them for no benefit.
            build_kwargs["force_image_size"] = image_size

        try:
            model = open_clip.create_model_and_transforms(
                model_name, **build_kwargs
            )[0]
        except TypeError:
            if image_size is None:
                raise
            raise TypeError(
                f"The installed open_clip does not support force_image_size; "
                f"leave clip_image_size unset to use {model_name}'s native "
                f"resolution, or upgrade open_clip_torch."
            )

        # `visual` is the vision tower *including* the projection into the
        # joint image-text space — open_clip's `CLIP.encode_image` is literally
        # `self.visual(image)`.  Keeping only this half drops the ~63 M-param
        # text tower, which is dead weight for an episodic-memory key.
        visual = getattr(model, "visual", None)
        if visual is None:                               # pragma: no cover
            raise RuntimeError(
                f"open_clip model {model_name!r} has no `.visual` attribute; "
                "cannot isolate the vision tower."
            )
        self.model = visual

        # Native (or forced) input resolution. open_clip stores an int or a
        # (H, W) pair depending on version and architecture.
        native = getattr(visual, "image_size", image_size or 224)
        if isinstance(native, (tuple, list)):
            native = native[0]
        self.image_size = int(native)

        # CLIP's constants, preferring whatever the checkpoint declares.
        mean = image_mean or getattr(visual, "image_mean", None) or _CLIP_MEAN
        std = image_std or getattr(visual, "image_std", None) or _CLIP_STD
        self._mean = torch.tensor(tuple(mean)).view(1, 3, 1, 1)
        self._std = torch.tensor(tuple(std)).view(1, 3, 1, 1)

        # eval() matters for the same reason it does in ResNetEncoder: any
        # batch-statistic or stochastic layer left in training mode makes the
        # same frame embed differently depending on its batch, and the QEC
        # exact-hit path never fires.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.model.to(device)

        # Width is read from a real forward rather than an attribute: it is the
        # ground truth, it costs one image, and `output_dim` is not present on
        # every open_clip vision tower (TimmModel wrappers, older releases).
        self.state_dim = self._probe_state_dim()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _probe_state_dim(self) -> int:
        dev = next(self.model.parameters()).device
        probe = torch.zeros(1, 3, self.image_size, self.image_size, device=dev)
        out = self.model(probe)
        if out.ndim != 2:                                # pragma: no cover
            raise RuntimeError(
                f"CLIP vision tower returned shape {tuple(out.shape)}; "
                "expected (B, d). MFEC needs one vector per observation."
            )
        return int(out.shape[-1])

    # ------------------------------------------------------------------
    # Encoder contract
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        dev = obs.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)

        x = obs.float().reshape(-1, *obs.shape[-3:])     # (..., C, H, W) -> (B, C, H, W)
        if x.shape[-2:] != (self.image_size, self.image_size):
            # align_corners is only valid for bilinear/bicubic; both are.
            x = F.interpolate(
                x,
                (self.image_size, self.image_size),
                mode=self.interpolation,
                align_corners=False,
            )
        x = (x - self._mean.to(dev)) / self._std.to(dev)

        out = self.model(x).float()
        if self.normalize:
            # Makes MFEC's Euclidean kNN equal to cosine kNN, which is the
            # metric CLIP was trained under. Also pins ||phi|| to 1.0, so the
            # QEC's relative near-exact tolerance becomes 3e-5 * 2 = 6e-5 —
            # tighter than the random projection's 9.8e-5, i.e. conservative.
            out = F.normalize(out, dim=-1)
        return out

    def state(self) -> dict:
        return {"model_state_dict": self.model.state_dict()}

    def load_state(self, state: dict) -> None:
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
