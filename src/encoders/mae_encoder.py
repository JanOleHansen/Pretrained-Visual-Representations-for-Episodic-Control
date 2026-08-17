"""Frozen MAE (Masked Autoencoder) ViT as MFEC's φ (pretrained visual representation).

Why MAE is in the ablation at all
---------------------------------
The other three PVR arms all optimise, one way or another, for **semantic
discriminability**: ``ResNetEncoder`` for ImageNet class labels, ``DINOv2Encoder``
for self-distilled view agreement, ``CLIPEncoder`` for image-text cosine
similarity.  Read as a study of "what does the pretraining objective do to the
geometry of an episodic-memory key", those three do not vary the causal
variable — they are three flavours of the same answer.

MAE is the one arm whose objective is **not** a similarity objective at all:
He et al. (2022) mask ~75% of the patches and regress the missing *pixels*.
Nothing in that loss ever compares two images.  So MAE is the control that
tells you whether MFEC's kNN benefits from a representation trained to place
similar things nearby, or merely from a representation that retains
information.  Its expected position is *below* the other PVRs; that is the
hypothesis, not a defect.

POOLING — the choice that decides whether this arm measures MAE
---------------------------------------------------------------
**MAE's CLS token is effectively undertrained.**  It is not supervised by
anything: the reconstruction loss is computed on patch tokens, and the CLS token
participates only indirectly through self-attention.  The MAE paper's own
feature evaluations, and every downstream recipe that follows it, use **global
average pooling over the patch tokens** (He et al. 2022, §4 — the fine-tuning
recipe sets ``global_pool=True``).  Taking the CLS token here would produce a
weak embedding and the result would be written up as a property of MAE, when it
is really a property of the pooling.

So this encoder:

* defaults to ``pooling="mean"`` — mean over the **patch** tokens, with the
  prefix tokens dropped;
* keeps ``pooling="cls"`` available, because "CLS is bad for MAE" is worth being
  able to *demonstrate* rather than assert;
* reads the prefix-token count off ``model.num_prefix_tokens`` rather than
  assuming 1.  It is 1 for ``vit_base_patch16_224.mae`` (CLS only), but timm
  ViTs with register tokens report more, and slicing off a hardcoded 1 would
  silently average a register token into the embedding.

**The pooling is done here, from ``forward_features``, not delegated to timm.**
Two independent reasons, both verified against timm's source and pinned by
``tests/test_mae_encoder.py::test_the_timm_default_pooling_is_token_not_avg``:

1. **timm's default for this tag is ``global_pool='token'``, i.e. CLS.**  The
   ``vit_base_patch16_224.mae`` entry in timm's ``default_cfgs`` sets
   ``num_classes=0`` but says nothing about pooling, so
   ``VisionTransformer.__init__``'s own default (``'token'``) applies.  Relying
   on the default would silently give this arm the undertrained CLS token.
2. **Passing ``global_pool='avg'`` to fix that is a trap.**  In
   ``VisionTransformer.__init__``, ``use_fc_norm`` defaults to
   ``global_pool == 'avg'``, which makes ``self.norm`` an ``nn.Identity`` and
   ``self.fc_norm`` a real LayerNorm — but the MAE pretrain checkpoint contains
   ``norm.*`` and no ``fc_norm.*``.  The pretrained final LayerNorm would be
   dropped and replaced by a freshly initialised one, and ``forward_features``
   would return un-normalised tokens.  Building with the default and pooling by
   hand keeps the pretrained ``norm`` in the forward path.

NOT L2-normalised — deliberately
---------------------------------
``CLIPEncoder`` L2-normalises because cosine similarity is the metric CLIP's
contrastive loss was computed under, so normalising puts MFEC's Euclidean kNN
onto CLIP's own metric.  **MAE has no metric objective**, so there is no
corresponding argument here, and inventing one would add a second difference
between this arm and its neighbours.  ``DINOv2Encoder`` and ``ResNetEncoder``
do not normalise either; leaving MAE unnormalised holds that choice constant
across all three non-CLIP PVR arms, so ``clip`` remains the *only* arm whose
embedding lives on the unit sphere.

Preprocessing
-------------
* **ImageNet mean/std**, which is what MAE was pretrained with (unlike CLIP,
  which has its own constants).
* **Bilinear resize**, matching ``DINOv2Encoder`` and ``ResNetEncoder`` rather
  than CLIP's bicubic — again, one fewer thing varying across the non-CLIP arms.
* **No centre crop.**  Resizing a 210x160 Atari frame to a square distorts the
  aspect ratio; a centre crop would instead delete the left/right of the maze
  and the score row.  Losing pixels is much worse than distorting them when the
  embedding is a memory key.

Cost — this is the most expensive arm in the study
---------------------------------------------------
``vit_base_patch16_224`` at 224 px is **196 patch tokens + 1 CLS = 197**, against
the CLIP arm's ViT-B/32 **49 + 1 = 50**.  Same depth and width (12 layers,
d=768), so the transformer cost is ~4x the CLIP arm's per frame, and φ is this
pipeline's bottleneck.  ``state_dim=768`` also makes the eager QEC allocation
1.5x the 512-d arms' (see "QEC memory is sized by state_dim" in AGENTS.md).
There is no ViT-B/32 MAE checkpoint to fall back on; ``image_size=112`` would
cut the token count to 49 but interpolates the positional embeddings away from
the pretraining resolution, so 224 is the default and the cost is accepted.

Key stability
-------------
Like DINOv2 and CLIP this is a **float32 ViT**, so the QEC hash key is not
guaranteed to survive a change of batch shape on CUDA and — measured for every
other float32 arm — does not.  ``key b/s`` reading 0.000 in

    python scripts/encoder_diagnostics.py --mae --device cuda

is expected, is not fixable by lowering ``key_scale``, and costs only the O(1)
lookup path at evaluation (the near-exact rescue returns the same value).  See
"MEASURED: every float32 encoder fails key stability on CUDA" in AGENTS.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Encoder

#: ImageNet statistics — what MAE was pretrained with, and what
#: ``DINOv2Encoder`` / ``ResNetEncoder`` use.  NOT CLIP's constants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: The two pooling modes, in one place so ``__init__`` and the error message
#: cannot disagree about what is supported.
_POOLING_MODES = ("mean", "cls")

_INSTALL_HINT = (
    "MAEEncoder needs the `timm` package, which is not installed.\n"
    "    uv sync --extra mae          (or: pip install 'timm>=1.0.0')\n"
    "timm is used because it is the only maintained distribution of the MAE "
    "checkpoints under a stable tag (`vit_base_patch16_224.mae`), and its "
    "checkpoint_filter_fn is what remaps the original facebookresearch/mae "
    "release into a loadable ViT — including positional-embedding resampling."
)


class MAEEncoder(Encoder):
    """Frozen MAE ViT. ``embed(obs) -> (B, d)`` pooled patch-token features.

    Parameters
    ----------
    weights_path
        Local checkpoint.  ``None`` (default) lets timm resolve ``model_name``
        from the HuggingFace hub, which **requires network access** — pass a
        path on an offline cluster, exactly as ``CLIPEncoder`` and
        ``ResNetEncoder`` allow and ``DINOv2Encoder`` requires.

        A path is handed to timm as ``pretrained_cfg_overlay=dict(file=...)``
        rather than being ``torch.load``-ed directly.  That routes it through
        timm's own ``checkpoint_filter_fn``, which unwraps a ``{"model": ...}``
        wrapper, remaps the original ``mae_pretrain_vit_base.pth`` key names,
        and resamples ``pos_embed`` when ``image_size`` is not the checkpoint's.
        A raw ``load_state_dict(strict=True)`` (the ``DINOv2Encoder`` approach)
        does none of that and rejects the upstream MAE release outright.
    model_name
        Any timm ViT with an MAE tag, e.g. ``"vit_base_patch16_224.mae"``
        (768-d, the arm in the ablation) or ``"vit_large_patch16_224.mae"``
        (1024-d, ~3x the compute and a 1.3x wider QEC).
    image_size
        224 (default) is the pretraining resolution and the honest choice.
        Anything else is passed to timm as ``img_size``, which interpolates the
        positional embeddings — a deviation from pretraining, so set it only
        deliberately.  It is also the one knob that buys throughput back:
        tokens scale as ``(image_size / patch)^2``.
    pooling
        ``"mean"`` (default) averages the **patch** tokens, dropping the model's
        ``num_prefix_tokens`` prefix entries.  ``"cls"`` takes the CLS token, for
        ablating the choice.  See the module docstring — for MAE this is not a
        cosmetic preference, and ``"cls"`` is expected to score worse.
    pretrained
        Load pretrained weights (default ``True``).  ``False`` builds the
        architecture **untrained**, which is meaningless as a representation but
        numerically identical, so it reproduces the key-stability rows of
        ``scripts/encoder_diagnostics.py --mae-random-init`` with no download and
        no network.  Deliberately NOT exposed through ``make_encoder``, so it
        cannot be reached from a training config; a ``weights_path`` overrides it
        (a path means "load this file").
    """

    def __init__(
        self,
        weights_path: str | None = None,
        model_name: str = "vit_base_patch16_224.mae",
        image_size: int = 224,
        pooling: str = "mean",
        device: torch.device | None = None,
        pretrained: bool = True,
    ) -> None:
        try:
            import timm
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(_INSTALL_HINT) from exc

        if pooling not in _POOLING_MODES:
            raise ValueError(
                f"mae_pooling={pooling!r} is not a pooling mode; expected one "
                f"of {list(_POOLING_MODES)}. 'mean' averages the patch tokens "
                f"(what MAE evaluation conventionally does, since MAE's CLS "
                f"token is never directly supervised by the reconstruction "
                f"loss); 'cls' takes the CLS token and exists to ablate that."
            )

        self.model_name = model_name
        self.image_size = int(image_size)
        self.pooling = pooling

        # NOTE: global_pool is deliberately NOT passed.  The default ('token')
        # keeps `self.norm` as the pretrained LayerNorm; global_pool='avg' would
        # swap it for a randomly initialised `fc_norm` the MAE checkpoint does
        # not contain.  This encoder pools from forward_features itself, so
        # timm's own pooling is never used either way.  See the module
        # docstring, and test_the_timm_default_pooling_is_token_not_avg.
        build_kwargs: dict = {
            # A weights_path always means "load this", whatever `pretrained`
            # says — the file is the thing being asked for.
            "pretrained": bool(pretrained) or weights_path is not None,
            "num_classes": 0,       # no classifier head; MAE has no labels
            "img_size": self.image_size,
        }
        if weights_path is not None:
            # A local file wins over the hub; that is the offline-cluster path.
            build_kwargs["pretrained_cfg_overlay"] = {"file": weights_path}

        self.model = timm.create_model(model_name, **build_kwargs)

        # How many leading tokens are NOT patches.  1 (CLS) for the MAE ViTs,
        # but read rather than assumed: a model with register tokens reports
        # more, and averaging those in would quietly corrupt the embedding.
        self._num_prefix = int(getattr(self.model, "num_prefix_tokens", 1))

        # eval() matters for the same reason it does in ResNetEncoder: any
        # stochastic or batch-statistic layer left in training mode makes the
        # same frame embed differently depending on its batch, and the QEC
        # exact-hit path never fires.  (A ViT has no BatchNorm, but it does have
        # dropout/droppath, which are no-ops only in eval mode.)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

        if device is not None:
            self.model.to(device)

        # Read off a real forward rather than `model.embed_dim`: it is the
        # ground truth including whatever pooling does, it costs one image, and
        # it is the same discipline CLIPEncoder uses.  768 for ViT-B/16.
        self.state_dim = self._probe_state_dim()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, N, d)`` tokens -> ``(B, d)``.

        ``"mean"`` drops the ``num_prefix_tokens`` prefix entries first, so the
        average is over patch tokens only — the CLS token is not a patch and
        MAE's is undertrained, so including it would both dilute and bias the
        embedding.
        """
        if tokens.ndim != 3:                             # pragma: no cover
            raise RuntimeError(
                f"forward_features returned shape {tuple(tokens.shape)}; "
                f"expected (B, N, d) tokens. MAEEncoder pools them itself, so "
                f"a model whose forward_features is already pooled cannot be "
                f"used here."
            )
        if self.pooling == "cls":
            return tokens[:, 0]
        return tokens[:, self._num_prefix:].mean(dim=1)

    @torch.no_grad()
    def _probe_state_dim(self) -> int:
        dev = next(self.model.parameters()).device
        probe = torch.zeros(1, 3, self.image_size, self.image_size, device=dev)
        out = self._pool(self.model.forward_features(probe))
        if out.ndim != 2:                                # pragma: no cover
            raise RuntimeError(
                f"pooled MAE output has shape {tuple(out.shape)}; expected "
                "(B, d). MFEC needs one vector per observation."
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
            x = F.interpolate(
                x,
                (self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self._mean.to(dev)) / self._std.to(dev)  # ImageNet-normalise

        # No F.normalize here, and that is deliberate — see the module
        # docstring.  MAE has no metric objective for a unit sphere to serve,
        # and DINOv2/ResNet leave their output raw too.
        return self._pool(self.model.forward_features(x)).float()

    def state(self) -> dict:
        return {"model_state_dict": self.model.state_dict()}

    def load_state(self, state: dict) -> None:
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
