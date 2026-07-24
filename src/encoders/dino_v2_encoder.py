import torch
import torch.nn.functional as F
from .base import Encoder

# ImageNet stats DINOv2 was trained with
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Encoder(Encoder):
    """Frozen DINOv2 ViT as MFEC's φ (pretrained visual representation).

    embed(obs) -> (B, embed_dim) CLS-token features, deterministic.
    obs is (..., 3, H, W) float in [0, 1] (post-ToTensorImage). Resized to
    `image_size` (multiple of 14) and ImageNet-normalised before the ViT.

    state_dim is fixed by the backbone (vits14=384 ... vitg14=1536), read
    straight off the model's embed_dim after loading.
    """

    def __init__(
            self,
            weights_path: str,
            model_name: str = "dinov2_vits14",
            repo_dir: str | None = None,            # local clone of facebookresearch/dinov2
            image_size: int = 224,
            device: torch.device | None = None,
    ):
        assert image_size % 14 == 0, "DINOv2 ViT requires image_size to be a multiple of 14"

        self.model_name = model_name
        self.image_size = image_size

        # Build architecture without downloading weights, then load yours.
        if repo_dir is not None:
            self.model = torch.hub.load(repo_dir, model_name, source="local", pretrained=False)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)

        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd   # unwrap if wrapped

        self.model.load_state_dict(sd, strict=True)

        # state_dim comes straight from the backbone (vits14=384 ... vitg14=1536).
        self.state_dim = int(self.model.embed_dim)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        dev = obs.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)

        x = obs.float().reshape(-1, *obs.shape[-3:])  # flatten leading dims -> (B, 3, H, W)

        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        x = (x - self._mean.to(dev)) / self._std.to(dev)  # ImageNet-normalise

        return self.model(x).float()

    def state(self) -> dict:
        return {"model_state_dict": self.model.state_dict()}

    def load_state(self, state: dict) -> None:
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()