import numpy as np
import torch
from .base import Encoder

class RandomProjectionEncoder(Encoder):
    def __init__(self, obs_flat_dim: int, state_dim: int = 64, seed: int | None = None):
        self.state_dim = state_dim
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((obs_flat_dim, state_dim))
        proj /= np.linalg.norm(proj, axis=0)
        self.projection = proj
        self._proj_tensor: torch.Tensor | None = None

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        if self._proj_tensor is None or self._proj_tensor.device != obs.device:
            self._proj_tensor = torch.tensor(self.projection, dtype=torch.float32, device=obs.device)

        flat = obs.float().reshape(-1, self._proj_tensor.shape[0])
        return flat @ self._proj_tensor
    
    def state(self) -> dict:
        return {"projection": self.projection}
    
    def load_state(self, state: dict) -> None:
        self.projection = state["projection"]
        self._proj_tensor = None  # Reset tensor to force re-creation on next embed call