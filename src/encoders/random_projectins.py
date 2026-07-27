import numpy as np
import torch
from .base import Encoder


class RandomProjectionEncoder(Encoder):
    """Fixed Gaussian random projection φ: x → Ax (Blundell et al. 2016, §3).

    Numerical note — why the matmul runs in float64
    ------------------------------------------------
    MFEC's exact-match path (``QEC._key_to_slot``) hashes the embedding by
    rounding ``state * key_scale`` to int32, so the *same* game state must
    produce a bit-identical embedding every time it is seen.

    A float32 matmul does not guarantee that across different batch shapes:
    BLAS picks a different kernel (and therefore a different reduction order)
    for a 1-row GEMV than for a 16-row GEMM.  Accumulating 7 k–28 k terms in
    float32 leaves roughly 1e-6 of absolute error, which is the same order as
    the 1e-5 quantisation step at the default ``key_scale`` — so the training
    loop (which always embeds ``num_envs`` rows at a time) and
    ``BaseTrainer.evaluate`` (which builds a single env, i.e. 1 row) would
    disagree on the key for an identical frame, and evaluation would silently
    never take the exact-match path.

    Accumulating in float64 drops that error to ~1e-15 relative, ten orders of
    magnitude below the quantisation step, making the key invariant to batch
    shape, device and kernel choice.  The cost is negligible: the projection is
    a (D x 64) matrix and D is at most 28 k.
    """

    def __init__(self, obs_flat_dim: int, state_dim: int = 64, seed: int | None = None):
        self.state_dim = state_dim
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((obs_flat_dim, state_dim))
        proj /= np.linalg.norm(proj, axis=0)
        self.projection = proj
        self._proj_tensor: torch.Tensor | None = None

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        if self._proj_tensor is None or self._proj_tensor.device != obs.device:
            self._proj_tensor = torch.tensor(
                self.projection, dtype=torch.float64, device=obs.device
            )

        flat = obs.reshape(-1, self._proj_tensor.shape[0]).double()
        return (flat @ self._proj_tensor).float()

    def state(self) -> dict:
        return {"projection": self.projection}

    def load_state(self, state: dict) -> None:
        self.projection = state["projection"]
        self._proj_tensor = None  # Reset tensor to force re-creation on next embed call