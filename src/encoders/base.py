from __future__ import annotations
import torch

class Encoder:
    """"
    Maps observations to fixed-dim state embeddings for episodic memory.and
        
    Contract:
        mbed(obs) -> (B, d) float32 on obs.device
      obs is (..., C, H, W); leading dims are flattened to B.
      Must be deterministic: identical pixels -> identical embedding,
      or the QEC exact-hit hash path breaks.
"""

    state_dim: int

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

    def state(self) -> dict:
        raise NotImplementedError


    def load_state(self, state: dict) -> None:
        raise NotImplementedError