import torch
from omegaconf import DictConfig 
from .maxvit.AE.maxvit_ae import MaxVITAE

def build_encoder(encoder_cfg: DictConfig, latent_dim: int = 512, device: torch.device = 'cpu'):
    if encoder_cfg.name == "maxvit":
        return MaxVITAE(encoder_cfg, latent_dim=latent_dim, device=device)
