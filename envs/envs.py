from omegaconf import DictConfig 
from .car_racing_env import CarRacingWithInfo

def make_env(env_cfg: DictConfig):
    if env_cfg.name == "CarRacing-v3":
        return CarRacingWithInfo(env_name=env_cfg.name, render_mode=env_cfg.render_mode, img_size=env_cfg.img_size, max_episode_steps=env_cfg.max_steps)
    else:
        NotImplementedError(f"Unsupported environment: {env_cfg.name}")