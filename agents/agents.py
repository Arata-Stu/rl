from omegaconf import DictConfig
from .sac import SACAgent
from .td3 import TD3Agent

def get_agents(agent_cfg: DictConfig, latent_dim: int, state_dim: int, action_dim: int, device: str = 'cpu'):
    total_state_dim = latent_dim + state_dim

    if agent_cfg.name == "SAC":
        return SACAgent(total_state_dim=total_state_dim,
                        action_dim=action_dim,
                        hidden_dim=agent_cfg.hidden_dim,
                        lr=agent_cfg.lr,
                        tau=agent_cfg.tau,
                        gamma=agent_cfg.gamma,
                        device=device)
    
    elif agent_cfg.name == "TD3":
        return TD3Agent(total_state_dim=total_state_dim,
                        action_dim=action_dim,
                        hidden_dim=agent_cfg.hidden_dim,
                        lr=agent_cfg.lr,
                        tau=agent_cfg.tau,
                        gamma=agent_cfg.gamma,
                        policy_noise=agent_cfg.policy_noise,
                        noise_clip=agent_cfg.noise_clip,
                        policy_delay=agent_cfg.policy_delay,
                        device=device)
        