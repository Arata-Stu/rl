import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils.helper import soft_update
from models.Critic.critic import Critic
from models.Actor.actor import ActorTD3

class TD3Agent:
    def __init__(self, total_state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, tau=0.005, gamma=0.99,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.total_state_dim = total_state_dim
        self.action_dim = action_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_counter = 0

        # ネットワーク初期化
        self.actor = ActorTD3(total_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = ActorTD3(total_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(total_state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(total_state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        if noise > 0:
            noise = torch.randn_like(action) * noise
            action = (action + noise).clamp(-1, 1)
        return action.cpu().detach().numpy()[0]

    def update(self, batch):
        # batch は辞書型：{"z", "state", "action", "reward", "next_z", "next_state", "done"}
        z = torch.FloatTensor(batch["z"]).to(self.device)
        state_extra = torch.FloatTensor(batch["state"]).to(self.device)
        actions = torch.FloatTensor(batch["action"]).to(self.device)
        rewards = torch.FloatTensor(batch["reward"]).to(self.device)
        next_z = torch.FloatTensor(batch["next_z"]).to(self.device)
        next_state_extra = torch.FloatTensor(batch["next_state"]).to(self.device)
        dones = torch.FloatTensor(batch["done"]).to(self.device)

        state = torch.cat([z, state_extra], dim=1)
        next_state = torch.cat([next_z, next_state_extra], dim=1)

        # 目標行動の計算（ノイズ付き）
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1, 1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Critic の更新
        current_Q1, current_Q2 = self.critic(state, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_counter += 1
        # 遅延更新による Actor の更新
        if self.update_counter % self.policy_delay == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ターゲットネットワークのソフト更新
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if self.update_counter % self.policy_delay == 0 else 0.0
        }