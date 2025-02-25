import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils.helper import soft_update, transform_action
from models.Critic.critic import Critic
from models.Actor.actor import ActorSAC

# SACエージェント
class SACAgent:
    def __init__(self, total_state_dim, action_dim, hidden_dim=256, lr=3e-4, tau=0.005, gamma=0.99, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.total_state_dim = total_state_dim  # encoder の z 次元 + vehicle info 次元
        self.action_dim = action_dim

        # ネットワーク初期化
        self.actor = ActorSAC(total_state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(total_state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(total_state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 自動エントロピー調整のための alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_dim)

    def select_action(self, z: np.ndarray, state: np.ndarray, action_space, evaluate: bool=False):
        """
        エージェントの行動を選択し、環境の action_space に適合するように変換する。

        Args:
            z (np.ndarray): 画像特徴量
            state (np.ndarray): その他の状態情報
            action_space (gymnasium.spaces.Box): 環境の action_space
            evaluate (bool): True の場合は決定論的な行動を選択

        Returns:
            np.ndarray: 環境に適用できるアクション
        """
        z = torch.FloatTensor(z).to(self.device)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = torch.cat([z, state], dim=1)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor.forward(state)
                raw_action = torch.tanh(mean).cpu().numpy()  # [-1,1] に制限
        else:
            raw_action, _, _ = self.actor.sample(state)
            raw_action = raw_action.cpu().detach().numpy()  # [-1,1] に制限

        # 環境の action_space に適用
        transformed_action = transform_action(raw_action, action_space).squeeze(0)
        
        return transformed_action


    def update(self, batch):
        # batch は辞書型：{"z", "state", "action", "reward", "next_z", "next_state", "done"}
        z = torch.FloatTensor(batch["z"]).to(self.device)
        state_extra = torch.FloatTensor(batch["state"]).to(self.device)
        actions = torch.FloatTensor(batch["action"]).to(self.device)
        rewards = torch.FloatTensor(batch["reward"]).to(self.device)
        next_z = torch.FloatTensor(batch["next_z"]).to(self.device)
        next_state_extra = torch.FloatTensor(batch["next_state"]).to(self.device)
        dones = torch.FloatTensor(batch["done"]).to(self.device)

        # 状態は encoder の出力と vehicle info の concat
        state = torch.cat([z, state_extra], dim=1)
        next_state = torch.cat([next_z, next_state_extra], dim=1)

        # Critic 更新
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - torch.exp(self.log_alpha) * next_log_prob
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 更新
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha 更新
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ターゲットネットワークのソフト更新
        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": torch.exp(self.log_alpha).item()
        }