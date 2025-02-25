import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorSAC, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        状態から行動をサンプルし、行動とその対数確率、平均行動を返す。
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # reparameterization trick を利用してサンプル
        x_t = normal.rsample()  
        # tanh を適用して出力を [-1, 1] に収める
        y_t = torch.tanh(x_t)
        action = y_t
        # 対数確率の計算 (tanh の影響も補正)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # 平均行動も tanh を通す
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorTD3, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh を用いて出力範囲を [-1, 1] に制限
        action = torch.tanh(self.fc3(x))
        return action
