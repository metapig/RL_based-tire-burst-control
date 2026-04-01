import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy().flatten()  # 必须展平！
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float).to(td_delta.device)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetContinuous, self).__init__()
        self.fc = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

        # ===== 隐藏层 =====
        torch.nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.fc.bias, 0)

        # ===== Mu 层（关键：小！！）=====
        torch.nn.init.orthogonal_(self.fc_mu.weight, gain=0.0)
        torch.nn.init.constant_(self.fc_mu.bias, 0)

        # ===== Std 层（关键：不要大）=====
        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -1)   

        # ===== Action scale =====
        self.register_buffer('action_scales', torch.tensor([1, 1, 1]))

    def forward(self, x):
       x = F.relu(self.fc(x))
       mu = torch.tanh(self.fc_mu(x)) * self.action_scales

       log_std = self.fc_std(x)
       log_std = torch.clamp(log_std, -5, 2)
       std = torch.exp(log_std) 

       return mu, std

class valueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(valueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v
    
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr,
                 critic_lr, gamma, lmbda, epochs, eps, device):
        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dim).to(device)
        self.critic = valueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
    
    def save_model(self, ckpt_path="ppo_checkpoint.pth", episode=0):
        torch.save({
        'episode': episode,
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'actor_optim': self.actor_optimizer.state_dict(),
        'critic_optim': self.critic_optimizer.state_dict(),
        }, ckpt_path)

    def reload_model(self, ckpt_path="ppo_checkpoint.pth"):
        ckpt = torch.load(ckpt_path)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optim'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optim'])
        start_episode = ckpt['episode'] + 1

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device) 
        mu, std = self.actor(state)  
        dist = torch.distributions.Normal(mu, std)
        action = mu#dist.sample()       
        log_probs = dist.log_prob(action)  

        # 返回给环境的时候可以 squeeze batch 维
        return action.squeeze(0).detach().cpu().numpy(), log_probs.sum(dim=1).item()

    def update(self, transition_dict):
        # states = torch.from_numpy(np.array(transition_dict['states'], dtype=np.float32)).to(self.device)
        # actions = torch.from_numpy(np.array(transition_dict['actions'], dtype=np.float32)).to(self.device)
        # rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).to(self.device)
        # next_states = torch.from_numpy(np.array(transition_dict['next_states'], dtype=np.float32)).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype = torch.float).to(self.device)
        # old_log_probs = torch.tensor(transition_dict['log_probs'],dtype=torch.float32).to(device=self.device)

        # td_targets = rewards.unsqueeze(1) + self.gamma * self.critic(next_states) * (1 - dones.unsqueeze(1))
        # td_deltas = (td_targets - self.critic(states)).squeeze()  # (batch,) 而非 (batch, 1)
        # advantages = compute_advantage(self.gamma, self.lmbda, td_deltas)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # critic_loss = F.mse_loss(self.critic(states), td_targets.detach())
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # for _ in range(self.epochs):
        #     mu, std = self.actor(states)
        #     dist = torch.distributions.Normal(mu, std)
        #     new_log_probs = dist.log_prob(actions).sum(axis=1)
        #     ratio = torch.exp(new_log_probs - old_log_probs)
        #     clip_frac = ((ratio > 1 + self.eps) | (ratio < 1 - self.eps)).float().mean()
        #     print(f"[PPO] clip frac: {clip_frac:.3f}")

        #     surr1 = ratio * advantages
        #     surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

        #     actor_loss = -torch.min(surr1, surr2).mean()
        #     self.actor_optimizer.zero_grad()
        #     actor_loss.backward()
        #     self.actor_optimizer.step()

        states: torch.Tensor = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float32, device=self.device)
        actions: torch.Tensor = torch.tensor(np.array(transition_dict["actions"]), dtype=torch.float32, device=self.device)
        rewards: torch.Tensor = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device)
        next_states: torch.Tensor = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float32, device=self.device)
        dones: torch.Tensor = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device)
        old_log_probs: torch.Tensor = torch.tensor(transition_dict["log_probs"], dtype=torch.float32, device=self.device)

        # ========= Critic Forward =========
        values: torch.Tensor = self.critic(states).squeeze()
        next_values: torch.Tensor = self.critic(next_states).squeeze()

        # ========= TD target =========
        td_targets: torch.Tensor = rewards + self.gamma * next_values * (1 - dones)
        td_deltas: torch.Tensor = td_targets - values

        # ========= GAE =========
        advantages: torch.Tensor = compute_advantage(
            self.gamma,
            self.lmbda,
            td_deltas
        ).to(self.device)

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========= Critic Update =========
        critic_loss: torch.Tensor = F.mse_loss(values, td_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========= Actor Update =========
        clip_fraction_total: float = 0.0
        entropy_total: float = 0.0

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=1)
            
            ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs)
            # PPO clip ratio
            surr1: torch.Tensor = ratio * advantages
            surr2: torch.Tensor = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

            actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ===== 统计指标 =====
            clip_frac: torch.Tensor = (
                (ratio > 1 + self.eps) | (ratio < 1 - self.eps)
            ).float().mean()

            clip_fraction_total += clip_frac.item()
            entropy_total += dist.entropy().mean().item()

        # ========= Metrics =========
        metrics: Dict[str, float] = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_total / self.epochs,
            "clip_frac": clip_fraction_total / self.epochs,
        }

        return metrics