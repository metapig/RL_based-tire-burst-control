import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict

from src.models.parameters import ACTION_SCALES
from src.utils.utils import compute_advantage

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dim=256):
        super(PolicyNetContinuous, self).__init__()
        self.fc = torch.nn.Linear(state_dim + latent_dim, hidden_dim)
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
        torch.nn.init.constant_(self.fc_std.bias, 0)

        # Physical limits: steering angle and four wheel control moments.
        self.register_buffer(
            'action_scales', torch.tensor(ACTION_SCALES, dtype=torch.float32)
        )

    def forward(self, x):
       x = F.relu(self.fc(x))
       mu = self.fc_mu(x)

       log_std = self.fc_std(x)
       log_std = torch.clamp(log_std, -5, 0)
       std = torch.exp(log_std)

       return mu, std

class valueNet(torch.nn.Module):
    def __init__(self, state_dim, env_dim, hidden_dim):
        super(valueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + env_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v
    
class PrivilegedEncoder(torch.nn.Module):
    def __init__(self, env_dim, latent_dim = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, latent_dim)
        )

    def forward(self, entrinsic_state):
        return self.net(entrinsic_state)


class PPO:
    def __init__(self, state_dim, env_dim, action_dim, hidden_dim, latent_dim,actor_lr,
                 critic_lr, gamma, lmbda, epochs, eps, device,
                 entropy_coef=1e-3):
        self.actor = PolicyNetContinuous(state_dim, latent_dim, action_dim, hidden_dim).to(device)
        self.critic = valueNet(state_dim, env_dim, hidden_dim).to(device)
        self.privileged_encoder = PrivilegedEncoder(env_dim, latent_dim).to(device)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.privileged_encoder.parameters()), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.entropy_coef = entropy_coef
    
    def save_model(self, ckpt_path="ppo_checkpoint.pth", episode=0):
        torch.save({
        'episode': episode,
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'actor_optim': self.actor_optimizer.state_dict(),
        'critic_optim': self.critic_optimizer.state_dict(),
        "privileged_encoder": self.privileged_encoder.state_dict()
        }, ckpt_path)

    def reload_model(self, ckpt_path="ppo_checkpoint.pth"):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.privileged_encoder.load_state_dict(ckpt['privileged_encoder'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optim'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optim'])
        return ckpt['episode'] + 1

    def select_action(self, state, env_info, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        env_info = torch.as_tensor(env_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.privileged_encoder(env_info)
            mu, std = self.actor(torch.cat((state, z), dim=-1))
            dist = torch.distributions.Normal(mu, std)
            raw_action = mu if deterministic else dist.sample()
            normalized_action = torch.tanh(raw_action)
            action = normalized_action * self.actor.action_scales
            log_probs = self._squashed_log_prob(dist, raw_action, normalized_action)

        return action.squeeze(0).cpu().numpy(), log_probs.item()

    def _squashed_log_prob(self, dist, raw_action, normalized_action):
        jacobian = self.actor.action_scales * (1.0 - normalized_action.pow(2))
        return (dist.log_prob(raw_action) - torch.log(jacobian + 1e-6)).sum(dim=1)

    def update(self, transition_dict):
        if not transition_dict["states"]:
            raise RuntimeError(
                "No transitions were collected. Check the burst trigger and "
                "CarSim import/export channel mapping."
            )

        states = torch.as_tensor(
            np.asarray(transition_dict["states"]),
            dtype=torch.float32, device=self.device
        )
        env_infos = torch.as_tensor(
            np.asarray(transition_dict["env_infos"]),
            dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(transition_dict["actions"]),
            dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray(transition_dict["next_states"]),
            dtype=torch.float32, device=self.device
        )
        terminateds = torch.as_tensor(
            transition_dict["terminateds"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            transition_dict["log_probs"],
            dtype=torch.float32, device=self.device
        )


        # ========= GAE and lambda return =========
        with torch.no_grad():
            values = self.critic(torch.cat([states, env_infos], dim=-1)).squeeze(-1)
            next_values = self.critic(torch.cat([next_states, env_infos], dim=-1)).squeeze(-1)
            td_targets = (rewards + self.gamma * next_values * (1.0 - terminateds))

            td_deltas = td_targets - values

            raw_advantages = compute_advantage(self.gamma, self.lmbda, td_deltas, dones)  # terminated or truncated
            returns = raw_advantages + values

        # Advantage normalization
        advantages = ( raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + 1e-8)

        # ========= Critic Update =========
        critic_loss_total = 0.0
        for _ in range(self.epochs):
            predicted_values = self.critic(torch.cat([states, env_infos], dim=-1)).squeeze(-1)
            critic_loss = F.mse_loss(predicted_values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            critic_loss_total += critic_loss.item()

        # ========= Actor Update =========
        clip_fraction_total: float = 0.0
        entropy_total: float = 0.0

        for _ in range(self.epochs):
            z = self.privileged_encoder(env_infos)
            actor_inputs = torch.cat([states, z], dim=-1)
            mu, std = self.actor(actor_inputs)
            dist = torch.distributions.Normal(mu, std)
            normalized_actions = torch.clamp(
                actions / self.actor.action_scales,
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )
            raw_actions = torch.atanh(normalized_actions)
            new_log_probs = self._squashed_log_prob(
                dist, raw_actions, normalized_actions
            )
            
            ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs)
            # PPO clip ratio
            surr1: torch.Tensor = ratio * advantages
            surr2: torch.Tensor = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

            entropy = dist.entropy().sum(dim=1).mean()
            actor_loss = (-torch.min(surr1, surr2).mean() - self.entropy_coef * entropy)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_and_encoder_parameters = (
                list(self.actor.parameters())
                + list(self.privileged_encoder.parameters())
            )
            torch.nn.utils.clip_grad_norm_(actor_and_encoder_parameters, 0.5)
            self.actor_optimizer.step()

            # ===== 统计指标 =====
            clip_frac: torch.Tensor = ((ratio > 1 + self.eps) | (ratio < 1 - self.eps)).float().mean()

            clip_fraction_total += clip_frac.item()
            entropy_total += entropy.item()

        # ========= Metrics =========
        metrics: Dict[str, float] = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss_total / self.epochs,
            "entropy": entropy_total / self.epochs,
            "clip_frac": clip_fraction_total / self.epochs,
            "normalized_std": std.mean().item(),
        }

        return metrics
