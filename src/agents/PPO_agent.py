import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict

from src.models.parameters import ACTION_SCALES
from src.utils.utils import compute_advantage
from src.agents.history_encoder import CNN

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dim=256,
                 initial_std = 0.2, min_std = 0.03, max_std = 0.6):
        super().__init__()
        self.mean_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

        self.log_std = torch.nn.Parameter(torch.full((action_dim,), np.log(initial_std), dtype=torch.float32))

        self.register_buffer("log_std_min", torch.full((action_dim,), np.log(min_std), dtype=torch.float32))
        self.register_buffer("log_std_max", torch.full((action_dim,), np.log(max_std), dtype=torch.float32))
        self.register_buffer("action_scales", torch.tensor(ACTION_SCALES, dtype=torch.float32))

        self._initialize_weights()

    def _initialize_weights(self):
        linear_layers = [module for module in self.mean_net if isinstance(module, torch.nn.Linear)]

        for layer in linear_layers[:-1]:
            torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2)) # 这里是为什么？
            torch.nn.init.constant_(layer.bias, 0.0)

        torch.nn.init.orthogonal_(linear_layers[-1].weight, gain = 0.01)
        torch.nn.init.constant_(linear_layers[-1].bias, 0.0)

    def forward(self, x):
       mu = self.mean_net(x)

       log_std = self.log_std.expand_as(mu)
       std = torch.exp(log_std)

       return mu, std

    @torch.no_grad()
    def constrain_std(self):
        self.log_std.copy_(
            torch.maximum(
                torch.minimum(self.log_std, self.log_std_max),
                self.log_std_min,
            )
        )

class valueNet(torch.nn.Module):
    def __init__(self, state_dim, env_dim, hidden_dim):
        super().__init__()
        self.valueNet = torch.nn.Sequential(
            torch.nn.Linear(state_dim + env_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        linear_layers = [
            module
            for module in self.valueNet
            if isinstance(module, torch.nn.Linear)
        ]

        for layer in linear_layers[:-1]:
            torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(layer.bias, 0.0)

        torch.nn.init.orthogonal_(linear_layers[-1].weight, gain=1.0)
        torch.nn.init.constant_(linear_layers[-1].bias, 0.0)

    def forward(self, x):
        return self.valueNet(x)
    
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

        self._initialize_weights()

    def _initialize_weights(self):
        linear_layers = [
            module
            for module in self.net
            if isinstance(module, torch.nn.Linear)
        ]

        for layer in linear_layers:
            torch.nn.init.orthogonal_(layer.weight, gain=1.0)
            torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, entrinsic_state):
        return self.net(entrinsic_state)


class PPO:
    def __init__(self, state_dim, env_dim, action_dim, hidden_dim, latent_dim,actor_lr,
                 critic_lr, gamma, lmbda, epochs, eps, device,
                 entropy_coef=1e-4):
        self.actor = PolicyNetContinuous(state_dim, latent_dim, action_dim, hidden_dim).to(device)
        self.critic = valueNet(state_dim, env_dim, hidden_dim).to(device)
        self.privileged_encoder = PrivilegedEncoder(env_dim, latent_dim).to(device)
        self.history_encoder = CNN(state_dim, latent_dim).to(device)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.privileged_encoder.parameters()), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.state_dim = state_dim
        self.env_dim = env_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.entropy_coef = entropy_coef
    
    def save_model(self, ckpt_path="ppo_checkpoint.pth", episode=0):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
        'episode': episode,
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'actor_optim': self.actor_optimizer.state_dict(),
        'critic_optim': self.critic_optimizer.state_dict(),
        "privileged_encoder": self.privileged_encoder.state_dict()
        }, ckpt_path)

    def reload_model(self, ckpt_path="ppo_checkpoint.pth"):
        try:
            ckpt = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.privileged_encoder.load_state_dict(ckpt['privileged_encoder'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optim'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optim'])
        return ckpt['episode'] + 1

    def _as_batch_tensor(self, value, expected_dim, name):
        value = torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        )
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {name} shape [B, {expected_dim}], got "
                f"{tuple(value.shape)}."
            )
        return value

    def encode_env_info(self, env_info):
        env_info = self._as_batch_tensor(
            env_info, self.env_dim, "env_info"
        )
        return self.privileged_encoder(env_info)

    def predict_latent(self, history):
        history = torch.as_tensor(
            history, dtype=torch.float32, device=self.device
        )
        if history.ndim == 2:
            history = history.unsqueeze(0)
        return self.history_encoder(history)

    def select_action_oracle(self, state, env_info, deterministic=False):
        with torch.no_grad():
            z = self.encode_env_info(env_info)
        return self.select_action_from_z(state, z, deterministic)

    def select_action_adaptive(self, state, history, deterministic=False):
        with torch.no_grad():
            z_hat = self.predict_latent(history)
        return self.select_action_from_z(state, z_hat, deterministic)

    def select_action_from_z(self, state, z, deterministic=False):
        state = self._as_batch_tensor(state, self.state_dim, "state")
        z = self._as_batch_tensor(z, self.latent_dim, "z")

        action_input = torch.cat((state, z), dim=-1)

        with torch.no_grad():
            mu, std = self.actor(action_input)
            dist = torch.distributions.Normal(mu, std)
            raw_action = mu if deterministic else dist.sample()
            normalized_action = torch.tanh(raw_action)
            action = normalized_action * self.actor.action_scales
            log_probs = self._squashed_log_prob(dist, raw_action)

        return action.squeeze(0).cpu().numpy(), log_probs.item(), raw_action.squeeze(0).cpu().numpy()

    def _squashed_log_prob(self, dist, raw_action):
        gaussian_log_prob = dist.log_prob(raw_action)

        log_tanh_jacobian = 2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))

        return (gaussian_log_prob - log_tanh_jacobian - torch.log(self.actor.action_scales)).sum(dim=-1)

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
        raw_actions = torch.as_tensor(
            np.asarray(transition_dict["raw_actions"]),
            dtype=torch.float32,
            device=self.device,
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
            new_log_probs = self._squashed_log_prob(dist, raw_actions)
            
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
            # PPO clip ratio
            surr1: torch.Tensor = ratio * advantages
            surr2: torch.Tensor = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

            entropy_raw_action = dist.rsample()
            entropy_log_prob = self._squashed_log_prob(dist, entropy_raw_action)
            entropy = -entropy_log_prob.mean()
            actor_loss = (-torch.min(surr1, surr2).mean() - self.entropy_coef * entropy)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_and_encoder_parameters = (
                list(self.actor.parameters())
                + list(self.privileged_encoder.parameters())
            )
            torch.nn.utils.clip_grad_norm_(actor_and_encoder_parameters, 0.5)
            self.actor_optimizer.step()
            self.actor.constrain_std()

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
