import unittest

import numpy as np
import torch

from src.agents.PPO_agent import PPO
from src.envs.carsim_env import _classify_episode_end
from src.models.reward_model import compute_reward
from src.models.state_model import STATE_DIM, build_state


class MarkovStateTests(unittest.TestCase):
    def test_state_contains_all_reward_history_inputs(self):
        exports = np.asarray(
            (80.0, -2.5, 5.0, -1.0, 3.0, -1.0, 80.0, 0.825)
        )
        previous_action = np.asarray((1.5, 100.0, -100.0, 50.0, -50.0))

        state = build_state(exports, previous_action)

        self.assertEqual(state.shape, (STATE_DIM,))
        np.testing.assert_allclose(
            state,
            (0.8, -0.5, 0.5, -0.2, 0.2, 0.5,
             0.5, 0.5, -0.5, 0.25, -0.25),
            rtol=0.0,
            atol=1e-6,
        )

    def test_reward_is_deterministic_given_transition_information(self):
        exports = np.asarray((80.0, 1.0, -2.0, 0.5, -3.0, -1.0, 80.0, 0.8))
        previous_action = np.asarray((0.2, 10.0, -10.0, 20.0, -20.0))
        action = np.asarray((0.3, 20.0, -20.0, 30.0, -30.0))

        first = compute_reward(exports, action, previous_action)
        second = compute_reward(exports.copy(), action.copy(), previous_action.copy())

        self.assertEqual(first, second)

    def test_time_limit_is_a_truncation(self):
        terminated, truncated = _classify_episode_end(
            code=1, t_current=12.001, t_stop=12.0
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_solver_stop_is_a_true_termination(self):
        terminated, truncated = _classify_episode_end(
            code=1, t_current=5.0, t_stop=12.0
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_solver_error_at_time_limit_is_a_true_termination(self):
        terminated, truncated = _classify_episode_end(
            code=1, t_current=12.001, t_stop=12.0, solver_error=True
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_ppo_accepts_expanded_state(self):
        agent = PPO(
            state_dim=STATE_DIM,
            env_dim=0,
            action_dim=5,
            hidden_dim=32,
            actor_lr=1e-4,
            critic_lr=1e-3,
            gamma=0.995,
            lmbda=0.95,
            epochs=1,
            eps=0.2,
            device=torch.device("cpu"),
        )
        states = [np.zeros(STATE_DIM, dtype=np.float32) for _ in range(8)]
        actions = []
        log_probs = []
        for state in states:
            action, log_prob = agent.select_action(state)
            actions.append(action)
            log_probs.append(log_prob)

        metrics = agent.update(
            {
                "states": states,
                "actions": actions,
                "next_states": [state.copy() for state in states],
                "rewards": [0.0] * len(states),
                "dones": [False] * (len(states) - 1) + [True],
                "log_probs": log_probs,
            }
        )

        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
