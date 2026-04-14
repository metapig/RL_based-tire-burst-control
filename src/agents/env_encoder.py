import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EnvEncoder(nn.Module):
    """
        路面附着系数μ识别网络
        输入：历史状态-动作序列
        输出：μ估计值 (0.2 ~ 1.0)
    """

    def __init__(self, state_dim: int  = 3, action_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 1, history_len: int = 25, mu_range: Tuple[float, float] = (0.2, 1.0)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.mu_min, self.mu_max = mu_range

        self.gru = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=0.1 if num_layers > 1 else 0.0)
        
        # 回归头：从hidden state预测mu
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出0-1，再映射到mu_range
        )

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # 零历史的可学习嵌入
        self.zero_embed = nn.Parameter(torch.randn(1, num_layers, hidden_dim) * 0.01)

    def forward(self, history_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_seq: (batch, seq_len, 6) 或 (seq_len, 6)
            valid_len: (batch,) 有效长度
        Returns:
            mu_pred: (batch, 1) 估计的附着系数，范围[mu_min, mu_max]
        """
        if history_seq.dim() == 2:
            history_seq = history_seq.unsqueeze(0)  # (1, seq_len, 6)

        batch_size, seq_len, _ = history_seq.shape

        # GRU前向
        if seq_len == 0:
            # 空历史：使用可学习嵌入
            h_n = self.zero_embed.expand(-1, batch_size, -1)
        else:
            _, h_n = self.gru(history_seq)  # h_n: (num_layers, batch, hidden_dim)
        
        # 取最后一层的hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # 预测mu
        mu_normalized = self.mu_head(last_hidden)  # (batch, 1)，范围[0,1]
        mu_pred = self.mu_min + mu_normalized * (self.mu_max - self.mu_min)
        
        return mu_pred
    
    def predict_mu(self, state: np.ndarray, action: np.ndarray,
                history_buffer: Dict[str, List]) -> float:
        """便捷方法：从buffer预测当前mu（用于调试）"""
        seq = self._buffer_to_tensor(history_buffer, state, action)
        with torch.no_grad():
            mu = self.forward(seq).item()
        return mu
    
    def _buffer_to_tensor(self, 
                         buffer: Dict[str, List],
                         current_state: np.ndarray,
                         current_action: np.ndarray) -> torch.Tensor:
        """构造历史序列tensor（包含当前步）"""
        seq_len = len(buffer['states'])
        
        if seq_len == 0:
            # 只有当前步
            seq = torch.tensor(
                np.concatenate([current_state, current_action]),
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 6)
        else:
            # 历史 + 当前
            states = np.array(buffer['states'])
            actions = np.array(buffer['actions'])
            current = np.concatenate([current_state, current_action]).reshape(1, -1)
            
            full_states = np.vstack([states, current_state.reshape(1, -1)])
            full_actions = np.vstack([actions, current_action.reshape(1, -1)])
            full_seq = np.hstack([full_states, full_actions])  # (seq_len+1, 6)
            
            seq = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len+1, 6)
        
        return seq
