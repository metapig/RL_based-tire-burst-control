from src.agents.mlp_env_encoder import EncoderDataset, MLPEnvEncoder


# Backward-compatible name. The RMA MVP intentionally uses an MLP, not GRU/LSTM.
EnvEncoder = MLPEnvEncoder

__all__ = ["EnvEncoder", "MLPEnvEncoder", "EncoderDataset"]
