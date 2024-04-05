import gymnasium as gym
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ElMo(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, n_layer):
        super(ElMo, self).__init__()

        self.embedding = Embedding(input_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_size, bidirectional=True, num_layers=n_layer,
                            batch_first=True)

    def forward(self, observation):
        embed = self.embedding(observation)
        lstm_out, _ = self.lstm(embed)
        # assert lstm_out.shape == (32, 128, 1024)
        mask_pos = observation['mask_pos'][:, :, None].expand(-1, -1, lstm_out.size(-1)).to(dtype=torch.long)
        # assert mask_pos.shape == (32, 1, 1024)
        lstm_out = lstm_out.gather(1, mask_pos)
        # assert lstm_out.shape == (32, 1, 1024)
        return lstm_out


class Embedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(Embedding, self).__init__()
        self.primary_embed = nn.Embedding(input_size, d_model)
        self.ss_embed = nn.Embedding(input_size, d_model)
        self.cord_encode = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observation):
        primary_embed = self.primary_embed(observation['primary'].to(dtype=torch.long))
        ss_embed = self.ss_embed(observation['ss'].to(dtype=torch.long))
        embed = primary_embed + ss_embed
        concatenated_tensor = torch.cat(
            (observation['x'].unsqueeze(-1), observation['y'].unsqueeze(-1), observation['y'].unsqueeze(-1)), dim=2)
        cord_embed = self.cord_encode(concatenated_tensor)
        embed += cord_embed
        # assert embed.shape == (32, 128, 768)
        return self.norm(embed)


class CustomNetwork(nn.Module):
    def __init__(self, feature_dim, latent_dim_pi=64, latent_dim_vf=64):
        super().__init__()

        self.latent_dim_pi = feature_dim * 2
        self.latent_dim_vf = feature_dim * 2

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 768, hidden_dim: int = 512,
                 n_layer: int = 2):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        self.elmo = ElMo(observation_space['primary'].shape[1], features_dim, hidden_dim, n_layer)

    def forward(self, observations):
        for key in observations.keys(): observations[key] = observations[key].squeeze(1)
        return self.elmo(observations)
