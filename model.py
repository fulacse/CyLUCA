from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F


class ElMo(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, n_layer):
        super(ElMo, self).__init__()

        self.embedding = Embedding(input_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_size, bidirectional=True, num_layers=n_layer,
                            batch_first=True)
        self.headPolicy = nn.Linear(hidden_size * 2, 1)
        self.headValue = nn.Linear(hidden_size * 2, 1)

    def forward(self, observation):
        embed = self.embedding(observation)
        lstm_out, _ = self.lstm(embed)
        assert lstm_out.shape == (32, 128, 1024)
        mask_pos = torch.tensor(observation['mask_pos'])[:, :, None].expand(-1, -1, lstm_out.size(-1))
        lstm_out = lstm_out.gather(1, mask_pos)
        assert lstm_out.shape == (32, 1, 1024)
        headPolicy = self.headPolicy(lstm_out)
        headValue = self.headValue(lstm_out)
        return headPolicy, headValue


class Embedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(Embedding, self).__init__()
        self.primary_embed = nn.Embedding(input_size, d_model)
        self.ss_embed = nn.Embedding(input_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observation):
        primary_embed = self.primary_embed(torch.tensor(observation['primary'], dtype=torch.long))
        ss_embed = self.ss_embed(torch.tensor(observation['ss'], dtype=torch.long))
        embed = primary_embed + ss_embed
        embed *= F.tanh(torch.tensor(observation['x']).unsqueeze(-1)) * F.tanh(
            torch.tensor(observation['y']).unsqueeze(-1)) * F.tanh(torch.tensor(observation['z']).unsqueeze(-1))
        assert embed.shape == (32, 128, 768)
        return self.norm(embed)
