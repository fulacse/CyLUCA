from model import ElMo
import torch
import numpy as np

batch_size = 32
max_len = 128
embedding_size = 768
hidden_size = 512
num_layers = 2
num_classes = 25

ElMo = ElMo(max_len, embedding_size, hidden_size, num_layers)

observation = {
    'primary': np.random.randint(0, 20, (batch_size, max_len)),
    'ss': np.random.randint(0, 3, (batch_size, max_len)),
    'mask_pos': np.random.randint(0, 128, (batch_size, 1)),
    'x': np.random.randn(batch_size, max_len),
    'y': np.random.randn(batch_size, max_len),
    'z': np.random.randn(batch_size, max_len),
}
print("Primary: ", observation['primary'].shape)
print("SS: ", observation['ss'].shape)
print("Mask_pos: ", observation['mask_pos'].shape)
print("X: ", observation['x'].shape)
print("Y: ", observation['y'].shape)
print("Z: ", observation['z'].shape)
policy, valua = ElMo(observation)
print(policy.shape, valua.shape)