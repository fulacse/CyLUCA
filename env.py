from random import randint

import gymnasium as gym
import numpy as np

from tools.CyPred import cypred
from tools.file_reader import parse_ss2_file


class LUCAEnv(gym.Env):
    def __init__(self, min_pred=2, max_pred=8, max_steps=10):
        super(LUCAEnv, self).__init__()
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.max_steps = max_steps

        self.actions_to_nums = {'[REMOVE]': 0, 'Q': 1, 'P': 2, 'A': 3, 'V': 4, 'T': 5, 'G': 6, 'E': 7, 'Y': 8, 'D': 9,
                                'X': 10, 'N': 11, 'S': 12, 'I': 13, 'K': 14, 'Z': 15, 'F': 16, 'H': 17, 'R': 18,
                                'M': 19, 'W': 20, 'C': 21, 'L': 22}
        self.nums_to_actions = {num: action for action, num in self.actions_to_nums.items()}
        self.action_space = gym.spaces.Discrete(len(self.actions_to_nums))
        self.AA_to_nums = {'[PAD]': 0, '[AIR]': 1, 'Q': 2, 'P': 3, 'A': 4, 'V': 5, 'T': 6, 'G': 7, 'E': 8, 'Y': 9,
                           'D': 10, 'X': 11, 'N': 12, 'S': 13, 'I': 14, 'K': 15, 'Z': 16, 'F': 17, 'H': 18, 'R': 19,
                           'M': 20, 'W': 21, 'C': 22, 'L': 23}
        self.nums_to_AA = {num: AA for AA, num in self.AA_to_nums.items()}
        self.SS_to_nums = {'[PAD]': 0, '[AIR]': 1, '[UNKNOWN]': 2, 'H': 3, 'E': 4, 'C': 5}
        self.nums_to_SS = {num: SS for SS, num in self.SS_to_nums.items()}
        self.observation_space = gym.spaces.Dict({
            'primary': gym.spaces.Box(low=0, high=len(self.AA_to_nums), shape=(1, 128), dtype=np.int32),
            # 128 is the max length of the protein
            'ss': gym.spaces.Box(low=0, high=len(self.SS_to_nums), shape=(1, 128), dtype=np.int32),
            # 128 is the max length of the protein
            'mask_pos': gym.spaces.Box(low=0, high=128, shape=(1, 1), dtype=np.int32),  # value between 0 and 128
            'x': gym.spaces.Box(low=-3, high=1, shape=(1, 128), dtype=np.float32),  # value between 0 and 1, -1 if air, -2 if unknown, -3 if padding
            'y': gym.spaces.Box(low=-3, high=1, shape=(1, 128), dtype=np.float32),  # value between 0 and 1, -1 if air, -2 if unknown, -3 if padding
            'z': gym.spaces.Box(low=-3, high=1, shape=(1, 128), dtype=np.float32),  # value between 0 and 1, -1 if air, -2 if unknown, -3 if padding
        })

        self.acid_amino = []
        self.secondary_structure = []
        self.start_mask_pos = 0
        self.x = []
        self.y = []
        self.z = []

        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        if action == 0:  # remove one amino acid
            del self.acid_amino[self.start_mask_pos - 1]
            del self.secondary_structure[self.start_mask_pos - 1]
            del self.x[self.start_mask_pos - 1]
            del self.y[self.start_mask_pos - 1]
            del self.z[self.start_mask_pos - 1]
            self.start_mask_pos -= 1
        else:  # add one amino acid
            self.acid_amino = self.acid_amino[:self.start_mask_pos] + [action] + self.acid_amino[self.start_mask_pos:]
            self.secondary_structure = self.secondary_structure[:self.start_mask_pos] + [
                self.SS_to_nums['[UNKNOWN]']] + self.secondary_structure[self.start_mask_pos:]
            self.x = self.x[:self.start_mask_pos] + [-2] + self.x[self.start_mask_pos:]
            self.y = self.y[:self.start_mask_pos] + [-2] + self.y[self.start_mask_pos:]
            self.z = self.z[:self.start_mask_pos] + [-2] + self.z[self.start_mask_pos:]
            self.start_mask_pos += 1

        cypred_score = cypred(''.join([self.nums_to_AA[aa] for aa in
                                       self.acid_amino]))  # score given by cypred, > 0 if the protein is cycle and < 0 if not
        done = self.current_step >= self.max_steps or cypred_score > 0  # done if the protein is cycle or the max steps is reached

        if done:
            reward = cypred_score
        else:
            reward = -0.1 if action == 0 else 0  # -0.1 if the action is remove, 0 if the action is add

        return {
            'primary': list_to_numpy(self.acid_amino, self.AA_to_nums['[PAD]'], np.int32),
            'ss': list_to_numpy(self.secondary_structure, self.SS_to_nums['[PAD]'], np.int32),
            'mask_pos': np.array([[self.start_mask_pos]], np.int32),
            'x': list_to_numpy(self.x, -3, np.float32),
            'y': list_to_numpy(self.y, -3, np.float32),
            'z': list_to_numpy(self.z, -3, np.float32)
        }, reward, done, done, {}

    def reset(self,seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.current_step = 0
        self.acid_amino = []
        self.secondary_structure = []
        self.start_mask_pos = 0
        self.x = []
        self.y = []
        self.z = []

        path = "data/5D8VA/20f6ee26-e2ab-11ee-905a-00163e100d53.ss2"
        predictions = parse_ss2_file(path)
        for prediction in predictions:
            self.acid_amino.append(self.AA_to_nums[prediction[1]])
            self.secondary_structure.append(self.SS_to_nums[prediction[2]])
            self.x.append(prediction[3][0])
            self.y.append(prediction[3][1])
            self.z.append(prediction[3][2])

        len_mask = randint(self.min_pred, self.max_pred)  # min_pred ~ max_pred
        can_start_mask_pos = [i for i in range(len(self.acid_amino) - len_mask) if
                              self.secondary_structure[i] == self.SS_to_nums['C']]
        self.start_mask_pos = can_start_mask_pos[randint(0, len(can_start_mask_pos) - 1)]
        self.acid_amino = self.acid_amino[:self.start_mask_pos] + [self.AA_to_nums['[AIR]']] + self.acid_amino[
                                                                                               self.start_mask_pos + len_mask:]
        self.secondary_structure = self.secondary_structure[:self.start_mask_pos] + [
            self.SS_to_nums['[AIR]']] + self.secondary_structure[self.start_mask_pos + len_mask:]
        self.x = self.x[:self.start_mask_pos] + [-1] + self.x[self.start_mask_pos + len_mask:]
        self.y = self.y[:self.start_mask_pos] + [-1] + self.y[self.start_mask_pos + len_mask:]
        self.z = self.z[:self.start_mask_pos] + [-1] + self.z[self.start_mask_pos + len_mask:]

        return {
            'primary': list_to_numpy(self.acid_amino, self.AA_to_nums['[PAD]'], np.int32),
            'ss': list_to_numpy(self.secondary_structure, self.SS_to_nums['[PAD]'], np.int32),
            'mask_pos': np.array([[self.start_mask_pos]], np.int32),
            'x': list_to_numpy(self.x, -3, np.float32),
            'y': list_to_numpy(self.y, -3, np.float32),
            'z': list_to_numpy(self.z, -3, np.float32)
        }, {}

    def render(self, mode='human'):
        print(f"Current step: {self.current_step}")
        print(f"\tMask position: {self.start_mask_pos}")
        print(f"\tProtein: {''.join([self.nums_to_AA[aa] for aa in self.acid_amino])}")
        print(f"\tSecondary structure: {''.join([self.nums_to_SS[ss] for ss in self.secondary_structure])}")
        print(f"\tX: {self.x}")
        print(f"\tY: {self.y}")
        print(f"\tZ: {self.z}")

    def close(self):
        pass


def list_to_numpy(list, pad, dtype, shape=(1, 128)) -> np.array:
    # Convert list to a NumPy array
    initial_array = np.array(list, dtype=dtype)

    # Ensure the array is 2D with a single row
    initial_array_reshaped = initial_array.reshape(1, -1)

    # Calculate the padding width needed
    padding_width = shape[1] - initial_array_reshaped.shape[1]

    # Pad the array with zeros to reach the desired shape
    padded_array = np.pad(initial_array_reshaped, ((0, 0), (0, padding_width)), 'constant', constant_values=pad)

    assert padded_array.shape == shape, f"Expected shape {shape} but got shape {padded_array.shape}"
    return padded_array
