from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import math

from stable_baselines3.common.callbacks import BaseCallback
from env import LUCAEnv
from model import CustomNetwork, CustomCombinedExtractor

# Create environment
env = LUCAEnv(pad_length=128)


# Custom Policy network
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, custom_parame, *args, **kwargs):
        self.ortho_init = False
        self.custom_parame = custom_parame
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.custom_parame["hidden_dim"])


class CustomCallback(BaseCallback):
    def __init__(self, save_freq, save_path, T_0=1, T_mult=2, eta_min=0, eta_max=0.02, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.current_step = 0
        self.n_cycles = 0
        self.T_cur = 0

    def _on_step(self) -> bool:
        # Update the current step and T_cur
        self.current_step += 1
        self.T_cur += 1

        if self.T_cur > self.T_0:
            self.n_cycles += 1
            self.T_cur = self.T_cur % self.T_0
            self.T_0 *= self.T_mult

        # Adjust the entropy coefficient
        new_entropy_coef = self.cosine_annealing_warm_restarts_entropy_scheduler()
        self.model.ent_coef = new_entropy_coef

        # Save the model every `save_freq` steps
        if self.current_step % self.save_freq == 0:
            self.model.save(f"{self.save_path}_{self.current_step}")

        return True

    def cosine_annealing_warm_restarts_entropy_scheduler(self):
        T_cur = self.T_cur
        T_0 = self.T_0
        eta_min = self.eta_min
        eta_max = self.eta_max
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_0))


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=768, hidden_dim=512, n_layer=2),
    custom_parame=dict(hidden_dim=512),
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=2, learning_rate=0.0001, n_steps=10,
            batch_size=10, n_epochs=8, gae_lambda=1, clip_range=0.1, tensorboard_log="./tensorboard", )
custom_callback = CustomCallback(save_freq=1_000, save_path='./trained/ppo_CyLUCA')
model.learn(total_timesteps=10_000, progress_bar=True, callback=custom_callback)
