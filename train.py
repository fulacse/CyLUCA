import gymnasium as gym

from stable_baselines3 import PPO
from env import LUCAEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from model import CustomNetwork, CustomCombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = LUCAEnv(traget=0.5, pad_length=64)


# Custom Policy network
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, custom_parame, *args, **kwargs):
        self.ortho_init = False
        self.custom_parame = custom_parame
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.custom_parame["hidden_dim"])


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=768, hidden_dim=512, n_layer=2),
    custom_parame=dict(hidden_dim=512),
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=2, learning_rate=0.0001, n_steps=10,
            batch_size=10, n_epochs=8, gae_lambda=1, clip_range=0.1, ent_coef=0.01, tensorboard_log="./tensorboard", )
model.learn(total_timesteps=100)
model.save("ppo_CyLUCA")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_CyLUCA")

obs, info = env.reset()
env.render("human")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")
    print(f"\tReward: {rewards}")
    if dones:
        break
