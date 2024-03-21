import gymnasium as gym

from stable_baselines3 import PPO
from env import LUCAEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from model import CustomNetwork, CustomCombinedExtractor

# Create environment
env = LUCAEnv()


# Custom Policy network
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, custom_parame, *args, **kwargs):
        self.ortho_init = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    custom_parame='leet',
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_CyLUCA")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_CyLUCA")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
