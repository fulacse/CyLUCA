from stable_baselines3 import PPO
from env import LUCAEnv

model = PPO.load("trained/ppo_CyLUCA_5000")
env = LUCAEnv(pad_length=128)

obs, info = env.reset(sequences_selected=['1UCSA', '3NIRA', '5D8VA', '5NW3A'])
env.render("human")
for _ in range(30):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")
    print(f"\tReward: {rewards}")
    if dones:
        obs, info = env.reset(sequences_selected=['1UCSA', '3NIRA', '5D8VA', '5NW3A'])
        env.render("human")
