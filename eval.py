from stable_baselines3 import PPO
from env import LUCAEnv

model = PPO.load("trained/ppo_CyLUCA_3")
env = LUCAEnv(traget_score=1, pad_length=64)

obs, info = env.reset()
env.render("human")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")
    print(f"\tReward: {rewards}")
    if dones:
        break