from stable_baselines3 import PPO
from env import LUCAEnv

model = PPO.load("trained/ppo_CyLUCA_8000")
env = LUCAEnv(traget_score=1, pad_length=128)

obs, info = env.reset()
env.render("human")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")
    print(f"\tReward: {rewards}")
    if dones:
        break