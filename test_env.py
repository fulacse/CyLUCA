from env import LUCAEnv

env = LUCAEnv()
observation, info = env.reset()
env.render()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"\tReward: {reward}")

    if terminated or truncated:
        break

env.close()