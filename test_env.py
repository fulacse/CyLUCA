from gymnasium.utils.env_checker import check_env

from env import LUCAEnv

env = LUCAEnv()
check_env(env)

observation, info = env.reset()
env.render()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"\tReward: {reward}")
    #print(observation)

    if terminated or truncated:
        break

env.close()