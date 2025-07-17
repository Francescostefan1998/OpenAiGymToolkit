import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
obs, info = env.reset()

print(obs)

obs, reward, terminated, truncated, info = env.step(0)
print(obs)

obs, reward, terminated, truncated, info = env.step(1)
print(obs)