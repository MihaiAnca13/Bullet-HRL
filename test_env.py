import numpy as np
from gym.wrappers import TimeLimit
import time
from FetchBulletEnv import FetchBulletEnv, goal_distance

fps = 60
time_step = 1. / fps

env = FetchBulletEnv(render_mode='GUI', time_step=time_step, assets_path='assets')
env = TimeLimit(env, max_episode_steps=50)
env.reset()

a = np.array([0.3, 0.3, 0.3, -0.0])
for i in range(100000):
    obs, reward, done, info = env.step(action=a)

    if done:
        print('done, resetting')
        env.reset()