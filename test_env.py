import numpy as np
from gym.wrappers import TimeLimit
import time
from FetchBulletEnv import FetchBulletEnv, goal_distance

fps = 240
time_step = 1. / fps

env = FetchBulletEnv(render_mode='GUI', time_step=time_step, assets_path='assets')
env = TimeLimit(env, max_episode_steps=50)
env.reset()

a = np.array([0.3, 0.3, 0.3, -0.0])
a = np.random.uniform([-0.08, 0.03499, -0.66, 0.], [0.048, 0.0349, -0.55, 0.04])
print("==========================a:", a)
for i in range(100000):
    obs, reward, done, info = env.step(action=a)

    if True:
        a = np.random.uniform([-0.08, 0.03499, -0.66, 0.], [0.048, 0.0349, -0.55, 0.04])
        print('done, resetting')
        env.reset()