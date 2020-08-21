import numpy as np

from FetchBulletEnv import FetchBulletEnv

fps = 240.
time_step = 1. / 240.

env = FetchBulletEnv(render_mode='GUI', time_step=time_step)
env.reset()

for i in range(100000):
    obs, reward, done, info = env.step(action=np.array([0., 0., 0., -0.01]))
