import numpy as np
from gym.wrappers import TimeLimit
import time
from FetchBulletEnv import FetchBulletEnv, goal_distance

fps = 240.
time_step = 1. / fps

env = FetchBulletEnv(render_mode='GUI', time_step=time_step, assets_path='/home/mihai/PycharmProjects/HER-lightning/Bullet-HRL/assets/')
env = TimeLimit(env, max_episode_steps=50)
env.reset()

a = np.array([0.0, -0.4, 0, -0.0])
while True:
    for i in range(20):
        obs, reward, done, info = env.step(action=a)
    # print(env.goal)
    time.sleep(5)
    env.reset()


    if done:
        print('done, resetting')
        # a = np.random.uniform([-0.08, 0.03499, -0.66, 0.], [0.048, 0.2, -0.55, 0.0001])
        env.reset()