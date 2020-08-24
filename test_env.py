import numpy as np
from gym.wrappers import TimeLimit

from FetchBulletEnv import FetchBulletEnv, goal_distance

fps = 120
time_step = 1. / fps

env = FetchBulletEnv(render_mode='GUI', time_step=time_step, assets_path='assets')
env = TimeLimit(env, max_episode_steps=50)
env.reset()

for i in range(100000):
    obs, reward, done, info = env.step(action=np.array([0., 0., 0., -0.04]))

    a = [0., 0., 0., 0.01]
    p = env.sim.bullet_client.getJointState(env.sim.panda, 10)[0]
    p = [0., 0., 0., p]
    b = goal_distance(p, a, np.array([0.03, 0.03, 0.03, 0.01]))
    print(b)


    # if done:
    #     print('done, resetting')
    #     env.reset()