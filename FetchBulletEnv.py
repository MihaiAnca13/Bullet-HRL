import sys
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc

from FetchBulletSim import FetchBulletSim


def goal_distance(state, target, thresholds):
    assert len(state) == len(target) == len(thresholds)
    for i in range(len(state)):
        if abs(state[i] - target[i]) > thresholds[i]:
            return False
    return True


assets_path = "Bullet-HRL/assets/"


class FetchBulletEnv(gym.GoalEnv):

    def __init__(self, render_mode='DIRECT', time_step=1. / 240., seed=None, thresholds=np.array([0.03, 0.03, 0.03]), assets_path=assets_path):
        self.time_step = time_step
        self.render_mode = render_mode
        self.thresholds = thresholds

        self.seed(seed)

        if render_mode == 'DIRECT':
            bullet_instance = bc.BulletClient(p.DIRECT)
        elif render_mode == 'GUI':
            bullet_instance = bc.BulletClient(p.GUI)
            bullet_instance.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
            bullet_instance.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            bullet_instance.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
            bullet_instance.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                                       cameraTargetPosition=[0.35, -0.13, 0])
        else:
            sys.exit(f'FetchBulletEnv - render mode not supported: {render_mode}')

        # bullet_instance.setAdditionalSearchPath(pd.getDataPath())
        bullet_instance.setAdditionalSearchPath(assets_path)
        bullet_instance.setTimeStep(time_step)
        bullet_instance.setGravity(0, -9.8, 0)
        self.sim = FetchBulletSim(bullet_instance, [0, 0, 0], self.np_random)

        obs = self.sim.reset()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        obs = self.sim.reset()
        self.goal = self.sim.goal_pos.copy()
        return obs

    def close(self):
        self.sim.close()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs = self.sim.step(action)

        if self.render_mode == 'GUI':
            time.sleep(self.time_step)

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        return obs, reward, done, info

    def _is_success(self, achieved_goal, desired_goal):
        return goal_distance(achieved_goal, desired_goal, self.thresholds)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return int(self._is_success(achieved_goal, desired_goal)) - 1