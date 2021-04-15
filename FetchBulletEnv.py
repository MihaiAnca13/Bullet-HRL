import sys
import time

import gym
import numpy as np
import pybullet as p
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc

from FetchBulletSim import FetchBulletSim

# set goal distance threshold
def goal_distance(state, target, thresholds):
    assert len(state) == len(target) == len(thresholds)
    for i in range(len(state)):
        if abs(state[i] - target[i]) > thresholds[i]:
            return False
    return True

# define assests path
assets_path = "Bullet-HRL/assets/"

# create environment
class FetchBulletEnv(gym.GoalEnv):

    def __init__(self, render_mode='DIRECT', time_step=1. / 240., seed=None, thresholds=np.array([0.03, 0.03, 0.03]), assets_path=assets_path, n_substeps=5):

        # define environment properties
        self.time_step = time_step
        self.render_mode = render_mode
        self.thresholds = thresholds
        self.n_substeps = n_substeps

        # generate starting seed
        self.seed(seed)

        # initially render the environment
        if render_mode == 'DIRECT':
            bullet_instance = bc.BulletClient(p.DIRECT)
        elif render_mode == 'GUI':
            bullet_instance = bc.BulletClient(p.GUI)
            bullet_instance.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
            bullet_instance.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            bullet_instance.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
            bullet_instance.resetDebugVisualizerCamera(cameraDistance=2.3, cameraYaw=38, cameraPitch=-22,
                                                       cameraTargetPosition=[0.35, -0.13, 0])
        else:
            sys.exit(f'FetchBulletEnv - render mode not supported: {render_mode}')

        # set environment properties
        # bullet_instance.setAdditionalSearchPath(pd.getDataPath())
        bullet_instance.setAdditionalSearchPath(assets_path)
        bullet_instance.setTimeStep(time_step)
        bullet_instance.setGravity(0, -9.8, 0)
        self.sim = FetchBulletSim(bullet_instance, [0, 0, 0], self.np_random)

        # setup simulator and initialise goals and observations
        obs = self.sim.reset()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    # define starting seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # render simulalator
    def render(self, mode=''):
        raise NotImplementedError()

    # reset simulator
    def reset(self):
        obs = self.sim.reset()
        self.goal = self.sim.goal_pos.copy()
        return obs

    # close simulator
    def close(self):
        self.sim.close()

    # enact the next action and assess if goal has been met
    def step(self, action):

        # ensure the action is thecorrect shape
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # display action taken and get new observations
        for i in range(self.n_substeps):
            if self.render_mode == 'DIRECT':
                obs = self.sim.step(action)
            elif self.render_mode == 'GUI':
                obs = self.sim.step(action, rendering=True, time_step=self.time_step)
                time.sleep(self.time_step)
            else:
                obs = None

        # assess if goal has been achieved and get reward
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        return obs, reward, done, info

    # check if the goal state has been achieved
    def _is_success(self, achieved_goal, desired_goal, thresholds=None):
        if thresholds is not None:
            return goal_distance(achieved_goal, desired_goal, thresholds)
        else:
            return goal_distance(achieved_goal, desired_goal, self.thresholds)

    # check if blue gripper is grasping the blue box
    def is_gripper_grasping(self):
       return self.sim.detect_gripper_collision()

    # check if red gripper is grasping the box
    def is_red_gripper_grasping(self):
        return self.sim.detect_red_gripper_collision()

    # check if both arms have picked up their respective boxes
    def _pickup_success(self):
        if self.is_gripper_grasping() == True:
            if self.is_red_gripper_grasping() == True:
                return True
        return False

    # compute value of reward for the timestep
    def compute_reward(self, achieved_goal, desired_goal, info):
        if info is not None and 'thresholds' in info and info['thresholds'] is not None:
            reward = int(self._is_success(achieved_goal, desired_goal, thresholds=info['thresholds']) - 1) * 2
        else:
            reward = (int(self._is_success(achieved_goal, desired_goal)) - 1) * 2
        if reward == 2:
            reward += int(self._pickup_success)
        return reward

    # update marker in environment
    def update_markers(self, target):
        assert len(target) == 4
        marker_positions = [target[:-1].copy(), target[:-1].copy()]
        marker_positions[0][2] += target[-1]
        marker_positions[1][2] -= target[-1]
        self.sim.move_finger_markers(marker_positions)
