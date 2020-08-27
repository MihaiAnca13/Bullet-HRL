import numpy as np
import pybullet as p

from FetchBulletEnv import FetchBulletEnv
from FetchBulletSim import FetchBulletSim


def goal_distance(state, target, thresholds):
    assert len(state) == len(target) == len(thresholds)
    for i in range(len(state)):
        if abs(state[i] - target[i]) > thresholds[i]:
            return False
    return True


assets_path = "Bullet-HRL/assets/"


class FetchBulletRenderEnv(FetchBulletEnv):

    def __init__(self, render_mode='DIRECT', time_step=1. / 240., seed=None, thresholds=np.array([0.03, 0.03, 0.03]),
                 assets_path=assets_path, n_substeps=5):
        super().__init__(render_mode, time_step, seed, thresholds, assets_path, n_substeps)

        self.close()

        p.connect(p.GUI,
                  options="--minGraphicsUpdateTimeMs=0 --mp4=\"records/pybullet_grasp.mp4\" --mp4fps=60")
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                     cameraTargetPosition=[0.35, -0.13, 0])
        p.setAdditionalSearchPath(assets_path)
        p.setTimeStep(time_step)
        p.setGravity(0, -9.8, 0)

        self.sim = FetchBulletSim(p, [0, 0, 0], self.np_random)
