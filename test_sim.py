import time

import numpy as np
import pybullet as p
import pybullet_data as pd
from pybullet_utils import bullet_client as bc
from gym.utils import seeding

import FetchBulletSim as panda_sim

# video requires ffmpeg available in path
createVideo = True
fps = 240.
timeStep = 1. / fps

if createVideo:
    p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps=" + str(fps))
else:
    p = bc.BulletClient(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35,-0.13,0])
p.setAdditionalSearchPath('assets')

p.setTimeStep(timeStep)
p.setGravity(0,-9.8,0)

np_random, seed = seeding.np_random(1)
panda = panda_sim.FetchBulletSim(p,[0,0,0], np_random=np_random)
panda.control_dt = timeStep

# logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
# panda.bullet_client.submitProfileTiming("start")
for i in range(100000):
    # panda.bullet_client.submitProfileTiming("full_step")

    panda.step(target=np.array([-0.1014052646308704954, 0.22013282199293425, -0.4641104737542781, 0.01]))
    # panda.bullet_client.stepSimulation()
    # print(panda.get_gripper_pos())

    if createVideo:
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    if not createVideo:
        # pass
        time.sleep(timeStep)
# panda.bullet_client.submitProfileTiming()
# panda.bullet_client.submitProfileTiming()
# panda.bullet_client.stopStateLogging(logId)
