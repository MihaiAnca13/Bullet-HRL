import math
import time

import numpy as np
import pybullet as p

pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
# jointPositions = [1.076, 0.060, 0.506, -2.020, -0.034, 2.076, 2.384, 0.03, 0.03]
jointPositions = [1.238, 0.505, 0.383, -2.003, -0.289, 2.456, 2.589, 0.04, 0.04]
rp = jointPositions


class FetchBulletSim(object):
    def __init__(self, bullet_client, offset, np_random):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        self.np_random = np_random

        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # Loading objects in
        self.bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)

        self.cubeId = self.bullet_client.loadURDF("cube.urdf", np.array([0.1, 0.3, -0.5]) + self.offset,
                                                  flags=flags)

        # self.markerId = self.bullet_client.loadURDF("assets/marker.urdf", np.array([0.1, 0.2, -0.55]) + self.offset,
        #                                             flags=flags)

        self.targetId = self.bullet_client.loadURDF("marker.urdf", np.array([0.15, 0.03, -0.55]) + self.offset,
                                                    flags=flags)
        self.bullet_client.changeVisualShape(self.targetId, -1, rgbaColor=[1, 0, 0, 1])

        self.finger_marker1Id = self.bullet_client.loadURDF("finger_marker.urdf",
                                                            np.array([0.2, 0.3, -0.5]) + self.offset, flags=flags)
        self.finger_marker2Id = self.bullet_client.loadURDF("finger_marker.urdf",
                                                            np.array([0.2, 0.3, -0.5]) + self.offset, flags=flags)

        # Loading the robotic arm
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)

        # self.control_dt = 1. / 240.
        # self.finger_target = 0
        # self.gripper_height = 0.2

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        # save normal initial state
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            # jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1

        self.initial_state = self.bullet_client.saveState()

        # generate and save pose with object grasped
        gripper_pos = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)[0]
        box_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        self.bullet_client.resetBasePositionAndOrientation(self.cubeId, gripper_pos, box_orientation)
        self.secondary_state = self.bullet_client.saveState()

        self.goal_pos = None
        self.fixed_orn = box_orientation

        self.reset()

    def _sample_goal(self):
        # self.goal_pos = self.np_random.uniform([-0.136, 0.03499, 0. - 0.718], [0.146, 0.0349, -0.457])
        self.goal_pos = self.np_random.uniform([-0.08, 0.03499, -0.66], [0.048, 0.0349, -0.55])
        orn = self.fixed_orn

        if self.np_random.random() <= 1:
            self.goal_pos[1] += self.np_random.uniform(0.14, 0.19)  # height offset

        self.bullet_client.resetBasePositionAndOrientation(self.targetId, self.goal_pos, orn)

    def _randomize_obj_start(self):
        # object_pos = self.np_random.uniform([-0.136, 0.03499, -0.718], [0.146, 0.0349, -0.457])
        object_pos = self.np_random.uniform([-0.07, 0.03499, -0.65], [0.047, 0.0349, -0.56])
        self.bullet_client.resetBasePositionAndOrientation(self.cubeId, object_pos, self.fixed_orn)

        # (0.14620011083425424, 0.034989999999999744, -0.4577289226704112)
        # (-0.13639202772104536, 0.03498999999999295, -0.7185703988375852)

    def reset(self):
        # reset initial positions
        if self.np_random.random() < 0.5:
            self.bullet_client.restoreState(self.initial_state)
            self._randomize_obj_start()
        else:
            self.bullet_client.restoreState(self.secondary_state)

        self._sample_goal()

        self.bullet_client.stepSimulation()

        return self._get_obs()

    def step(self, action, rendering=False, time_step=1. / 240., extract_image=False):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        current_gripper_state = self.get_gripper_state()
        gripper_ctrl = np.clip(current_gripper_state + gripper_ctrl, 0.0, 0.05)

        pos_ctrl *= 0.1  # limit maximum change in position
        rot_ctrl = self.fixed_orn  # fixed rotation of the end effector, expressed as a quaternion

        current_gripper_pos = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)[0]
        target_gripper_pos = current_gripper_pos + pos_ctrl

        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex,
                                                                   target_gripper_pos, rot_ctrl, ll, ul, jr, rp,
                                                                   maxNumIterations=20)

        # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, gripper_ctrl,
                                                     force= 100.)

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                     force=5 * 240.)

        self.bullet_client.stepSimulation()
        if rendering:
            time.sleep(time_step)

        c = self.bullet_client.getBasePositionAndOrientation(self.cubeId)[0]
        if c[1] < 0.034:
            c = list(c)
            c[1] = 0.034
            self.bullet_client.resetBasePositionAndOrientation(self.cubeId, c, self.fixed_orn)

        if not extract_image:
            return self._get_obs()
        else:
            viewMatrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.35, -0.13, 0],
                distance=1.2,
                yaw=38,
                pitch=-22,
                roll=0,
                upAxisIndex=1
            )
            projectionMatrix = p.computeProjectionMatrixFOV(
                fov=120,
                aspect=RENDER_WIDTH/RENDER_HEIGHT,
                nearVal=0.01,
                farVal=1.5
            )
            w, h, rgbPixels, depthPixels, segmentationMaskBuffer = self.bullet_client.getCameraImage(
                width=int(RENDER_WIDTH / 2),
                height=int(RENDER_HEIGHT / 2),
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix,
                renderer=p.ER_TINY_RENDERER,
            )
            return self._get_obs(), rgbPixels[:,:,:-1] # extracting 3 channels of of 4

    def _get_obs(self):
        gripper_pos, gripper_velp, gripper_velr = np.take(
            self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex, computeLinkVelocity=True), [0, 6, 7])
        gripper_state = self.get_gripper_state()

        obj_pos = self.bullet_client.getBasePositionAndOrientation(self.cubeId)[0]
        obj_velp, obj_velr = self.bullet_client.getBaseVelocity(self.cubeId)

        obj_rel_pos = np.array(obj_pos) - np.array(gripper_pos)

        obs = np.concatenate([
            np.array(gripper_pos), np.array(obj_pos), obj_rel_pos, np.array([gripper_state]), np.array(obj_velp),
            np.array(obj_velr), np.array(gripper_velp), np.array(gripper_velr)
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': np.array(obj_pos).copy(),
            'desired_goal': self.goal_pos.copy()
        }

    def get_gripper_pos(self):
        return self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)[0]

    def get_gripper_state(self):
        return self.bullet_client.getJointState(self.panda, 9)[0]

    def move_finger_markers(self, target_pos):
        target_pos1, target_pos2 = target_pos
        orn = self.bullet_client.getQuaternionFromEuler([0., 0., 0.])
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker1Id, target_pos1, orn)
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker2Id, target_pos2, orn)

    def close(self):
        del self.bullet_client

    def detect_gripper_collision(self):
        aabbMin, aabbMax = self.bullet_client.getAABB(self.panda, 9)
        # drawing collision line?
        # f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        # t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        # self.bullet_client.addUserDebugLine(f, t, [1, 1, 1])
        body_link_ids = self.bullet_client.getOverlappingObjects(aabbMin, aabbMax)
        body_ids = [x[0] for x in body_link_ids]
        if self.cubeId in body_ids:
            return True
        return False
