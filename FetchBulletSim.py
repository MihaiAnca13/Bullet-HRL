import math

import numpy as np
import pybullet as p
import time

pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions = [0.997, 0.177, 0.700, -1.935, -0.129, 2.068, 0.966, 0.040, 0.040]
rp = jointPositions
RedJointPositions = [0.997, 0.177, 0.700, -1.935, -0.129, 2.068, 0.966, 0.040, 0.040]


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

        self.RedcubeId = self.bullet_client.loadURDF("cube_red.urdf", np.array([0.1, 0.3, -0.9]) + self.offset,
                                                  flags=flags)

        self.BluecubeId = self.bullet_client.loadURDF("cube_blue.urdf", np.array([0.1, 0.3, -0.5]) + self.offset,
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
        orn = p.getQuaternionFromEuler([-math.pi/2, math.pi,0])
        # eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.Redpanda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, -1.4]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)

        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.Bluepanda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0.2]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)

        # self.control_dt = 1. / 240.
        # self.finger_target = 0
        # self.gripper_height = 0.2

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.Bluepanda,
                                                9,
                                                self.Bluepanda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        # save normal initial state for Bluepanda
        for j in range(self.bullet_client.getNumJoints(self.Bluepanda)):
            self.bullet_client.changeDynamics(self.Bluepanda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.Bluepanda, j)
            # print("info=",info)
            # jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.Bluepanda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.Bluepanda, j, jointPositions[index])
                index = index + 1

        self.initial_state = self.bullet_client.saveState()

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.Redpanda,
                                                9,
                                                self.Redpanda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        # save normal initial state for Redpanda
        for j in range(self.bullet_client.getNumJoints(self.Redpanda)):
            self.bullet_client.changeDynamics(self.Redpanda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.Redpanda, j)
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.Redpanda, j, RedJointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.Redpanda, j, RedJointPositions[index])
                index = index + 1

        self.Red_initial_state = self.bullet_client.saveState()

        # generate and save pose with object grasped
        gripper_pos = self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]
        box_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, gripper_pos, box_orientation)
        self.secondary_state = self.bullet_client.saveState()

        self.goal_pos = None
        self.fixed_orn = box_orientation
        red_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., -math.pi/2, 0.])
        self.red_fixed_orn = red_orientation
        blue_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2, 0.])
        self.blue_fixed_orn = blue_orientation

        self.reset()

    def _sample_goal(self):
        #self.goal_pos = self.np_random.uniform([-0.136, 0.03499, 0. - 0.718], [0.146, 0.0349, -0.457])
        self.goal_pos = np.array([0.0, 0.28, -0.6])
        orn = self.fixed_orn

        #if self.np_random.random() <= 1:
            #self.goal_pos[1] += self.np_random.uniform(0.14, 0.23)  # height offset

        self.bullet_client.resetBasePositionAndOrientation(self.targetId, self.goal_pos, orn)

    def _randomize_obj_start(self):
        object_pos = self.np_random.uniform([-0.3, 0.03499, -0.3], [0.3, 0.03499, -0.2])
        self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, object_pos, self.fixed_orn)
        red_object_pos = self.np_random.uniform([-0.3, 0.03499, -0.7], [0.3, 0.03499, -0.8])
        self.bullet_client.resetBasePositionAndOrientation(self.RedcubeId, red_object_pos, self.fixed_orn)
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

    def step(self, action, rendering=False, time_step=1. / 240.):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        current_gripper_state = self.get_gripper_state()
        gripper_ctrl = np.clip(current_gripper_state + gripper_ctrl, 0.0, 0.05)

        pos_ctrl *= 0.1  # limit maximum change in position
        rot_ctrl = self.blue_fixed_orn  # fixed rotation of the end effector, expressed as a quaternion

        current_gripper_pos = self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]
        target_gripper_pos = current_gripper_pos + pos_ctrl

        jointPoses = self.bullet_client.calculateInverseKinematics(self.Bluepanda, pandaEndEffectorIndex,
                                                                   target_gripper_pos, rot_ctrl, ll, ul, jr, rp,
                                                                   maxNumIterations=20)
        #print('Blue:', jointPoses)
        # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.Bluepanda, i, self.bullet_client.POSITION_CONTROL, gripper_ctrl,
                                                     force= 100.)

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.Bluepanda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                     force=5 * 240.)

        self.bullet_client.stepSimulation()
        if rendering:
            time.sleep(time_step)

        c = self.bullet_client.getBasePositionAndOrientation(self.BluecubeId)[0]
        if c[1] < 0.034:
            c = list(c)
            c[1] = 0.034
            self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, c, self.fixed_orn)

        return self._get_obs()

    def get_gripper_pos(self):
        return self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]

    def get_gripper_state(self):
        return self.bullet_client.getJointState(self.Bluepanda, 9)[0]

    def move_finger_markers(self, target_pos):
        target_pos1, target_pos2 = target_pos
        orn = self.bullet_client.getQuaternionFromEuler([0., 0., 0.])
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker1Id, target_pos1, orn)
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker2Id, target_pos2, orn)

    def close(self):
        del self.bullet_client

    def detect_gripper_collision(self):
        aabbMin, aabbMax = self.bullet_client.getAABB(self.Bluepanda, 9)
        # drawing collision line?
        # f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        # t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        # self.bullet_clieoxnt.addUserDebugLine(f, t, [1, 1, 1])
        body_link_ids = self.bullet_client.getOverlappingObjects(aabbMin, aabbMax)
        body_ids = [x[0] for x in body_link_ids]
        if self.BluecubeId in body_ids:
            return True
        return False

    def RedStep(self, action, rendering=False, time_step=1. / 240.):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        current_gripper_state = self.get_red_gripper_state()
        gripper_ctrl = np.clip(current_gripper_state + gripper_ctrl, 0.0, 0.05)

        pos_ctrl *= 0.1  # limit maximum change in position
        rot_ctrl = self.red_fixed_orn  # fixed rotation of the end effector, expressed as a quaternion

        current_gripper_pos = self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex)[0]
        target_gripper_pos = current_gripper_pos + pos_ctrl

        jointPoses = self.bullet_client.calculateInverseKinematics(self.Redpanda, pandaEndEffectorIndex,
                                                                   target_gripper_pos, rot_ctrl, ll, ul, jr, rp,
                                                                   maxNumIterations=20)
        #print('Red:', jointPoses)
        # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.Redpanda, i, self.bullet_client.POSITION_CONTROL, gripper_ctrl,
                                                     force= 100.)

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.Redpanda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                     force=5 * 240.)

        self.bullet_client.stepSimulation()
        if rendering:
            time.sleep(time_step)

        c = self.bullet_client.getBasePositionAndOrientation(self.RedcubeId)[0]
        if c[1] < 0.034:
            c = list(c)
            c[1] = 0.034
            self.bullet_client.resetBasePositionAndOrientation(self.RedcubeId, c, self.fixed_orn)

        return self._get_obs()

    def get_red_gripper_pos(self):
        return self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex)[0]

    def get_red_gripper_state(self):
        return self.bullet_client.getJointState(self.Redpanda, 9)[0]

    def move_red_finger_markers(self, target_pos):
        target_pos1, target_pos2 = target_pos
        orn = self.bullet_client.getQuaternionFromEuler([0., 0., 0.])
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker1Id, target_pos1, orn)
        self.bullet_client.resetBasePositionAndOrientation(self.finger_marker2Id, target_pos2, orn)

    def red_close(self):
        del self.bullet_client

    def detect_red_gripper_collision(self):
        aabbMin, aabbMax = self.bullet_client.getAABB(self.Redpanda, 9)
        # drawing collision line?
        # f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        # t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        # self.bullet_client.addUserDebugLine(f, t, [1, 1, 1])
        body_link_ids = self.bullet_client.getOverlappingObjects(aabbMin, aabbMax)
        body_ids = [x[0] for x in body_link_ids]
        if self.RedcubeId in body_ids:
            return True
        return False

    def _get_obs(self):
        gripper_pos, gripper_velp, gripper_velr = np.take(
            self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex, computeLinkVelocity=True), [0, 6, 7])
        gripper_state = self.get_gripper_state()

        obj_pos = self.bullet_client.getBasePositionAndOrientation(self.BluecubeId)[0]
        obj_velp, obj_velr = self.bullet_client.getBaseVelocity(self.BluecubeId)

        obj_rel_pos = np.array(obj_pos) - np.array(gripper_pos)

        red_gripper_pos, red_gripper_velp, red_gripper_velr = np.take(
            self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex, computeLinkVelocity=True), [0, 6, 7])
        red_gripper_state = self.get_red_gripper_state()

        red_obj_pos = self.bullet_client.getBasePositionAndOrientation(self.RedcubeId)[0]
        red_obj_velp, red_obj_velr = self.bullet_client.getBaseVelocity(self.RedcubeId)

        red_obj_rel_pos = np.array(red_obj_pos) - np.array(red_gripper_pos)

        obs = np.concatenate([
            np.array(gripper_pos), np.array(obj_pos), obj_rel_pos, np.array([gripper_state]), np.array(obj_velp),
            np.array(obj_velr), np.array(gripper_velp), np.array(gripper_velr),
            np.array(red_gripper_pos), np.array(red_obj_pos), red_obj_rel_pos, np.array([red_gripper_state]),
            np.array(red_obj_velp), np.array(red_obj_velr), np.array(red_gripper_velp), np.array(red_gripper_velr)
        ])

        pos = np.concatenate([np.array(obj_pos), np.array(red_obj_pos)])
        gpos = np.concatenate([np.array(self.goal_pos), np.array(self.goal_pos)])

        return {
            'observation': obs.copy(),
            'achieved_goal': pos.copy(),
            'desired_goal': gpos.copy()
        }