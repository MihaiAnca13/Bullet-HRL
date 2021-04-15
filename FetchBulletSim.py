import math

import numpy as np
import pybullet as p
import time

pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

# lower limits for null space
ll = [-7]*pandaNumDofs
# upper limits for null space
ul = [7]*pandaNumDofs
# joint ranges for null space
jr = [7]*pandaNumDofs
# rest poses for null space
jointPositions = [0.997, 0.177, 0.700, -1.935, -0.129, 2.068, 0.966, 0.040, 0.040]
brp = jointPositions
RedJointPositions = [0.997, 0.177, 0.700, -1.935, -0.129, 2.068, 0.966, 0.040, 0.040]
rrp = RedJointPositions

# create objects and their control
class FetchBulletSim(object):

    def __init__(self, bullet_client, offset, np_random):

        # set up universal parameters
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        self.np_random = np_random

        # allow changes to be made to URDF files
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # load objects in
        self.bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)

        self.BluecubeId = self.bullet_client.loadURDF("cube_blue.urdf", np.array([0.1, 0.3, -0.5]) + self.offset,
                                                    flags=flags)
        self.RedcubeId = self.bullet_client.loadURDF("cube_red.urdf", np.array([0.1, 0.3, -0.5]) + self.offset,
                                                    flags=flags)

        self.bluetargetId = self.bullet_client.loadURDF("marker.urdf", np.array([0.15, 0.03, -0.55]) + self.offset,
                                                    flags=flags)
        self.bullet_client.changeVisualShape(self.bluetargetId, -1, rgbaColor=[0.1, 0.3, 0.9, 1])
        self.redtargetId = self.bullet_client.loadURDF("marker.urdf", np.array([0.15, 0.03, -0.55]) + self.offset,
                                                    flags=flags)
        self.bullet_client.changeVisualShape(self.redtargetId, -1, rgbaColor=[0.9, 0.1, 0.3, 1])

        # load the robotic arm
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        self.Bluepanda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0.2]) + self.offset, orn,
                                                    useFixedBase=True, flags=flags)
        orn = p.getQuaternionFromEuler([-math.pi / 2, math.pi, 0])
        self.Redpanda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, -1.4]) + self.offset, orn,
                                                    useFixedBase=True, flags=flags)

        # create a constraint to keep the fingers centered for Bluepanda
        c = self.bullet_client.createConstraint(self.Bluepanda,
                                                9,
                                                self.Bluepanda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # save normal initial state for Bluepanda
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.Bluepanda)):
            self.bullet_client.changeDynamics(self.Bluepanda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.Bluepanda, j)
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.Bluepanda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.Bluepanda, j, jointPositions[index])
                index = index + 1

        # create a constraint to keep the fingers centered for Redpanda
        c = self.bullet_client.createConstraint(self.Redpanda,
                                                9,
                                                self.Redpanda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # save normal initial state for Redpanda
        index = 0
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

        # save initial states for both arms
        self.initial_state = self.bullet_client.saveState()

        # generate and save pose with object grasped
        box_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        blue_gripper_pos = self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]
        self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, blue_gripper_pos, box_orientation)
        red_gripper_pos = self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex)[0]
        self.bullet_client.resetBasePositionAndOrientation(self.RedcubeId, red_gripper_pos, box_orientation)
        self.secondary_state = self.bullet_client.saveState()

        # generate a fixed orientation for the end effectors
        self.goal_pos = None
        self.fixed_orn = box_orientation
        blue_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2, 0.])
        self.blue_fixed_orn = blue_orientation
        red_orientation = self.bullet_client.getQuaternionFromEuler([math.pi / 2., -math.pi / 2, 0.])
        self.red_fixed_orn = red_orientation

        # reset objects to be in the new positions
        self.reset()

    # set goal positions for both arms
    def _sample_goal(self):
        #self.goal_pos = np.array([0.0, 0.28, -0.5, 0.0, 0.28, -0.7])   # static goal
        self.goal_pos = self.np_random.uniform([-0.3, 0.03499, -0.3, -0.3, 0.03499, -0.8], [0.3, 0.0349, -0.2, 0.3, 0.03499, -0.7])
        orn = self.fixed_orn

        # height offset for random goal
        if self.np_random.random() <= 1:
            self.goal_pos[1] += self.np_random.uniform(0.14, 0.23)
            self.goal_pos[4] += self.np_random.uniform(0.14, 0.23)

        # set targets for the respective arms
        self.bullet_client.resetBasePositionAndOrientation(self.bluetargetId, self.goal_pos[:3], orn)
        self.bullet_client.resetBasePositionAndOrientation(self.redtargetId, self.goal_pos[3:], orn)

    # function to randomly spawn the boxes within the environment, but on their respective sides
    def _randomize_obj_start(self):
        object_pos = self.np_random.uniform([-0.3, 0.05, -0.3], [0.3, 0.05, -0.2])
        self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, object_pos, self.fixed_orn)
        red_object_pos = self.np_random.uniform([-0.3, 0.05, -0.8], [0.3, 0.05, -0.7])
        self.bullet_client.resetBasePositionAndOrientation(self.RedcubeId, red_object_pos, self.fixed_orn)

    # function to reset the arm positions and randomly spawn the boxes
    def reset(self):
        if self.np_random.random() < 0.5:     # spawn boxes randomly in the tray
            self.bullet_client.restoreState(self.initial_state)
            self._randomize_obj_start()
        else:                                 # spawn boxes within the grippers
            self.bullet_client.restoreState(self.secondary_state)

        self._sample_goal()
        self.bullet_client.stepSimulation()
        return self._get_obs()

    # execute actions for the blue arm
    def step(self, action, rendering=False, time_step=1. / 240.):

        # ensure action received is only for the blue arm
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        # execute gripper control
        current_gripper_state = self.get_gripper_state()
        gripper_ctrl = np.clip(current_gripper_state + gripper_ctrl, 0.0, 0.05)

        pos_ctrl *= 0.1  # limit maximum change in position
        rot_ctrl = self.blue_fixed_orn  # fixed rotation of the end effector, expressed as a quaternion

        # compare current position and goal position, calculate the inverse kinematics
        current_gripper_pos = self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]
        target_gripper_pos = current_gripper_pos + pos_ctrl
        jointPoses = self.bullet_client.calculateInverseKinematics(self.Bluepanda, pandaEndEffectorIndex,
                                                                   target_gripper_pos, rot_ctrl, ll, ul, jr, brp,
                                                                   maxNumIterations=20)

        # set the joints the the corresponding calculated values
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.Bluepanda, i, self.bullet_client.POSITION_CONTROL, gripper_ctrl,
                                                     force= 100.)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.Bluepanda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                     force=5 * 240.)

        # ensure end effector orientation is as desired
        c = self.bullet_client.getBasePositionAndOrientation(self.BluecubeId)[0]
        if c[1] < 0.034:
            c = list(c)
            c[1] = 0.034
            self.bullet_client.resetBasePositionAndOrientation(self.BluecubeId, c, self.fixed_orn)

        # return new position observations
        return self._get_obs()

    # get the current end effector position for Bluepanda
    def get_gripper_pos(self):
        return self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex)[0]

    # get the current state of the gripper for Bluepanda
    def get_gripper_state(self):
        return self.bullet_client.getJointState(self.Bluepanda, 9)[0]

    # close the simulation
    def close(self):
        del self.bullet_client

    # detect is the gripper has grasped the box
    def detect_gripper_collision(self):
        aabbMin, aabbMax = self.bullet_client.getAABB(self.Bluepanda, 9)
        body_link_ids = self.bullet_client.getOverlappingObjects(aabbMin, aabbMax)
        body_ids = [x[0] for x in body_link_ids]
        if self.BluecubeId in body_ids:
            return True
        return False

    # execute actions for the red arm
    def RedStep(self, action, rendering=False, time_step=1. / 240.):

        # ensure action received is only for the red arm
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        # execute gripper control
        current_gripper_state = self.get_red_gripper_state()
        gripper_ctrl = np.clip(current_gripper_state + gripper_ctrl, 0.0, 0.05)

        pos_ctrl *= 0.1  # limit maximum change in position
        rot_ctrl = self.red_fixed_orn  # fixed rotation of the end effector, expressed as a quaternion

        # compare current position and goal position, calculate the inverse kinematics
        current_gripper_pos = self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex)[0]
        target_gripper_pos = current_gripper_pos + pos_ctrl
        jointPoses = self.bullet_client.calculateInverseKinematics(self.Redpanda, pandaEndEffectorIndex,
                                                                   target_gripper_pos, rot_ctrl, ll, ul, jr, rrp,
                                                                   maxNumIterations=20)

        # set the joints the the corresponding calculated values
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.Redpanda, i, self.bullet_client.POSITION_CONTROL, gripper_ctrl,
                                                     force= 100.)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.Redpanda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                     force=5 * 240.)

        # render the motion of both the red and blue arm
        self.bullet_client.stepSimulation()
        if rendering:
            time.sleep(time_step)

        # ensure end effector orientation is as desired
        c = self.bullet_client.getBasePositionAndOrientation(self.RedcubeId)[0]
        if c[1] < 0.034:
            c = list(c)
            c[1] = 0.034
            self.bullet_client.resetBasePositionAndOrientation(self.RedcubeId, c, self.fixed_orn)

        # return new position observations
        return self._get_obs()

    # get the current end effector position for Redpanda
    def get_red_gripper_pos(self):
        return self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex)[0]

    # get the current state of the gripper for Redpanda
    def get_red_gripper_state(self):
        return self.bullet_client.getJointState(self.Redpanda, 9)[0]

    # close the simulation
    def red_close(self):
        del self.bullet_client

    # detect is the gripper has grasped the box
    def detect_red_gripper_collision(self):
        aabbMin, aabbMax = self.bullet_client.getAABB(self.Redpanda, 9)
        body_link_ids = self.bullet_client.getOverlappingObjects(aabbMin, aabbMax)
        body_ids = [x[0] for x in body_link_ids]
        if self.RedcubeId in body_ids:
            return True
        return False

    # get observations about the object positions
    def _get_obs(self):

        # get end effector information for Bluepanda
        gripper_pos, gripper_velp, gripper_velr = np.take(
            self.bullet_client.getLinkState(self.Bluepanda, pandaEndEffectorIndex, computeLinkVelocity=True), [0, 6, 7])
        gripper_state = self.get_gripper_state()

        # get object position information for the blue cube
        obj_pos = self.bullet_client.getBasePositionAndOrientation(self.BluecubeId)[0]
        obj_velp, obj_velr = self.bullet_client.getBaseVelocity(self.BluecubeId)
        obj_rel_pos = np.array(obj_pos) - np.array(gripper_pos)

        # get end effector information for Redpanda
        red_gripper_pos, red_gripper_velp, red_gripper_velr = np.take(
            self.bullet_client.getLinkState(self.Redpanda, pandaEndEffectorIndex, computeLinkVelocity=True), [0, 6, 7])
        red_gripper_state = self.get_red_gripper_state()

        # get object postion information for the red cube
        red_obj_pos = self.bullet_client.getBasePositionAndOrientation(self.RedcubeId)[0]
        red_obj_velp, red_obj_velr = self.bullet_client.getBaseVelocity(self.RedcubeId)
        red_obj_rel_pos = np.array(red_obj_pos) - np.array(red_gripper_pos)

        # form a full array containing all information about the objects in the environment
        obs = np.concatenate([
            np.array(gripper_pos), np.array(obj_pos), obj_rel_pos, np.array([gripper_state]), np.array(obj_velp),
            np.array(obj_velr), np.array(gripper_velp), np.array(gripper_velr),
            np.array(red_gripper_pos), np.array(red_obj_pos), red_obj_rel_pos, np.array([red_gripper_state]),
            np.array(red_obj_velp), np.array(red_obj_velr), np.array(red_gripper_velp), np.array(red_gripper_velr)
        ])
        pos = np.concatenate([np.array(obj_pos), np.array(red_obj_pos)])
        gpos = np.array(self.goal_pos)

        # return information about the current observations, positioning and goal
        return {
            'observation': obs.copy(),
            'achieved_goal': pos.copy(),
            'desired_goal': gpos.copy()
        }