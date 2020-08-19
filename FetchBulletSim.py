import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions = [1.076, 0.060, 0.506, -2.020, -0.034, 2.076, 2.384, 0.03, 0.03]
rp = jointPositions

class FetchBulletSim(object):
  def __init__(self, bullet_client, offset):
    self.bullet_client = bullet_client
    self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
    self.offset = np.array(offset)
    
    #print("offset=",offset)
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    # Loading objects in    
    self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)

    self.cubeId = self.bullet_client.loadURDF("assets/cube.urdf",np.array( [0.1, 0.3, -0.5])+self.offset, flags=flags)

    self.markerId = self.bullet_client.loadURDF("assets/marker.urdf",np.array( [0.1, 0.2, -0.55])+self.offset, flags=flags)

    self.targetId = self.bullet_client.loadURDF("assets/marker.urdf",np.array( [0.15, 0.03, -0.55])+self.offset, flags=flags)
    self.bullet_client.changeVisualShape(self.targetId,-1,rgbaColor=[1,0,0,1])

    self.finger_marker1Id = self.bullet_client.loadURDF("assets/finger_marker.urdf",np.array( [0.2, 0.3, -0.5])+self.offset, flags=flags)
    self.finger_marker2Id = self.bullet_client.loadURDF("assets/finger_marker.urdf",np.array( [0.2, 0.3, -0.5])+self.offset, flags=flags)

    # Loading the robotic arm
    orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)

    index = 0
    self.control_dt = 1./240.
    self.finger_target = 0
    self.gripper_height = 0.2

    #create a constraint to keep the fingers centered
    c = self.bullet_client.createConstraint(self.panda,
                       9,
                       self.panda,
                       10,
                       jointType=self.bullet_client.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
    self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
 
    # save normal initial state
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      #print("info=",info)
      jointName = info[1]
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1

    self.initial_state = self.bullet_client.saveState()

    # generate and save pose with object grasped
    gripper_pos = self.bullet_client.getLinkState(self.panda, 11)[0]
    box_orientation = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    self.bullet_client.resetBasePositionAndOrientation(self.cubeId, gripper_pos, box_orientation)
    self.secondary_state = self.bullet_client.saveState()


    # self.t = 0.

  def _sample_goal(self):
    gripper_pos = self.bullet_client.getLinkState(self.panda, 11)[0] # panda_grasptarget
    goal_orientation = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])

    height_offset = 0.03
    if np.random.random() < 0.5:
      height_offset += np.random.uniform(0, 0.2)

    goal_pos = gripper_pos + np.random.uniform(-0.2, 0.2)


  def _randomize_obj_start(self):
    pass


  def reset(self):
    # reset initial positions
    if np.random.random() < 0.5:
      self.bullet_client.restoreState(self.initial_state)
      self._randomize_obj_start()
    else:
      self.bullet_client.restoreState(self.secondary_state)

    # 

  def step(self):
    # if self.state==6:
    #   self.finger_target = 0.01
    # if self.state==5:
    self.finger_target = 0.04 
    self.bullet_client.submitProfileTiming("step")


     #target for fingers
    for i in [9,10]:
      self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,self.finger_target ,force= 10)

    orn = self.bullet_client.getQuaternionFromEuler([0.,0.,0.])
    a = self.bullet_client.getLinkState(self.panda, 9, computeLinkVelocity=False, computeForwardKinematics=False)
    # need fixed quaternation
    self.bullet_client.resetBasePositionAndOrientation(self.finger_marker1Id, a[0], orn)

    a = self.bullet_client.getLinkState(self.panda, 10, computeLinkVelocity=False, computeForwardKinematics=False)
    self.bullet_client.resetBasePositionAndOrientation(self.finger_marker2Id, a[0], orn)


    self.bullet_client.submitProfileTiming()

# remainings of all class:
    # alpha = 0.9 #0.99
    #   #gripper_height = 0.034
    #   self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.03
    #     self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.2
      
    #   t = self.t
    #   self.t += self.control_dt
    #   pos = [self.offset[0]+0.2 * math.sin(1.5 * t), self.offset[1]+self.gripper_height, self.offset[2]+-0.6 + 0.1 * math.cos(1.5 * t)]
      
    #     pos, o = self.bullet_client.getBasePositionAndOrientation(self.cubeId)
    #     pos = [pos[0], self.gripper_height, pos[2]]
    #     self.prev_pos = pos

    #     pos = self.prev_pos
    #     diffX = pos[0] - self.offset[0]
    #     diffZ = pos[2] - (self.offset[2]-0.6)
    #     self.prev_pos = [self.prev_pos[0] - diffX*0.1, self.prev_pos[1], self.prev_pos[2]-diffZ*0.1]

      	
    #   orn = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    #   self.bullet_client.submitProfileTiming("IK")
    #   jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, pos, orn, ll, ul,
    #     jr, rp, maxNumIterations=20)
    #   self.bullet_client.submitProfileTiming()
    #   for i in range(pandaNumDofs):
    #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)