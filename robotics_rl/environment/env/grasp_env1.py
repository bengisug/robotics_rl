""" Baxter Grasp gym environment 1
"""
import numpy as np
import pybullet as pb
import math
from copy import deepcopy
from gym import spaces
from itertools import chain

from robotics_rl.environment.env.baxter_env import BaxterEnv
from robotics_rl.environment.robot.baxter_robot import BaxterRobot
from robotics_rl.environment.object.common import Plane, PhantomBall, Box, Table


class GraspEnv1(BaxterEnv):

    def __init__(self, render, reward_type="shaped", control_type="position"):
        super().__init__(render)
        numJoints = 8
        self.control_type = control_type
        self.action_space = spaces.Box(np.ones(numJoints) * -1,
                                       np.ones(numJoints),
                                       dtype="float32")
        self.observation_space = spaces.Box(
            np.full(numJoints * 3 + 3 + 3 + 3 + 8, -np.Infinity),
            np.full(numJoints * 3 + 3 + 3 + 3 + 8, np.Infinity),
            dtype="float32"
        )
        self.robotActuatorNames = (
            "right_s0",
            "right_s1",
            "right_e0",
            "right_e1",
            "right_w0",
            "right_w1",
            "right_w2",
            "right_gripper"
        )
        self.fingerTipName = (
            "r_gripper_r_finger_tip_joint"
        )

        self.tableSize = np.array([0.75, 0.5, 0.6])
        self.tablePosition = [0, 0, 0]

        self.objectRadius = 0.025
        self.objectPosition = self._generate_object()
        self.targetPosition = self._generate_target()

        self.reward_type = reward_type

    def _generate_target(self):
        self.targetPosition = self.objectPosition
        self.targetPosition[2] = self.objectPosition[2] + 0.1

        if hasattr(self, "targetBall"):
            self.targetBall.setPosition(self.targetPosition)

        return self.targetPosition

    def _generate_object(self):
        self.objectPosition = deepcopy(self.tablePosition)
        self.objectPosition[2] = self.tableSize[2]
        return self.objectPosition

    def _getObservation(self):
        observation = []
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()
        actuatorObs = self.robot.readActuators(self.robotActuatorNames)
        tip_orientation = self.robot.joints["r_gripper_r_finger_joint"].getWorldOrientation()
        box_position, box_orientation = self.box.getPositionOrientation()
        observation = list(
            chain(*actuatorObs, tip, tip_orientation, box_orientation, self.objectPosition, self.targetPosition))
        return np.array(observation, dtype="float32")

    def _getReward(self):
        target = self.target
        objectPosition = self.objectPosition

        reward = 0
        sparse_reward = 0

        tip_target = objectPosition
        tip_target[2] = objectPosition[2] + 0.1

        left_tip_target = objectPosition
        left_tip_target[0] = objectPosition[0] - 0.02

        right_tip_target = objectPosition
        right_tip_target[0] = objectPosition[0] + 0.02

        tip = tuple(map(np.mean, zip(self.robot.joints[self.fingerTipName].getWorldPosition(),
                                 self.robot.joints["r_gripper_l_finger_joint"].getWorldPosition())))

        tip_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(tip, tip_target)))

        tip_object_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(tip, objectPosition)))

        object_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(target, objectPosition)))

        object_move_distance = math.sqrt(sum((x - y) ** 2
                                        for x, y in zip(self.box_init_pos, objectPosition)))

        cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)

        left_finger = self.robot.joints["r_gripper_l_finger_joint"].getWorldPosition()
        right_finger = self.robot.joints["r_gripper_r_finger_joint"].getWorldPosition()

        left_finger_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(left_finger,left_tip_target)))
        right_finger_distance = math.sqrt(sum((x - y) ** 2
                                             for x, y in zip(right_finger, right_tip_target)))
        orientation = self.robot.joints["r_gripper_r_finger_joint"].getWorldOrientation()
        orientation_diff = np.square(np.subtract(orientation, (0, 1, 0, 0))).mean()
        gripper_dist = self.robot.actuators["right_gripper"].position
        actuator_values = np.array(self.robot.readActuators(self.robot.actuators.keys()))
        actuator_pos, actuator_vel, actuator_torque = np.split(actuator_values, 3, axis=-1)
        power = np.squeeze(actuator_vel * actuator_torque).sum()

        reward = (-tip_object_distance - 2 * (
                    left_finger_distance + right_finger_distance) - 5 * object_move_distance - object_distance)

        if self.objectPosition[2] < self.tableSize[2]:
            reward = -10
        table_contact = pb.getContactPoints(self.box.objId, self.table.id)
        if table_contact:
            reward -= 1
        else:
            sparse_reward = 1
            reward += 6

        if self.reward_type == "shaped":
            return reward/3
        else:
            return sparse_reward

    def getHindsightReward(self, objectPosition, target):
        reward = 0
        sparse_reward = 0

        tip_target = objectPosition
        tip_target[2] = objectPosition[2] + 0.1

        left_tip_target = objectPosition
        left_tip_target[0] = objectPosition[0] - 0.02

        right_tip_target = objectPosition
        right_tip_target[0] = objectPosition[0] + 0.02

        tip = tuple(map(np.mean, zip(self.robot.joints[self.fingerTipName].getWorldPosition(),
                                     self.robot.joints["r_gripper_l_finger_joint"].getWorldPosition())))

        tip_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(tip, tip_target)))

        tip_object_distance = math.sqrt(sum((x - y) ** 2
                                            for x, y in zip(tip, objectPosition)))

        object_distance = math.sqrt(sum((x - y) ** 2
                                        for x, y in zip(target, objectPosition)))

        object_move_distance = math.sqrt(sum((x - y) ** 2
                                             for x, y in zip(self.box_init_pos, objectPosition)))

        cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)

        left_finger = self.robot.joints["r_gripper_l_finger_joint"].getWorldPosition()
        right_finger = self.robot.joints["r_gripper_r_finger_joint"].getWorldPosition()

        left_finger_distance = math.sqrt(sum((x - y) ** 2
                                             for x, y in zip(left_finger, left_tip_target)))
        right_finger_distance = math.sqrt(sum((x - y) ** 2
                                              for x, y in zip(right_finger, right_tip_target)))
        orientation = self.robot.joints["r_gripper_r_finger_joint"].getWorldOrientation()
        orientation_diff = np.square(np.subtract(orientation, (0, 1, 0, 0))).mean()
        gripper_dist = self.robot.actuators["right_gripper"].position
        actuator_values = np.array(self.robot.readActuators(self.robot.actuators.keys()))
        actuator_pos, actuator_vel, actuator_torque = np.split(actuator_values, 3, axis=-1)
        power = np.squeeze(actuator_vel * actuator_torque).sum()

        reward = (-tip_object_distance - 2 * (
                left_finger_distance + right_finger_distance) - 5 * object_move_distance - object_distance)

        if self.objectPosition[2] < self.tableSize[2]:
            reward = -10
        table_contact = pb.getContactPoints(self.box.objId, self.table.id)
        if table_contact:
            reward -= 1
        else:
            sparse_reward = 1
            reward += 6

        if self.reward_type == "shaped":
            return reward / 3
        else:
            return sparse_reward

    def _reset(self):
        self.target = self._generate_target()
        self.targetBall.setPosition(self.target)

        self.posbox = self._generate_object()
        self.box.setPosition(self.posbox)
        self.box_init_pos, self.box_init_or = self.box.getPositionOrientation()

        return self._getObservation()

    def _betweenSteps(self):
        self.objectPosition = list(self.box.getPositionOrientation()[0])
        self._generate_target()

    def _step(self, action):
        assert len(action) == len(self.robotActuatorNames), (
            "Action length must be {}, given: {}!".format(
                len(self.robotActuatorNames),
                len(action)))

        controlInputs = {name: act for name, act in zip(
            self.robotActuatorNames, action)}
        self.robot.controlActuators(controlInputs, controlType=self.control_type)

        self.stepSimulation()
        reward = self._getReward()
        done = False
        return self._getObservation(), reward, done, {}

    def _load_objects(self):
        self.plane = Plane()
        self.robot = BaxterRobot()
        self.table = Table(position = self.tablePosition)

        self.box = Box(size=0.0225,
                       color=np.random.rand(),
                       position=self.objectPosition)

        self.targetBall = PhantomBall(radius=0.05,
                                      color=1,
                                      position=[1.1, 1.1, 1.1])

        self.robotActuators = [self.robot.actuators[name]
                               for name in self.robotActuatorNames]
        self.fingerJoint = self.robot.joints[self.fingerTipName]