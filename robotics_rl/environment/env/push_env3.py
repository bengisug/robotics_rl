""" Baxter Push gym environment 3
"""
import numpy as np
import math
import pybullet as pb
import itertools
from gym import spaces
from itertools import chain

from robotics_rl.environment.env.baxter_env import BaxterEnv
from robotics_rl.environment.robot.baxter_robot import BaxterRobot
from robotics_rl.environment.object.common import Plane, PhantomBall, Box, Table


class PushEnv3(BaxterEnv):

    def __init__(self, render, reward_type="shaped", control_type="position"):
        super().__init__(render)
        numJoints = 8
        self.control_type = control_type
        self.action_space = spaces.Box(np.ones(numJoints) * -1,
                                       np.ones(numJoints),
                                       dtype="float32")
        self.observation_space = spaces.Box(
            np.full(numJoints * 3 + 3 + 3 + 3, -np.Infinity),
            np.full(numJoints * 3 + 3 + 3 + 3, np.Infinity),
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
        self.tablePosition = [0.25, 0, 0]

        self.objectLength = 0.05
        self.objectPosition = self._generate_object()

        self.targetPosition = self._generate_target()

        self.reward_type = reward_type
        self.r = self.objectLength / math.sqrt(2)

    def _generate_target(self):
        self.targetPosition = []
        range = np.random.rand() * (self.tableSize[0] / 2 - 0.2) + 0.2
        angle = np.random.rand() * 2 * np.pi
        self.targetPosition.append(self.tablePosition[0] + range * np.cos(angle))
        self.targetPosition.append(self.tablePosition[1] + range * np.sin(angle))
        self.targetPosition.append(self.tableSize[2])

        if hasattr(self, "targetBall"):
            self.targetBall.setPosition(self.targetPosition)

        return self.targetPosition

    def _generate_object(self):
        self.objectPosition = []
        range = np.random.rand() * (self.tableSize[0] / 4)
        angle = np.random.rand() * 2 * np.pi
        self.objectPosition.append(self.tablePosition[0] + range * np.cos(angle))
        self.objectPosition.append(self.tablePosition[1] + range * np.sin(angle))
        self.objectPosition.append(self.tableSize[2])

        return self.objectPosition

    def _getObservation(self):
        observation = []
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()
        actuatorObs = self.robot.readActuators(self.robotActuatorNames)
        observation = list(
            chain(*actuatorObs, tip, self.objectPosition, self.targetPosition))
        return np.array(observation, dtype="float32")

    def _getReward(self):
        target = self.target
        objectPosition = self.objectPosition
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()

        tip_target = [0, 0, 0]
        norm = np.sqrt((np.power(target[0] - objectPosition[0], 2)) + (np.power(target[1] - objectPosition[1], 2)))
        tip_target[0] = objectPosition[0] - ((target[0] - objectPosition[0]) / norm) * (self.r)
        tip_target[1] = objectPosition[1] - ((target[1] - objectPosition[1]) / norm) * (self.r)
        tip_target[2] = objectPosition[2]

        tip_target2 = [0, 0, 0]
        tip_target2[0] = objectPosition[0] - ((target[0] - objectPosition[0]) / norm) * (self.r)
        tip_target2[1] = objectPosition[1] - ((target[1] - objectPosition[1]) / norm) * (self.r)
        tip_target2[2] = objectPosition[2]

        tip_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(tip, tip_target)))

        object_distance = math.sqrt(sum((x - y) ** 2
                                        for x, y in itertools.islice(zip(self.objectPosition, target), 2)))

        table_contact = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.table.id)
        cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)
        contact = False
        reward = 0
        sparse_reward = 0
        if cp:
            contact = True

        if self.reward_type == 'sparse':
            reward = int(object_distance <= 0.05)
        elif self.reward_type == 'shaped':
            reward = 0

            cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)
            reward = -tip_distance / 3 + -object_distance
            contact = False
            if cp:
                reward = -object_distance
                contact = True

            if object_distance < 0.05:
                reward = 1
                sparse_reward = 1

        if self.reward_type == "shaped":
            return reward
        elif self.reward_type == "sparse":
            return sparse_reward

    def getHindsightReward(self, objectPosition, target):
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()

        tip_target = [0, 0, 0]
        norm = np.sqrt((np.power(target[0] - objectPosition[0], 2)) + (np.power(target[1] - objectPosition[1], 2)))
        tip_target[0] = objectPosition[0] - ((target[0] - objectPosition[0]) / norm) * (self.r)
        tip_target[1] = objectPosition[1] - ((target[1] - objectPosition[1]) / norm) * (self.r)
        tip_target[2] = objectPosition[2]

        tip_target2 = [0, 0, 0]
        tip_target2[0] = objectPosition[0] - ((target[0] - objectPosition[0]) / norm) * (self.r)
        tip_target2[1] = objectPosition[1] - ((target[1] - objectPosition[1]) / norm) * (self.r)
        tip_target2[2] = objectPosition[2]

        tip_distance = math.sqrt(sum((x - y) ** 2
                                     for x, y in zip(tip, tip_target)))

        object_distance = math.sqrt(sum((x - y) ** 2
                                        for x, y in itertools.islice(zip(self.objectPosition, target), 2)))

        table_contact = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.table.id)
        cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)
        contact = False
        reward = 0
        sparse_reward = 0
        if cp:
            contact = True

        if self.reward_type == 'sparse':
            reward = int(object_distance <= 0.05)
        elif self.reward_type == 'shaped':
            reward = 0

            cp = pb.getContactPoints(self.robot.joints[self.fingerTipName].bodyIndex, self.box.objId)
            reward = -tip_distance / 3 + -object_distance
            contact = False
            if cp:
                reward = -object_distance
                contact = True

            if object_distance < 0.05:
                reward = 1
                sparse_reward = 1

        if self.reward_type == "shaped":
            return reward
        elif self.reward_type == "sparse":
            return sparse_reward

    def _reset(self):
        self.target = self._generate_target()
        self.targetBall.setPosition(self.target)

        self.posbox = self._generate_object()
        self.box.setPosition(self.posbox)

        return self._getObservation()

    def _betweenSteps(self):
        self.objectPosition = list(self.box.getPositionOrientation()[0])

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

        self.box = Box(size=0.05,
                       color=np.random.rand(),
                       position=self.objectPosition)

        self.targetBall = PhantomBall(radius=0.1,
                                      color=1,
                                      position=[1.1, 1.1, 1.1])

        self.robotActuators = [self.robot.actuators[name]
                               for name in self.robotActuatorNames]
        self.fingerJoint = self.robot.joints[self.fingerTipName]
