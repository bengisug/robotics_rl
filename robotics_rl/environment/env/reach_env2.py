""" Baxter Reach gym environment 2
"""
import numpy as np
import math
import random
from copy import deepcopy
from gym import spaces
from itertools import chain

from robotics_rl.environment.env.baxter_env import BaxterEnv
from robotics_rl.environment.robot.baxter_robot import BaxterRobot
from robotics_rl.environment.object.common import Plane, Table, PhantomBall, Box


class ReachEnv2(BaxterEnv):

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

        self.target = self._generate_target()

        self.reward_type = reward_type
        self.rewardBoundary = 0.05

    def _generate_target(self):
        self.target = deepcopy(self.objectPosition)
        self.target[2] = self.objectPosition[2] + 0.1

        if hasattr(self, "targetBall"):
            self.targetBall.setPosition(self.target)

        return self.target

    def _generate_object(self):
        self.objectPosition = []
        for pos, bound in zip(self.tablePosition, self.tableSize/2):
            self.objectPosition.append(random.uniform(-bound, bound) + pos)
        self.objectPosition[2] = self.tableSize[2]
        return self.objectPosition

    def _getObservation(self):
        observation = []
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()
        actuatorObs = self.robot.readActuators(self.robotActuatorNames)
        observation = list(
            chain(*actuatorObs, tip, self.objectPosition, self.target))
        return np.array(observation, dtype="float32")

    def _getReward(self):
        target = self.target
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()

        distance = math.sqrt(sum((x - y) ** 2
                                 for x, y in zip(tip, target)))
        reward = 0
        if self.reward_type == 'shaped':
            reward = -distance
        elif self.reward_type == 'sparse':
            reward = int(distance <= self.rewardBoundary)

        return reward

    def getHindsightReward(self, tip, target):
        distance = math.sqrt(sum((x - y) ** 2
                                 for x, y in zip(tip, target)))
        reward = 0

        if self.reward_type == 'shaped':
            reward = -distance
        elif self.reward_type == 'sparse':
            reward = int(distance <= self.rewardBoundary)

        return reward, False

    def _reset(self):
        self.posbox = self._generate_object()
        self.box.setPosition(self.posbox)
        self.target = self._generate_target()
        self.targetBall.setPosition(self.target)
        return self._getObservation()

    def _betweenSteps(self):
        tipPosition = self.robot.joints[self.fingerTipName].getWorldPosition()
        self.fingerBall.setPosition(tipPosition)
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
        self.table = Table(position=self.tablePosition)
        self.targetBall = PhantomBall(radius=self.rewardBoundary,
                                      color=0.5,
                                      position=[1.1, 1.1, 1.1])
        self.fingerBall = PhantomBall(radius=0.03,
                                      color=0.85,
                                      position=[1, 1, 1])

        self.box = Box(size=self.objectLength,
                       color=np.random.rand(),
                       position=self.objectPosition)

        self.robotActuators = [self.robot.actuators[name]
                               for name in self.robotActuatorNames]
        self.fingerJoint = self.robot.joints[self.fingerTipName]
