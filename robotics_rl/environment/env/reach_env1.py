""" Baxter Reach gym environment 1
"""
import numpy as np
import math
import random
from gym import spaces
from itertools import chain

from robotics_rl.environment.env.baxter_env import BaxterEnv
from robotics_rl.environment.robot.baxter_robot import BaxterRobot
from robotics_rl.environment.object.common import Plane, PhantomBox, PhantomBall


class ReachEnv1(BaxterEnv):

    def __init__(self, render, box_scale=1, reward_type="shaped", control_type="position"):
        super().__init__(render)
        numJoints = 8
        self.control_type = control_type
        self.action_space = spaces.Box(np.ones(numJoints) * -1,
                                       np.ones(numJoints),
                                       dtype="float32")
        self.observation_space = spaces.Box(
            np.full(numJoints * 3 + 3 + 3, -np.Infinity),
            np.full(numJoints * 3 + 3 + 3, np.Infinity),
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
        self.targetBoxSize = np.array([0.5, 0.3, 0.3]) * box_scale
        self.targetBoxPosition = [0.25, 0.05, 1]

        self.target = self._generate_target()

        self.reward_type = reward_type
        self.rewardBoundary = 0.05

    def _generate_target(self):
        self.target = []
        for pos, bound in zip(self.targetBoxPosition, self.targetBoxSize):
            self.target.append(random.uniform(-bound, bound) + pos)

        if hasattr(self, "targetBall"):
            self.targetBall.setPosition(self.target)

        return self.target

    def _getObservation(self):
        observation = []
        tip = self.robot.joints[self.fingerTipName].getWorldPosition()
        actuatorObs = self.robot.readActuators(self.robotActuatorNames)
        observation = list(
            chain(*actuatorObs, tip, self.target))
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
        self.target = self._generate_target()
        self.targetBall.setPosition(self.target)
        return self._getObservation()

    def _betweenSteps(self):
        tipPosition = self.robot.joints[self.fingerTipName].getWorldPosition()
        self.fingerBall.setPosition(tipPosition)

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
        self.targetSpawnBox = PhantomBox(size=self.targetBoxSize,
                                         color=0.15,
                                         position=self.targetBoxPosition)
        self.targetBall = PhantomBall(radius=self.rewardBoundary,
                                      color=0.5,
                                      position=[1.1, 1.1, 1.1])
        self.fingerBall = PhantomBall(radius=0.03,
                                      color=0.85,
                                      position=[1, 1, 1])

        self.robotActuators = [self.robot.actuators[name]
                               for name in self.robotActuatorNames]
        self.fingerJoint = self.robot.joints[self.fingerTipName]