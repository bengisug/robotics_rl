""" Baxter base class to be used in environments.
"""
import pybullet as pb
from collections import namedtuple
from collections import OrderedDict


class BaxterRobot():
    CONTROL_TYPES = ("position", "torque", "velocity")

    def __init__(self, position=None):
        if position is None:
            position = [0, -0.8, 0.88]
        else:
            assert len(position) == 3, "Provide x, y and z in position!"

        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        baxterId = pb.loadURDF("baxter_description/urdf/toms_baxter.urdf",
                               useFixedBase=True,
                               flags=0 |
                                     pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

        self.bodyId = baxterId
        self.joints = OrderedDict()

        pb.resetBasePositionAndOrientation(
            baxterId, position, [0., 0., -1., -1])

        self.actuators = {name: Motor(joint) for name, joint
                          in self.joints.items() if name.find("finger") == -1}
        for jointId in range(pb.getNumJoints(baxterId)):
            joint = Joint(baxterId, jointId)
            if (joint.type == "JOINT_REVOLUTE"):
                self.actuators[joint.name] = Motor(joint)
            self.joints[joint.name] = joint

        self.actuators["left_gripper"] = Gripper(
            self.joints["l_gripper_l_finger_joint"],
            self.joints["l_gripper_r_finger_joint"]
        )
        self.actuators["right_gripper"] = Gripper(
            self.joints["r_gripper_l_finger_joint"],
            self.joints["r_gripper_r_finger_joint"]
        )
        self._initial_states = {name: joint.getState()
                                for name, joint in self.joints.items()}

    def controlActuators(self, controlInput, controlType="position"):
        # ControlInput : Dictionary
        for actuatorName, controlValue in controlInput.items():
            setattr(self.actuators[actuatorName], controlType, controlValue)

    def readActuators(self, actuatorNames):
        states = []
        for name in actuatorNames:
            actuator = self.actuators[name]
            states.append((
                actuator.position,
                actuator.velocity,
                actuator.torque)
            )
        return states

    def resetJoints(self):
        for jointName, state in self._initial_states.items():
            self.joints[jointName].resetState(state.position, state.velocity)


class Joint():
    JOINT_STATE = namedtuple("JOINT_STATE", "position velocity appliedTorque")
    JOINT_TYPES = ["JOINT_REVOLUTE", "JOINT_PRISMATIC",
                   "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"]
    VELOCITY_DIVIDER = 15

    def __init__(self, bodyIndex, jointIndex):
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex

        info = pb.getJointInfo(bodyUniqueId=self.bodyIndex,
                               jointIndex=self.jointIndex)
        self._name = info[1].decode("utf-8")
        self._type = self.JOINT_TYPES[info[2]]

        self._damping = info[6]
        self._friction = info[7]
        self._upperLimit = info[8]
        self._lowerLimit = info[9]
        self._maxForce = info[10]
        self._maxVelocity = info[11] / self.VELOCITY_DIVIDER

    def getState(self):
        state = pb.getJointState(self.bodyIndex, self.jointIndex)
        return self.JOINT_STATE(state[0], state[1], state[3])

    def resetState(self, position, velocity):
        pb.resetJointState(self.bodyIndex, self.jointIndex, position, velocity)

    def getWorldPosition(self):
        """ Abslute position of the link(joint) in catesian coordinates.
        """
        return pb.getLinkState(self.bodyIndex, self.jointIndex)[0]

    def getWorldOrientation(self):
        """ Orientation of the link(joint) in quaternion.
        """
        return pb.getLinkState(self.bodyIndex, self.jointIndex)[1]

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def damping(self):
        return self._damping

    @property
    def friction(self):
        return self._friction

    @property
    def upperLimit(self):
        return self._upperLimit

    @property
    def lowerLimit(self):
        return self._lowerLimit

    @property
    def maxForce(self):
        return self._maxForce

    @property
    def maxVelocity(self):
        return self._maxVelocity

    def fancyInfo(self):
        return """
    ----------- {} -----------
    Type: {}
    Damping: {}
    Friction: {}
    Upper Limit: {}
    Lower Limit: {}
    Maximum Force: {}
    Maximum Velocity: {}
    """.format(self.name,
               self.type,
               self.damping,
               self.friction,
               self.upperLimit,
               self.lowerLimit,
               self.maxForce,
               self.maxVelocity)


class Actuator(object):

    @staticmethod
    def setTorque(joint, torque):
        assert torque >= -1 and torque <= 1, (
            "Torque for {}: {} is out of range [-1, 1]".format(
                joint.name, torque))
        value = (torque + 1) * joint.maxForce / 2
        pb.setJointMotorControl2(bodyUniqueId=joint.bodyIndex,
                                 jointIndex=joint.jointIndex,
                                 controlMode=pb.TORQUE_CONTROL,
                                 force=value)

    @staticmethod
    def setPosition(joint, position):
        assert position >= -1 and position <= 1, (
            "Position for {}: {} is out of range [-1, 1]".format(
                joint.name, position))
        upper = joint.upperLimit
        lower = joint.lowerLimit
        value = (position + 1) * (upper - lower) / 2 + lower
        pb.setJointMotorControl2(bodyUniqueId=joint.bodyIndex,
                                 jointIndex=joint.jointIndex,
                                 controlMode=pb.POSITION_CONTROL,
                                 targetPosition=value,
                                 force=joint.maxForce,
                                 maxVelocity=joint.maxVelocity)

    @staticmethod
    def setVelocity(joint, velocity):
        assert velocity >= -1 and velocity <= 1, (
            "Velocity for {}: {} is out of range [-1, 1]".format(
                joint.name, velocity))
        value = (velocity + 1) * joint.maxVelocity / 2
        pb.setJointMotorControl2(bodyUniqueId=joint.bodyIndex,
                                 jointIndex=joint.jointIndex,
                                 controlMode=pb.VELOCITY_CONTROL,
                                 targetVelocity=velocity,
                                 force=joint.maxForce)


class Gripper(Actuator):

    def __init__(self, *joints):
        self.leftFinger, self.rightFinger = joints

    @property
    def position(self):
        leftState = self.leftFinger.getState()
        rightState = self.rightFinger.getState()
        return leftState.position - rightState.position

    @position.setter
    def position(self, value):
        Actuator.setPosition(self.leftFinger, -value)
        Actuator.setPosition(self.rightFinger, value)

    @property
    def velocity(self):
        leftState = self.leftFinger.getState()
        rightState = self.rightFinger.getState()
        return (leftState.velocity - rightState.velocity) / 2

    @velocity.setter
    def velocity(self, value):
        Actuator.setVelocity(self.leftFinger, -value)
        Actuator.setVelocity(self.rightFinger, value)

    @property
    def torque(self):
        leftState = self.leftFinger.getState()
        rightState = self.rightFinger.getState()
        return (leftState.appliedTorque - rightState.appliedTorque) / 2

    @torque.setter
    def torque(self, value):
        Actuator.setTorque(self.leftFinger, -value)
        Actuator.setTorque(self.rightFinger, value)


class Motor(Actuator):

    def __init__(self, joint):
        self.joint = joint

    @property
    def position(self):
        jointState = self.joint.getState()
        return jointState.position

    @position.setter
    def position(self, value):
        Actuator.setPosition(self.joint, value)

    @property
    def velocity(self):
        jointState = self.joint.getState()
        return jointState.velocity

    @velocity.setter
    def velocity(self, value):
        Actuator.setVelocity(self.joint, value)

    @property
    def torque(self):
        jointState = self.joint.getState()
        return jointState.appliedTorque

    @torque.setter
    def torque(self, value):
        Actuator.setTorque(self.joint, value)


if __name__ == "__main__":
    import os

    sid = pb.connect(pb.GUI)
    cwd = os.getcwd()
    pb.setAdditionalSearchPath(cwd + "../../data")
    pb.setGravity(0, 0, -9.8, sid)

    robot = BaxterRobot()
    robot.resetJoints()

    pb.setRealTimeSimulation(True)
    for i in range(1000000):
        pb.stepSimulation()
        robot.controlActuators({"left_e0": 1.0}, controlType="torque")

    print(robot.actuators.keys())
    print(robot.joints.keys())