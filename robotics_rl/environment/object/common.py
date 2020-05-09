""" Common objects loading functions
"""

import pybullet as pb
from matplotlib import cm


class CommonObject(object):
    """ Base class for common objects.
    """

    COLORMAP = cm.get_cmap("Pastel1")

    def __init__(self):
        pass
        # Check if the data folder is loaded

    def getPositionOrientation(self):
        return pb.getBasePositionAndOrientation(self.objId)

    def setPosition(self, position):
        _, orientation = self.getPositionOrientation()
        pb.resetBasePositionAndOrientation(self.objId, position, orientation)

    def setOrientation(self, orientation):
        position, _ = self.getPositionOrientation()
        pb.resetBasePositionAndOrientation(self.objId, position, orientation)


class Plane(CommonObject):
    """ Plane object.
        Load plane to the simulation at the initialization.
    """

    def __init__(self, zaxis=0, lateralFriction=0.99, restitution=0.99):
        super().__init__()
        self.id = pb.loadURDF("plane/plane.urdf", [0, 0, zaxis], useFixedBase=True)
        pb.changeDynamics(bodyUniqueId=self.id,
                          linkIndex=-1,
                          lateralFriction=lateralFriction,
                          restitution=restitution)
        pb.configureDebugVisualizer(pb.COV_ENABLE_PLANAR_REFLECTION, 1)


class Table(CommonObject):
    """ Table model.
    """

    def __init__(self,position):
        super().__init__()
        self.id = pb.loadURDF("table/table.urdf", position, useFixedBase=True)
        pb.configureDebugVisualizer(pb.COV_ENABLE_PLANAR_REFLECTION, 1)


class PhantomBall(CommonObject):
    """ Collusion free sphere object for visualization purposes.
    """

    def __init__(self, radius, color, position, alpha=0.4, orientation=[0, 0, 0, 1]):
        color = list(CommonObject.COLORMAP(color))
        color[3] = alpha
        visId = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=color)
        objId = pb.createMultiBody(baseMass=0.0,
                                   baseVisualShapeIndex=visId,
                                   basePosition=position,
                                   baseOrientation=orientation)
        self.objId = objId


class Ball(CommonObject):
    """ Interactive box object.
    """

    def __init__(self, radius, color, position, alpha=1, orientation=[0, 0, 0, 1]):

        color = list(CommonObject.COLORMAP(color))
        color[3] = alpha

        colBallId = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=color)

        objId = pb.createMultiBody(baseMass=0.1,
                                   baseCollisionShapeIndex=colBallId,
                                   basePosition=position,
                                   baseOrientation=orientation)
        self.objId = objId


class PhantomBox(CommonObject):
    """ Collusion free box object for visualization purposes.
    """

    def __init__(self, size, color, position,
                 alpha=0.4, orientation=[0, 0, 0, 1]):
        color = list(CommonObject.COLORMAP(color))
        color[3] = alpha
        visId = pb.createVisualShape(pb.GEOM_BOX,
                                     halfExtents=size,
                                     rgbaColor=color)
        objId = pb.createMultiBody(baseMass=0.0,
                                   baseVisualShapeIndex=visId,
                                   basePosition=position,
                                   baseOrientation=orientation)
        self.objId = objId


class Box(CommonObject):
    """ Interactive box object.
    """

    def __init__(self, size, color, position, alpha=1, orientation=[0, 0, 0, 1]):

        color = list(CommonObject.COLORMAP(color))
        color[3] = alpha

        colBoxId = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size, size, size])

        objId = pb.createMultiBody(baseMass=0.1,
                                   baseCollisionShapeIndex=colBoxId,
                                   basePosition=position,
                                   baseOrientation=orientation)
        self.objId = objId