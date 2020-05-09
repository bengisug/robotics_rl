""" Baxter Reach gym environment
"""
import pybullet as pb
import gym
import os


class BaxterEnv(gym.Env):

    BACKGROUND_COLOR = """
        --background_color_red=0.55
        --background_color_green=0.85
        --background_color_blue=0.95
    """
    STEP_FREQ = 10

    metadata = {"render.modes": ["gui", "direct"]}

    def __init__(self, render=True):
        super().__init__()
        self.render = render
        self.initialSetup = False

    @staticmethod
    def setupWorld(render):
        if render:
            sid = pb.connect(pb.GUI, options=BaxterEnv.BACKGROUND_COLOR)
        else:
            sid = pb.connect(pb.DIRECT, options=BaxterEnv.BACKGROUND_COLOR)

        path = os.path.abspath(__file__) + "../../data"
        pb.setAdditionalSearchPath(path)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        return sid

    def reset(self):
        if self.initialSetup is False:
            self.sid = self.setupWorld(self.render)
            self.initialSetup = True
            self._load_objects()
            self.initialState = pb.saveState()
        else:
            pb.restoreState(self.initialState)
        return self._reset()

    def step(self, action):
        assert self.initialSetup, "Call reset first!"
        return self._step(action)

    def stepSimulation(self):
        for i in range(240//BaxterEnv.STEP_FREQ):
            self._betweenSteps()
            pb.stepSimulation()

    def __del__(self):
        if self.initialSetup:
            pb.disconnect(self.sid)

    def _betweenSteps(self):
        pass

    def _reset(self):
        pass

    def _step(self):
        self.stepSimulation()
        pass

    def _load_objects(self):
        pass