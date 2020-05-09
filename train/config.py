import re
import importlib


class Config:

    def __init__(self, args):
        self.environment_type = args.environment_type
        self.env_name = args.environment_type
        self.env = args
        self.model_name = args.agent_type
        self.transition_type = args.agent_type
        self.buffer_type = args.buffer_type
        self.buffer_name = args.buffer_type
        self.train_file = args.agent_type
        self.hindsight = args.hindsight
        self.hindsight_mode = args.hindsight_mode
        self.buffer = args.buffer_size

    @property
    def env_name(self):
        return self._env_name

    @env_name.setter
    def env_name(self, env_name):
        index = re.search('[A-Z][a-z]*', env_name).end()
        self._env_name = env_name[:index].lower() + "_" + env_name[index:].lower()

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, args):
        env_module = importlib.import_module("." + self.env_name,
                                             "robotics_rl.environment.env")
        env_class = getattr(env_module, self.environment_type)
        if self.env_name != "push_env1":
            self._env = env_class(render=args.render, control_type=args.control_type, reward_type=args.reward_type)
        else:
            self._env = env_class(render=args.render, control_type=args.control_type, reward_type=args.reward_type,
                                  range_coef=args.range_coef, angle_coef=args.angle_coef)

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, agent_type):
        index = re.search('[A-Z]*', agent_type).end()
        self._model_name = agent_type[:index - 1].lower() + "_" + agent_type[index - 1:].lower()

    @property
    def buffer_name(self):
        return self._buffer_name

    @buffer_name.setter
    def buffer_name(self, buffer_type):
        index = re.search('[A-Z][a-z]*[A-Z]', buffer_type).end()
        self._buffer_name = buffer_type[:index - 1].lower() + "_" + buffer_type[index - 1:].lower()

    @property
    def train_file(self):
        return self._train_file

    @train_file.setter
    def train_file(self, agent_type):
        index = re.search('[A-Z]*', agent_type).end()
        self._train_file = "train.train_" + agent_type[:index - 1].lower()

    @property
    def transition_type(self):
        return self._transition_type

    @transition_type.setter
    def transition_type(self, agent_type):
        transition_module = importlib.import_module(".transition", "robotics_rl.agent.replay_buffer")
        index = re.search('[A-Z]*', agent_type).end()
        transition_type = agent_type[:index - 1] + "Transition"
        self._transition_type = getattr(transition_module, transition_type)

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buffer_size):
        if self.hindsight is True:
            buffer_module = importlib.import_module(".hindsight_buffer",
                                                    "robotics_rl.agent.replay_buffer")
            buffer_class = getattr(buffer_module, "HindsightBuffer")
            self._buffer = buffer_class(buffer_size, self.transition_type, self.hindsight_mode)
        else:
            buffer_module = importlib.import_module("." + self.buffer_name,
                                                 "robotics_rl.agent.replay_buffer")
            buffer_class = getattr(buffer_module, self.buffer_type)
            self._buffer = buffer_class(buffer_size, self.transition_type)
