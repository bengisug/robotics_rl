""" Hindsight Buffer
"""
import numpy as np
from copy import deepcopy
from collections import namedtuple, OrderedDict
from robotics_rl.agent.replay_buffer.uniform_buffer import UniformBuffer


HindsightTransition = namedtuple("HindsightTransition", ("transition", "timestamp"))

class HindsightBuffer(UniformBuffer):

    def __init__(self, size, tuple_type, mode='future'):
        super(HindsightBuffer, self).__init__(size, tuple_type)
        self.eps_queue = []
        self.timestamp = 0
        self.eps_cycle = 0
        self.K = 4
        self.mode = mode

    def __repr__(self):
        return "HindsightBuffer"

    def _push(self, *transition):
        super(HindsightBuffer, self).push(*transition)

    def push(self, *transition):
        htrans = HindsightTransition(transition=self.tuple_type(*transition), timestamp=self.timestamp)
        self._push(*transition)
        self.timestamp += 1
        if self.size != len(self.eps_queue):
            self.eps_queue.append(htrans)
        else:
            self.eps_queue[self.eps_cycle] = htrans
            self.eps_cycle = (self.eps_cycle + 1) % self.size

    def generate_goal(self, get_hindsight_reward, calculate_logprob=None, to_torch=None):
        if self.mode == "future":
            self._generate_goal_future(get_hindsight_reward, calculate_logprob, to_torch)
        elif self.mode == "success":
            self._generate_goal_success(get_hindsight_reward, calculate_logprob, to_torch)

    def _generate_goal_future(self, get_hindsight_reward, calculate_logprob=None, to_torch=None):
        for htrans in self.eps_queue:
            future_queue = list(filter(lambda t: t.timestamp >= htrans.timestamp, self.eps_queue))
            future_transition = (HindsightTransition(*zip(*future_queue)).transition)
            future_states = np.array(self.tuple_type(*zip(*future_transition)).next_state)
            goal_list = future_states[:, -6:-3]
            goals = []
            if len(goal_list) > 0:
                goals = goal_list[
                    np.random.choice(range(0, len(goal_list)), size=min(self.K, len(goal_list)), replace=False)]
            for goal in goals:
                transition = self._propose_goal(htrans.transition, goal, get_hindsight_reward, calculate_logprob,
                                                to_torch)
                self._push(*transition)

        self.eps_queue = []
        self.timestamp = 0

    def _generate_goal_success(self, get_hindsight_reward, calculate_logprob=None, to_torch=None):
        for htrans in self.eps_queue:
            if abs(htrans.transition.state[-6:-3] - htrans.transition.next_state[-6:-3]).any() > 1e-3:
                if self._check_reward(get_hindsight_reward, htrans.transition.state[-6:-3],
                                      htrans.transition.state[-3:], htrans.transition.state[-6:-3]):
                    goal = htrans.transition.next_state[-6:-3]
                    transition = self._propose_goal(htrans.transition, goal, get_hindsight_reward, calculate_logprob,
                                                    to_torch)
                    self._push(*transition)

        self.eps_queue = []
        self.timestamp = 0

    def _propose_goal(self, transition, goal, get_hindsight_reward, calculate_logprob=None, to_torch=None):
        hindsight_trans = deepcopy(transition)
        hindsight_trans.state[-3:] = goal
        hindsight_trans.next_state[-3:] = goal
        reward, done = get_hindsight_reward(hindsight_trans.state[-6:-3], goal)
        batch_dict = OrderedDict()
        for key, value in zip(hindsight_trans._asdict().keys(), hindsight_trans._asdict().values()):
            if key == "reward":
                batch_dict[key] = reward
            elif key == "terminal":
                batch_dict[key] = [done]
            elif key == 'logprob' and calculate_logprob is not None and to_torch is not None:
                torch_state = to_torch(hindsight_trans.state)
                logprob = calculate_logprob(torch_state, hindsight_trans.action).detach().unsqueeze(0)
                batch_dict[key] = logprob
            else:
                batch_dict[key] = value
        transition = self.tuple_type(**batch_dict)
        return transition

    def _check_reward(self, get_hindsight_reward, state, goal, proposed_goal):
        reward, done = get_hindsight_reward(state, goal)
        proposed_reward, proposed_done = get_hindsight_reward(state, proposed_goal)
        if proposed_reward > reward:
            return True
        else:
            return False

