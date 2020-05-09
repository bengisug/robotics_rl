import torch
from copy import deepcopy

from robotics_rl.agent.model.agent import BaseAgent


class DDPGAgent(BaseAgent):

    def __init__(self, actor, critic, lr_actor, lr_critic, gamma, grad_clip, transition_buffer, transition_type,
                 batch_size,
                 tau,
                 random_process, start_sigma,
                 end_sigma,
                 sigma_decay,
                 device):
        super().__init__(transition_type, batch_size, tau, gamma, grad_clip, device)
        self.actor = actor
        self.critic = critic
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)

        self.transition_buffer = transition_buffer
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.sigma_decay = sigma_decay
        self.sigma_decay_step = 0
        self.random_process = random_process
        self.to(self.device)

    def act(self, state):
        action = self.act_greedy(state)
        self.random_process.sigma = (self.start_sigma - self.end_sigma) * (
            max(0, 1 - self.sigma_decay * self.sigma_decay_step)) + self.end_sigma
        action += self.random_process.noise()
        action.clamp_(-1, 1)
        self.sigma_decay_step += 1
        return action

    def act_greedy(self, state):
        """
            Args:
                - state: Batch size 1 torch tensor.
        """
        self.eval()
        if state.shape[0] != 1:
            raise ValueError("Batch size of the state must be 1! Instead: {}".format(state.shape[0]))
        with torch.no_grad():
            action = self.actor(state)
        action = action.to("cpu").squeeze(0)

        return action

    def _td_loss(self, batch):
        with torch.no_grad():
            target_action = self.target_actor(batch.next_state)
            target_value = self.target_critic(batch.next_state, target_action)

        current_value = self.critic(batch.state, batch.action)
        next_value = target_value * (1 - batch.terminal) * self.gamma + batch.reward
        td_loss = torch.nn.functional.smooth_l1_loss(current_value, next_value)
        with torch.no_grad():
            td_error = torch.abs(current_value - next_value).flatten()

        return td_loss, td_error

    def _td_error(self, batch):
        with torch.no_grad():
            state = self._totorch(batch.state, torch.float32).view(1, -1)
            next_state = self._totorch(batch.next_state, torch.float32).view(1, -1)
            action = batch.action.view(1, -1)
            target_action = self.target_actor(next_state)
            target_value = self.target_critic(next_state, target_action)
            current_value = self.critic(state, action)
            next_value = target_value * (1 - batch.terminal[0]) * self.gamma + batch.reward
            td_error = torch.abs(current_value - next_value)

        return td_error

    def _policy_loss(self, batch):
        action = self.actor(batch.state)
        value = self.critic(batch.state, action)
        return -torch.mean(value)

    def push_transition(self, *transition):
        if repr(self.transition_buffer) != "PrioritizedBuffer":
            self.transition_buffer.push(*transition)
        elif repr(self.transition_buffer) == "PrioritizedBuffer":
            tran = self.transition_buffer.tuple_type(*transition)
            with torch.no_grad():
                td_error = self._td_error(tran)
                self.transition_buffer.push(*transition, td_error)

    def update(self):
        self.train()
        batch = self.transition_buffer.sample(self.batch_size)
        batch = self._batchtotorch(batch)

        # ----  Critic Update --------
        self.opt_critic.zero_grad()

        loss_td, td_error = self._td_loss(batch)
        loss_td.backward()
        if self.grad_clip:
            self._clip_grad(self.critic.parameters())
        self.opt_critic.step()

        # ---  Actor Update -------
        self.opt_actor.zero_grad()
        loss_policy = self._policy_loss(batch)
        loss_policy.backward()
        if self.grad_clip:
            self.clip_grad(self.actor.parameters())
        self.opt_actor.step()

        # ----- Target Update -----
        self.update_target()
        if repr(self.transition_buffer) == "PrioritizedBuffer":
            self.transition_buffer.update_priority(td_error)

        return loss_td.item(), -loss_policy.item()

    def update_target(self):
        for net, tarnet in zip((self.actor, self.critic),
                               (self.target_actor, self.target_critic)):

            for param, tparam in zip(net.parameters(), tarnet.parameters()):
                tparam.data += self.tau * (param.data - tparam.data)

    def generate_hindsight(self, get_hindsight_reward):
        assert repr(self.transition_buffer) == "HindsightBuffer"
        self.transition_buffer.generate_goal(get_hindsight_reward)


