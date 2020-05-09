import torch
import numpy as np

from copy import deepcopy

from robotics_rl.agent.model.agent import BaseAgent


class SACAgent(BaseAgent):

    def __init__(self, actor, critic1, critic2, lr_actor, lr_critic, gamma, grad_clip, transition_buffer,
                 transition_type,
                 batch_size,
                 tau,
                 action_shape, start_alpha,
                 end_alpha,
                 alpha_decay,
                 device):

        super(SACAgent, self).__init__(transition_type, batch_size, tau, gamma, grad_clip, device)

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)

        self.transition_buffer = transition_buffer
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha.detach())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.opt_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.opt_alpha = torch.optim.Adam((self.log_alpha,), lr=lr_actor)
        self.target_entropy = - np.product(action_shape)
        self.target_entropy = torch.tensor(self.target_entropy, dtype=torch.float32, device=self.device)

        self.alpha_decay = alpha_decay
        self.alpha_decay_step = 0
        self.to(self.device)

    def act(self, state):
        """
            Args:
                - state: Batch size 1 torch tensor.
        """
        self.eval()
        if state.shape[0] != 1:
            raise ValueError("Batch size of the state must be 1! Instead: {}".format(state.shape[0]))
        with torch.no_grad():
            action, logprob = self.actor.behavior_policy(state)
        action = action.to("cpu").squeeze(0)

        return action, logprob

    def act_greedy(self, state):
        """
            Args:
                - state: Batch size 1 torch tensor.
        """
        self.eval()
        if state.shape[0] != 1:
            raise ValueError("Batch size of the state must be 1! Instead: {}".format(state.shape[0]))
        with torch.no_grad():
            action = self.actor.greedy_policy(state)
        action = action.to("cpu").squeeze(0)

        return action

    def _target_q_predict(self, state, target_critic):
        """ Returns target values of actions by critic
        """
        with torch.no_grad():
            action, logprob = self.actor.behavior_policy(state)
            logprob = torch.sum(logprob, dim=-1).unsqueeze(-1)
            value = target_critic(state, action)
            value -= self.alpha * logprob
        return value

    def push_transition(self, *transition):
        if repr(self.transition_buffer) != "PrioritizedBuffer":
            self.transition_buffer.push(*transition)
        elif repr(self.transition_buffer) == "PrioritizedBuffer":
            tran = self.transition_buffer.tuple_type(*transition)
            with torch.no_grad():
                td_error = self._td_error(tran)
                self.transition_buffer.push(*transition, td_error,)

    def update(self):
        self.train()
        batch = self.transition_buffer.sample(self.batch_size)
        batch = self._batchtotorch(batch)

        # ----  Critic1 Update --------
        self.opt_critic1.zero_grad()

        loss_td1, td_error1 = self._td_loss(batch, self.critic1, self.target_critic1)
        loss_td1.backward()
        if self.grad_clip:
            self._clip_grad(self.critic1.parameters())
        self.opt_critic1.step()

        # ----  Critic2 Update --------
        self.opt_critic2.zero_grad()

        loss_td2, td_error2 = self._td_loss(batch, self.critic2, self.target_critic2)
        loss_td2.backward()
        if self.grad_clip:
            self._clip_grad(self.critic2.parameters())
        self.opt_critic2.step()

        # ---  Actor Update -------
        self.opt_actor.zero_grad()

        loss_policy, logprobs = self._policy_loss(batch)
        loss_policy.backward()
        if self.grad_clip:
            self._clip_grad(self.actor.parameters())
        self.opt_actor.step()

        # ---  Temperature Update -------
        self.opt_alpha.zero_grad()

        alpha_loss = self._alpha_loss(logprobs)
        alpha_loss.backward()
        self.opt_alpha.step()
        self.alpha = self.log_alpha.exp()

        # ----- Target Update -----
        self.update_target()
        if repr(self.transition_buffer) == "PrioritizedBuffer":
            td_error = (td_error1 + td_error2)/2
            self.transition_buffer.update_priority(td_error)

        return (loss_td1.item() + loss_td2.item()) / 2, -loss_policy.item()

    def _policy_loss(self, batch):
        """ Actions are resampled from each state in trajectory for every intention.
        """
        action, logprob = self.actor.learning_policy(batch.state)
        logprob = torch.sum(logprob, dim=-1).unsqueeze(-1)
        critic1 = self.critic1(batch.state, action)
        critic2 = self.critic2(batch.state, action)
        critic, indices = torch.min(torch.cat((critic1, critic2), 1), 1)
        critic = critic.unsqueeze(-1)
        policy_loss = (self.alpha * logprob - critic).mean()
        return policy_loss, logprob

    def _alpha_loss(self, logprobs):
        alpha_loss = (-self.log_alpha.exp() * (logprobs.detach() + self.target_entropy)).mean()
        return alpha_loss

    def _td_loss(self, batch, critic, target_critic):
        action_values = critic(batch.state, batch.action)
        target_values = self._target_q_predict(batch.next_state, target_critic)
        target = batch.reward + self.gamma * target_values * (1 - batch.terminal)
        td_loss = torch.nn.functional.mse_loss(action_values, target)
        with torch.no_grad():
            td_error = torch.abs(action_values - target).flatten()
        return td_loss, td_error

    def _td_error(self, batch):
        with torch.no_grad():
            state = self._totorch(batch.state, torch.float32).view(1, -1)
            next_state = self._totorch(batch.next_state, torch.float32).view(1, -1)
            action = batch.action.view(1, -1)
            critic1 = self.critic1(state, action)
            critic2 = self.critic2(state, action)
            critic, indices = torch.min(torch.cat((critic1, critic2), 1), 1)
            action_values = critic
            target_critic1 = self._target_q_predict(next_state, self.target_critic1)
            target_critic2 = self._target_q_predict(next_state, self.target_critic2)
            target_critic, indices = torch.min(torch.cat((target_critic1, target_critic2), 1), 1)
            target_values = target_critic.unsqueeze(-1)
            target = batch.reward + self.gamma * target_values.item() * (1 - batch.terminal[0])
            td_error = torch.abs(action_values - target)
        return td_error

    def update_target(self):
        for net, tarnet in zip((self.critic1, self.critic2),
                               (self.target_critic1, self.target_critic2)):

            for param, tparam in zip(net.parameters(), tarnet.parameters()):
                tparam.data += self.tau * (param.data - tparam.data)

    def generate_hindsight(self, get_hindsight_reward):
        assert repr(self.transition_buffer) == "HindsightBuffer"
        self.transition_buffer.generate_goal(get_hindsight_reward, self.actor.calculate_logprob, self._totorch)



