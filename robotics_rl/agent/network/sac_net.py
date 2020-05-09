import torch


class SACActor(torch.nn.Module):

    def __init__(self, obs_size, action_size, nonlinear=torch.nn.ReLU, hidden_layer_size=2, node_size=256):
        super(SACActor, self).__init__()
        self.action_size = action_size
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(obs_size, node_size))
        self.net.add_module("nonlinear_1", nonlinear())
        for i in range(hidden_layer_size - 1):
            self.net.add_module("fc_" + str(i + 2), torch.nn.Linear(node_size, node_size))
            self.net.add_module("nonlinear_" + str(i + 2), nonlinear())
        self.net.add_module("head", torch.nn.Linear(node_size, action_size * 2))

        self.log_std_min = -20
        self.log_std_max = 2
        self.epsilon = 1e-6

    def forward(self, x):
        return self.net(x)

    def policy_dist(self, x):
        parameters = self(x)
        mu, log_std = parameters.split(self.action_size, dim=-1)
        std = log_std.clamp(min=self.log_std_min, max=self.log_std_max).exp()
        return torch.distributions.Normal(mu, std)

    def greedy_policy(self, x):
        parameters = self(x)
        mu = parameters.split(self.action_size, dim=-1)[0]
        action = mu.tanh()
        return action

    def learning_policy(self, x):
        dist = self.policy_dist(x)
        action = dist.rsample()
        logprob = dist.log_prob(action)
        action = action.tanh()
        logprob = logprob - torch.log(1 - action.pow(2) + self.epsilon)
        return action, logprob

    def behavior_policy(self, x):
        dist = self.policy_dist(x)
        action = dist.sample()
        logprob = dist.log_prob(action)
        action = action.tanh()
        logprob = logprob - torch.log(1 - action.pow(2) + self.epsilon)
        return action, logprob

    def calculate_logprob(self, x, action_tan):
        action = 0.5 * torch.log((1 + (action_tan + self.epsilon))/(1 - (action_tan + self.epsilon)))
        dist = self.policy_dist(x)
        logprob = dist.log_prob(action) - torch.log(1 - action_tan.pow(2) + self.epsilon)
        return logprob


class SACCritic(torch.nn.Module):

    def __init__(self, obs_size, action_size, nonlinear=torch.nn.ReLU, hidden_layer_size=2, node_size=256):
        super(SACCritic, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(obs_size + action_size, node_size))
        self.net.add_module("nonlinear_1", nonlinear())
        for i in range(hidden_layer_size - 1):
            self.net.add_module("fc_" + str(i + 2), torch.nn.Linear(node_size, node_size))
            self.net.add_module("nonlinear_" + str(i + 2), nonlinear())
        self.net.add_module("head", torch.nn.Linear(node_size, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)