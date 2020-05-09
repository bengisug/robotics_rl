import torch


class DDPGActor(torch.nn.Module):

    def __init__(self, obs_size, action_size, nonlinear=torch.nn.ReLU, activation=torch.nn.modules.activation.Tanh,
                 hidden_layer_size=3, node_size=128):
        super(DDPGActor, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(obs_size, node_size))
        self.net.add_module("nonlinear_1", nonlinear())
        for i in range(hidden_layer_size-1):
            self.net.add_module("fc_" + str(i+2), torch.nn.Linear(node_size, node_size))
            self.net.add_module("nonlinear_" + str(i+2), nonlinear())
        self.net.add_module("head", torch.nn.Linear(node_size, action_size))
        self.net.add_module("activation", activation())

    def forward(self, x):
        return self.net(x)


class DDPGCritic(torch.nn.Module):

    def __init__(self, obs_size, action_size, nonlinear=torch.nn.ReLU, hidden_layer_size=3, node_size=128):
        super(DDPGCritic, self).__init__()
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