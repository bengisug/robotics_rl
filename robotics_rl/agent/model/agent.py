import torch
from collections import OrderedDict


class BaseAgent(torch.nn.Module):

    def __init__(self, transition_type, batch_size, tau, gamma, grad_clip, device):
        super(BaseAgent, self).__init__()
        self.transition_type = transition_type
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = device

    def act(self, state):
        raise NotImplementedError

    def act_greedy(self, state):
        raise NotImplementedError

    def push_transition(self, *transition):
        self.transition_buffer.push(*transition)

    def update(self):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError

    def _clip_grad(self, parameters):
        for param in parameters:
            param.grad.data.clamp_(-1, 1)

    def _batchtotorch(self, batch):
        batch_dict = OrderedDict()
        for key, value in zip(batch._asdict().keys(), batch._asdict().values()):
            torch_value = self._totorch(value, torch.float32).view(self.batch_size, -1)
            batch_dict[key] = torch_value
        return self.transition_type(**batch_dict)

    def _totorch(self, container, dtype=torch.float32):
        if isinstance(container[0], torch.Tensor):
            tensor = torch.stack(container)
        else:
            tensor = torch.tensor(container, dtype=dtype)
        return tensor.to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)

    def load_model(self, path):
        state_dict = torch.load(open(path + ".p", "rb"), map_location=self.device)
        self.load_state_dict(state_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), open(path + ".p", "wb"))
