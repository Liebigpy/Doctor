import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb

device = "cuda"

class BaseCritic(object):
    def update(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        raise NotImplementedError

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs).to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(device)
        self.q_net_target.to(device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        ob_no = from_numpy(ob_no)
        ac_na = from_numpy(ac_na).to(torch.long)
        next_ob_no = from_numpy(next_ob_no)
        reward_n = from_numpy(reward_n)
        terminal_n = from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            next_actions = self.q_net(next_ob_no).argmax(dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)

        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        self.learning_rate_scheduler.step()

        return {'Training Loss': to_numpy(loss)}


    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = from_numpy(obs)
        qa_values = self.q_net(obs)
        return to_numpy(qa_values)
