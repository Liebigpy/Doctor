
import numpy as np
from torch import nn
import torch
from typing import Union

device = "cuda"


Activation = Union[str, nn.Module]
_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}



def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        init_method=None,
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)

    return nn.Sequential(*layers)


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs).to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class RNDModel(nn.Module):
    def __init__(self, obs_shape, **kwargs):
        super().__init__()

        self.ob_dim = obs_shape
        self.output_size = 5
        self.n_layers = 2
        self.size = 400
        # self.optimizer_spec = optimizer_spec

        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_1
        )

        self.f_hat = build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_2
        )

        self.loss = nn.MSELoss()
        # self.optimizer = self.optimizer_spec.constructor(
        #     self.f_hat.parameters(),
        #     **self.optimizer_spec.optim_kwargs
        # )
        self.optimizer = torch.optim.Adam(self.f_hat.parameters(), lr=3e-4)

        self.f.to(device)
        self.f_hat.to(device)

    def forward(self, ob_no):
        # Get the prediction error for ob_no
        # Remember to detach the output of self.f!
        f_out = self.f(ob_no).detach()
        f_hat_out = self.f_hat(ob_no)
        #print("debug131", f_hat_out.shape) torch.Size([1, 4, 5])
        # print("debug130",torch.sqrt(torch.mean((f_hat_out - f_out) ** 2, dim=-1)).shape)torch.Size([1, 4])
        # print("debug131", torch.sqrt(torch.mean((f_hat_out - f_out) ** 2, dim=1)).shape)torch.Size([1, 5])
        return torch.sqrt(torch.mean((f_hat_out - f_out) ** 2, dim=-1))
        #return torch.sqrt(torch.mean((f_hat_out - f_out) ** 2, dim=1))


    def forward_np(self, ob_no):
        # ob_no = from_numpy(ob_no)
        error = self(ob_no)
        return error
        # return to_numpy(error)

    def update(self, ob_no):
        # Update f_hat using ob_no
        # Take the mean prediction error across the batch
        # error = self(from_numpy(ob_no))
        error = self(ob_no)
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print("debug154",loss.item(),loss.sum().item())194.39810180664062 194.39810180664062
        return {
            "rnd_loss": loss.item(),
            "rnd_error": error.mean().item()
        }
        #return loss.item()