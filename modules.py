import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        '''A basic RNN cell that implements a single step of recurrent processing.

        The cell takes an input vector and previous hidden state, combines them through
        linear transformations and a non-linear activation, and outputs the new hidden state.

        Args:
            input_size (int): Size of the input vector
            hidden_size (int): Size of the hidden state vector

        Attributes:
            input_to_hidden (nn.Linear): Linear transformation from input to hidden state
            hidden_to_hidden (nn.Linear): Linear transformation of previous hidden state
            activation (nn.Tanh): Non-linear activation function
        '''
        super(RNNCell, self).__init__()
        self.register_buffer('input_dim', torch.tensor(input_dim))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim))
        self.W_h = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.f_h = nn.Tanh()

    def forward(self, x_t: torch.tensor, h_t_minus_1: torch.tensor) -> torch.tensor:
        combined_inputs = torch.cat([x_t, h_t_minus_1], dim=1)
        h_t = self.f_h(self.W_h(combined_inputs))
        return h_t


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(LSTMCell, self).__init__()

        self.register_buffer('input_dim', torch.tensor(input_dim))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim))

        self.W_f = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.f_f = nn.Sigmoid()

        self.W_i = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.f_i = nn.Sigmoid()

        self.W_c_hat = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.f_c_hat = nn.Tanh()

        self.W_o = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.f_o = nn.Sigmoid()

    def forward(self, x_t: torch.tensor, h_t_minus_1: torch.tensor, c_t_minus_1: torch.tensor) -> torch.tensor:
        combined_inputs = torch.cat([x_t, h_t_minus_1], dim=1)
        f_t = self.f_f(self.W_f(combined_inputs))
        i_t = self.f_i(self.W_i(combined_inputs))
        c_hat_t = self.f_c_hat(self.W_c_hat(combined_inputs))

        c_t = torch.mul(c_t_minus_1, f_t) + torch.mul(i_t, c_hat_t)

        o_t = self.f_o(self.W_o(combined_inputs))
        h_t = torch.mul(o_t, torch.tanh(c_t))

        return h_t, c_t
