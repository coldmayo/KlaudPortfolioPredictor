import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        gate_size = hidden_size * 4

        self.W = nn.ParameterList()
        self.U = nn.ParameterList()
        self.b = nn.ParameterList()

        for layer in range(layer_num):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.W.append(nn.Parameter(torch.Tensor(layer_input_size, gate_size)))
            self.U.append(nn.Parameter(torch.Tensor(hidden_size, gate_size)))
            self.b.append(nn.Parameter(torch.Tensor(gate_size)))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for layer in range(self.layer_num):
            self.W[layer].data.uniform_(-stdv, stdv)
            self.U[layer].data.uniform_(-stdv, stdv)
            self.b[layer].data.uniform_(-stdv, stdv)

    def forward(self, x, init_states="None"):
        batch, seq_len, _ = x.size()



