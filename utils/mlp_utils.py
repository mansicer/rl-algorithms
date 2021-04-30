import torch
import torch.nn as nn

def build_mlp_with_relu(input_dim, output_dim, hidden_dims=[32]):
    layer_list = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
    for i in range(1, len(hidden_dims)):
        layer_list.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layer_list)
