from torch_geometric.nn import MessagePassing
import torch.nn as nn

class CustomPoolingMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = GNNPoolingAdd()

    def forward(self, x, edge_index, graphPoolData):
        return self.layer(x, edge_index, graphPoolData)

class GNNPoolingMax(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    def forward(self, x, edge_index,graphPoolData):
        return self.propagate(edge_index, size=(graphPoolData["inputSize"], graphPoolData["outputSize"]), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class CustomPoolingAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = GNNPoolingAdd()

    def forward(self, x, edge_index, edge_attr, graphPoolData):
        return self.layer(x, edge_index, edge_attr, graphPoolData)

class GNNPoolingAdd(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_attr, graphPoolData):
        return self.propagate(edge_index, size=(graphPoolData["inputSize"], graphPoolData["outputSize"]), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, 1)
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out
