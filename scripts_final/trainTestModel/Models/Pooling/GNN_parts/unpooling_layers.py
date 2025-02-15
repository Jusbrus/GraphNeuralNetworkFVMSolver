from torch_geometric.nn import MessagePassing
import torch.nn as nn

class CustomUnpooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = GNNUnpooling()

    def forward(self, x, edge_index, edge_attr, graphUnpoolData):
        return self.layer(x, edge_index, edge_attr, graphUnpoolData)

class GNNUnpooling(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_attr, graphUnpoolData):
        return self.propagate(edge_index, size=(graphUnpoolData["inputSize"], graphUnpoolData["outputSize"]), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, 1)
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out
