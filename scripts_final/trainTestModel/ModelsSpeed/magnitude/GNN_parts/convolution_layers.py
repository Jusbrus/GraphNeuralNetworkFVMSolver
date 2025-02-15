import torch
import time
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import matplotlib.pyplot as plt
import torch.nn as nn

class CustomGraphConv(nn.Module):
    def __init__(self,in_channels, out_channels, GNNData):
        super().__init__()
        self.pointCenter = GNNCenterPointCenterSimple(GNNData, in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr,graphData,weights_matrices):
        x = self.pointCenter(x, edge_index, edge_attr,graphData,weights_matrices)
        return x


class CustomGraphConvPointPoint(nn.Module):
    def __init__(self,in_channels, out_channels, GNNData):
        super().__init__()
        self.pointPoint = GNNpointPoint(GNNData, in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr,weights_matrices):
        x = self.pointPoint(x, edge_index, edge_attr,weights_matrices)
        return x


class GNNpointPoint(MessagePassing):
    def __init__(self, GNNData, in_channels, out_channels):
        super().__init__(aggr='add')
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(GNNData["nEdgeAttr"], out_channels, in_channels))
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters(in_channels, out_channels)

    def reset_parameters(self,in_channels, out_channels):
        scale_factor = torch.sqrt(torch.tensor(2.0 / (in_channels+out_channels)))

        with torch.no_grad():
            # self.weight_matrix.uniform_(-scale_factor, scale_factor)
            self.weight_matrix.uniform_(-scale_factor, scale_factor)

        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr,weights_matrices):
        conv_result = self.propagate(edge_index,weights_matrices=weights_matrices, x=x, edge_attr=edge_attr)
        return conv_result

    def message(self, x_j, edge_attr,weights_matrices):
        x_j = x_j.unsqueeze(-1)

        # weights_matrices = torch.einsum('ij,jkl->ikl', edge_attr, self.weight_matrix)
        # print(weights_matrices.shape)

        results = torch.matmul(weights_matrices, x_j).squeeze(-1)
        return results

    def update(self, aggr_out):
        return F.relu(aggr_out + self.bias)

# class GNNCenterPointCenterSimple(MessagePassing):
#     def __init__(self, GNNData, in_channels, out_channels):
#         super().__init__(aggr='add')
#         self.weight_matrix = torch.nn.Parameter(torch.Tensor(GNNData["nEdgeAttr"], out_channels, in_channels))
#         self.bias = Parameter(torch.empty(out_channels))
#         self.reset_parameters(in_channels, out_channels)
#
#     def reset_parameters(self, in_channels, out_channels):
#         scale_factor = torch.sqrt(torch.tensor(2.0 / (in_channels + out_channels)))
#
#         with torch.no_grad():
#             self.weight_matrix.uniform_(-scale_factor, scale_factor)
#
#         torch.nn.init.zeros_(self.bias)
#
#     def forward(self, x, edge_index, edge_attr, graphData, weights_matrices):
#         conv_result = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights_matrices=weights_matrices, size=(graphData["inputSize"], graphData["outputSize"]))
#         return conv_result
#
#     def message(self, x_j, edge_attr, weights_matrices):
#         x_j = x_j.unsqueeze(-1)
#         results = torch.matmul(weights_matrices, x_j).squeeze(-1)
#         return results
#
#     def update(self, aggr_out):
#         return F.relu(aggr_out + self.bias)

class GNNCenterPointCenterSimple(MessagePassing):
    def __init__(self, GNNData, in_channels, out_channels):
        super().__init__(aggr='add')
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(GNNData["nEdgeAttr"], out_channels, in_channels))
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters(in_channels, out_channels)

    def reset_parameters(self,in_channels, out_channels):
        scale_factor = torch.sqrt(torch.tensor(2.0 / (in_channels+out_channels)))

        with torch.no_grad():
            # self.weight_matrix.uniform_(-scale_factor, scale_factor)
            self.weight_matrix.uniform_(-scale_factor, scale_factor)


        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, graphData,weights_matrices):


        conv_result = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights_matrices=weights_matrices, size=(graphData["inputSize"], graphData["outputSize"]))
        # conv_result = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(graphData["inputSize"], graphData["inputSize"]))

        return conv_result

    def message(self, x_j, edge_attr,weights_matrices):
        x_j = x_j.unsqueeze(-1)
        # print("Fdsfdsfsdfsdf")
        # print(weights_matrices.shape)
        #
        # weights_matrices = torch.einsum('ij,jkl->ikl', edge_attr, self.weight_matrix)
        # print(weights_matrices.shape)
        results = torch.matmul(weights_matrices, x_j).squeeze(-1)
        return results

    def update(self, aggr_out):
        return F.relu(aggr_out + self.bias)

class SelfLoopLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SelfLoopLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x_result = self.lin(x)
        return x_result

