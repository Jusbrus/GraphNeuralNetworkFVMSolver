import torch
import torch.nn as nn
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


sys.path.append("/home/justinbrusche/scripts_final/trainTestModel/Models/AggregationTest")
from GNN_parts.convolution_layers import *
from GNN_parts.pooling_layers import *
from GNN_parts.unpooling_layers import *


def plot_mesh_triangles(fig, ax, cellpoints, x,y, noise_values, title, zlim=None,show_bar=True):
    # Unpack meshpoints into x and y coordinates
    # print("ds",noise_values.shape)

    noise_values = noise_values.flatten()
    # Clip the noise values to the 1st and 99th percentiles
    lower_bound = np.percentile(noise_values, 1)
    upper_bound = np.percentile(noise_values, 99)
    noise_values_clipped = np.clip(noise_values, lower_bound, upper_bound)

    # Create an array of triangle vertices using the indices from cellpoints
    triangles = np.array([[(x[i], y[i]) for i in cell] for cell in cellpoints])

    # Create a collection of polygons to represent the triangles
    collection = PolyCollection(triangles, array=noise_values_clipped, cmap='viridis',
                                )

    # Add the collection to the axes
    ax.add_collection(collection)

    # Auto scale the plot limits based on the data
    ax.autoscale()
    ax.set_aspect('equal')

    # Add a color bar
    if show_bar:
        fig.colorbar(collection, ax=ax, shrink=0.5, aspect=5, label='Noise Value')

    # Set labels and title
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title(title)

    # Set z-axis limits if provided (for 3D consistency)
    if zlim:
        ax.set_clim(zlim)


class _ConvBlock1(nn.Module):

    def __init__(self, in_channels_centers,in_channels_faces, out_channels_centerFace, out_channels, GNNData):
        super(_ConvBlock1, self).__init__()
        self.layerCenterFace = CustomGraphConv(in_channels_centers, out_channels_centerFace,GNNData["centerFace"])
        self.layerFacePoint = CustomGraphConv(out_channels_centerFace+in_channels_faces, out_channels, GNNData["facePoint"])

        # self.layerPointPoint = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, xCellCenters, xFace, graphData, GNNDataForward):
        xCellCenters = self.layerCenterFace(xCellCenters, graphData["centerFace"].edge_index,
                                            graphData["centerFace"].edge_attr, GNNDataForward["centerFace"])

        x = torch.cat([xCellCenters, xFace], dim=-1)

        x = self.layerFacePoint(x, graphData["facePoint"].edge_index, graphData["facePoint"].edge_attr,
                                     GNNDataForward["facePoint"])

        # x = self.layerPointPoint(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        return x


class _ConvBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock2, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData, poolDataForward):


        x = self.pooling(x, graphPoolData.edge_index, graphPoolData.edge_attr, poolDataForward)

        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        return x

class _ConvBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock3, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        return x

class _ConvBlock4(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock4, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        return x

class _ConvBlock4Less(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock4Less, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.layer3 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.layer4 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData["pooling"].edge_index,graphPoolData["pooling"].edge_attr,poolDataForward["pooling"])
        print("sum",torch.sum(x).item())
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer3(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer4(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        print("sum",torch.sum(x).item())

        x = self.unpooling(x, graphPoolData["unpooling"].edge_index,graphPoolData["unpooling"].edge_attr,poolDataForward["unpooling"])
        return x

class _ConvBlock5(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock5, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        # self.layer3 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        # self.layer4 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData["pooling"].edge_index,graphPoolData["pooling"].edge_attr,poolDataForward["pooling"])
        print("sum",torch.sum(x).item())
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer3(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer4(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        print("sum",torch.sum(x).item())

        x = self.unpooling(x, graphPoolData["unpooling"].edge_index,graphPoolData["unpooling"].edge_attr,poolDataForward["unpooling"])
        return x

class _ConvBlock6(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock6, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock7(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock7, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock8(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock8, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        # self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData,graphPoolData,poolDataForward):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock9(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock9, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConv(out_channels, out_channels, GNNData["pointCenter"])

    def forward(self, x, graphData, GNNDataForward):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x,graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"])
        return x

class _ConvBlock10(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        self.layer1 = SelfLoopLayer( )

    def forward(self, x):
        x = self.layer1(x)

        return x


class GraphUNetSmall(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_Small_modi_version2")
        super().__init__()
        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,12,16, GNNData)
        self.in1 = nn.InstanceNorm1d(16)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.in2 = nn.InstanceNorm1d(16)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.in3 = nn.InstanceNorm1d(16)
        self.convN_4 = _ConvBlock4(16, 32, GNNData)
        self.in4 = nn.InstanceNorm1d(32)
        self.convN_5 = _ConvBlock5(32, 32, GNNData)
        self.in5 = nn.InstanceNorm1d(32)
        self.convN_6 = _ConvBlock6(64, 32, GNNData)
        self.in6 = nn.InstanceNorm1d(32)
        self.convN_7 = _ConvBlock7(48, 32, GNNData)
        self.in7 = nn.InstanceNorm1d(32)
        self.convN_8 = _ConvBlock8(48, 16, GNNData)
        self.in8 = nn.InstanceNorm1d(16)
        self.convN_9 = _ConvBlock9(32, 16, GNNData)
        self.in9 = nn.InstanceNorm1d(16)
        self.final = _ConvBlock10(16, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward):
        # xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        # xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0])
        # convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        # convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        # convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])
        # convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_6out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        # convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        # convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out


class GraphUNetLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_Small_modi_version2")
        super().__init__()
        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,12,16, GNNData)
        self.in1 = nn.InstanceNorm1d(16)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.in2 = nn.InstanceNorm1d(16)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.in3 = nn.InstanceNorm1d(16)
        self.convN_4 = _ConvBlock4Less(16, 32, GNNData)
        self.in4 = nn.InstanceNorm1d(32)
        # self.convN_5 = _ConvBlock5(32, 32, GNNData)
        # self.in5 = nn.InstanceNorm1d(32)
        # self.convN_6 = _ConvBlock6(32, 32, GNNData)
        # self.in6 = nn.InstanceNorm1d(32)
        self.convN_7 = _ConvBlock7(48, 32, GNNData)
        self.in7 = nn.InstanceNorm1d(32)
        self.convN_8 = _ConvBlock8(48, 16, GNNData)
        self.in8 = nn.InstanceNorm1d(16)
        self.convN_9 = _ConvBlock9(32, 16, GNNData)
        self.in9 = nn.InstanceNorm1d(16)
        self.final = _ConvBlock10(16, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward):
        # xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        # xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0])
        # convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        # convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        # convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2])
        # convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        # convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        # convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out

class GraphUNetMedium(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_Medium_modi_version2")
        super().__init__()
        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,20,24, GNNData)
        self.in1 = nn.InstanceNorm1d(24)
        self.convN_2 = _ConvBlock2(24, 24, GNNData)
        self.in2 = nn.InstanceNorm1d(24)
        self.convN_3 = _ConvBlock3(24, 24, GNNData)
        self.in3 = nn.InstanceNorm1d(24)
        self.convN_4 = _ConvBlock4(24, 48, GNNData)
        self.in4 = nn.InstanceNorm1d(48)
        self.convN_5 = _ConvBlock5(48, 48, GNNData)
        self.in5 = nn.InstanceNorm1d(48)
        self.convN_6 = _ConvBlock6(96, 48, GNNData)
        self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(72, 48, GNNData)
        self.in7 = nn.InstanceNorm1d(48)
        self.convN_8 = _ConvBlock8(72, 24, GNNData)
        self.in8 = nn.InstanceNorm1d(24)
        self.convN_9 = _ConvBlock9(48, 24, GNNData)
        self.in9 = nn.InstanceNorm1d(24)
        self.final = _ConvBlock10(24, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward):
        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0])
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_6out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out


class GraphUNetFullScale(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_FullScale_modi_version2")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)
        self.convN_1 = _ConvBlock1(2, 4,28,32, GNNData)
        self.in1 = nn.InstanceNorm1d(32)
        self.convN_2 = _ConvBlock2(32, 32, GNNData)
        self.in2 = nn.InstanceNorm1d(32)
        self.convN_3 = _ConvBlock3(32, 32, GNNData)
        self.in3 = nn.InstanceNorm1d(32)
        self.convN_4 = _ConvBlock4(32, 64, GNNData)
        self.in4 = nn.InstanceNorm1d(64)
        self.convN_5 = _ConvBlock5(64, 64, GNNData)
        self.in5 = nn.InstanceNorm1d(64)
        self.convN_6 = _ConvBlock6(128, 64, GNNData)
        self.in6 = nn.InstanceNorm1d(64)
        self.convN_7 = _ConvBlock7(96, 64, GNNData)
        self.in7 = nn.InstanceNorm1d(64)
        self.convN_8 = _ConvBlock8(96, 32, GNNData)
        self.in8 = nn.InstanceNorm1d(32)
        self.convN_9 = _ConvBlock9(64, 32, GNNData)
        self.in9 = nn.InstanceNorm1d(32)
        self.final = _ConvBlock10(32, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward):
        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0])
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_6out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out

def print_tree(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_tree(value, indent + 4)

if __name__ == '__main__':
    import pickle
    from anytree import Node, RenderTree
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # test_case_path = "/home/justinbrusche/datasets/foundationalParameters/case_8"
    test_case_path = "/home/justinbrusche/datasets/Step1_1000/case_0"

    edgeAttrName = "MeshGraphNetCNNWithFace"
    edgeAttrName = "FVMModelConv"
    edgeAttrName = "conv_4666"


    # edgeAttrName = "test"

    file_path = f"{test_case_path}/embedding_{edgeAttrName}"
    with open(file_path, 'rb') as file:
        GNNdataDict = pickle.load(file)

    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    nNodes = dataDictMeshes[0]["nNodesDualMesh"]
    nFaces = dataDictMeshes[0]["Nfaces"]

    # print(dataDictMeshes[0]["cellCenters"])
    x1, y1 = 8.1, 8  # Replace with your actual target coordinates

    # Calculate the Euclidean distance between each cell and the target point
    distances = np.sqrt((dataDictMeshes[0]["cellCenters"][:, 0] - x1) ** 2 + (dataDictMeshes[0]["cellCenters"][:, 1] - y1) ** 2)

    closest_cell_index = np.argmin(distances)

    batchSize = 1
    inputChannels = 2
    device = "cuda"
    # device = "cpu"

    # x_tensor = torch.randn((batchSize, nNodes,inputChannels)).to(device)
    # face_tensor = torch.randn((batchSize, nFaces,4)).to(device)

    x_tensor = torch.zeros(batchSize, nNodes,inputChannels).to(device)
    face_tensor = torch.zeros(batchSize, nFaces,4).to(device)

    x_tensor[:,closest_cell_index,:] = 1
    # x_tensor[:,0,:] = 1
    # x_tensor[:,10443,:] = 1
    # x_tensor[:,1145,:] = 1
    # x_tensor[:,2665,:] = 1


    print("x_tensor.shape",x_tensor.shape)

    file_path = f"{test_case_path}/pointTypes.pt"
    # pointTypes = torch.load(file_path).float()

    model = GraphUNetLessLayers(GNNdataDict["GNNData"]).to(device)
    # model = GraphUNetSmall(GNNdataDict["GNNData"]).to(device)
    output1 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
                    GNNdataDict["poolDataForward"])

    output1 = output1.cpu().detach().numpy()[0,:,0]
    output1 = np.where(output1==0,0.0001,output1)
    output1 = np.log(output1)
    # output1 = np.where(output1>0,1,0)

    print("fdsd",output1.shape)

    x_coords = dataDictMeshes[0]["cellCenters"][:, 0]
    y_coords = dataDictMeshes[0]["cellCenters"][:, 1]

    cellPoints = dataDictMeshes[0]["cellPoints"]
    x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
    y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]
    fig = plt.figure(figsize=(24, 13))

    ax = fig.add_subplot(1, 1,1)
    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, output1,
                        title=f'aggregation distance')
    plt.show()

    print(output1)

    # import time
    # t_tab = []
    # for i in range(1):
    #     print(i)
    #     x_tensor = torch.randn((batchSize, nNodes, inputChannels)).to(device)
    #     print("x_tensor: ", x_tensor.shape)
    #     t1 = time.time()
    #     output1 = model(x_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"],pointTypes.to(device))
    #     t_tab.append(time.time()-t1)
    #     print(output1.shape)
    #     del output1
    #     del x_tensor
    #     torch.cuda.empty_cache()
    #
    # plt.plot(t_tab)
    # plt.show()
    # print("output1.shape",output1.shape)


