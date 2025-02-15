import torch
import torch.nn as nn
import sys

sys.path.append("/home/justinbrusche/scripts_final/trainTestModel/ModelsSpeed/perlin")
from GNN_parts.convolution_layers import *
from GNN_parts.pooling_layers import *
from GNN_parts.unpooling_layers import *
sys.path.append("/home/justinbrusche/scripts_final/GNN_pre_scripts")


class _ConvBlock1(nn.Module):

    def __init__(self, in_channels_centers,in_channels_faces, out_channels_centerFace, out_channels, GNNData):
        super(_ConvBlock1, self).__init__()
        self.layerCenterFace = CustomGraphConv(in_channels_centers, out_channels_centerFace,GNNData["centerFace"])
        self.layerFacePoint = CustomGraphConv(out_channels_centerFace+in_channels_faces, out_channels, GNNData["facePoint"])

        self.layerPointPoint = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, xCellCenters, xFace, graphData, GNNDataForward,weights_dict):
        xCellCenters = self.layerCenterFace(xCellCenters, graphData["centerFace"].edge_index,
                                            graphData["centerFace"].edge_attr, GNNDataForward["centerFace"],weights_dict['layerCenterFace'])

        x = torch.cat([xCellCenters, xFace], dim=-1)

        x = self.layerFacePoint(x, graphData["facePoint"].edge_index, graphData["facePoint"].edge_attr,
                                     GNNDataForward["facePoint"],weights_dict['layerFacePoint'])

        # x = self.layerPointPoint(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict['layerPointPoint'])

        return self.layerPointPoint(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict['layerPointPoint'])



class _ConvBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock2, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData, poolDataForward,weights_dict):


        x = self.pooling(x, graphPoolData.edge_index, graphPoolData.edge_attr, poolDataForward)

        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])

        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])

        return self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])


class _ConvBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock3, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData,poolDataForward,weights_dict):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])
        # x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])

        return self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])


class _ConvBlock4LessLayers(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock4LessLayers, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.layer3 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.layer4 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward,weights_dict):
        x = self.pooling(x, graphPoolData["pooling"].edge_index,graphPoolData["pooling"].edge_attr,poolDataForward["pooling"])
        # print("sum",torch.sum(x).item())
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])
        x = self.layer3(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer3"])
        x = self.layer4(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer4"])
        # print("sum",torch.sum(x).item())

        # x = self.unpooling(x, graphPoolData["unpooling"].edge_index,graphPoolData["unpooling"].edge_attr,poolDataForward["unpooling"])

        return self.unpooling(x, graphPoolData["unpooling"].edge_index,graphPoolData["unpooling"].edge_attr,poolDataForward["unpooling"])


class _ConvBlock7(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock7, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward,weights_dict):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])
        # x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)


class _ConvBlock8(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock8, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData,graphPoolData,poolDataForward,weights_dict):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer2"])
        # x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)


class _ConvBlock9(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock9, self).__init__()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConv(out_channels, out_channels, GNNData["pointCenter"])

    def forward(self, x, graphData, GNNDataForward,weights_dict):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr,weights_dict["layer1"])
        # x = self.layer2(x,graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"],weights_dict["layer2"])

        return self.layer2(x,graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"],weights_dict["layer2"])


class _ConvBlock10(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        self.layer1 = SelfLoopLayer( in_channels, out_channels)

    def forward(self, x):
        # x = self.layer1(x)

        return self.layer1(x)



class GraphUNetFullScaleLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_FullScale_LessLayers_no_last_norm")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,28,32, GNNData)
        self.in1 = nn.InstanceNorm1d(32)
        self.convN_2 = _ConvBlock2(32, 32, GNNData)
        self.in2 = nn.InstanceNorm1d(32)
        self.convN_3 = _ConvBlock3(32, 32, GNNData)
        self.in3 = nn.InstanceNorm1d(32)
        self.convN_4 = _ConvBlock4LessLayers(32, 64, GNNData)
        self.in4 = nn.InstanceNorm1d(64)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(96, 64, GNNData)
        self.in7 = nn.InstanceNorm1d(64)
        self.convN_8 = _ConvBlock8(96, 32, GNNData)
        self.in8 = nn.InstanceNorm1d(32)
        self.convN_9 = _ConvBlock9(64, 32, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
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

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        # convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        # convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out


class GraphUNetMediumLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_Medium_LessLayers_no_last_norm")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,20,24, GNNData)
        self.in1 = nn.InstanceNorm1d(24)
        self.convN_2 = _ConvBlock2(24, 24, GNNData)
        self.in2 = nn.InstanceNorm1d(24)
        self.convN_3 = _ConvBlock3(24, 24, GNNData)
        self.in3 = nn.InstanceNorm1d(24)
        self.convN_4 = _ConvBlock4LessLayers(24, 48, GNNData)
        self.in4 = nn.InstanceNorm1d(48)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(72, 48, GNNData)
        self.in7 = nn.InstanceNorm1d(48)
        self.convN_8 = _ConvBlock8(72, 24, GNNData)
        self.in8 = nn.InstanceNorm1d(24)
        self.convN_9 = _ConvBlock9(48, 24, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
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

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        # convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        # convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out

class GraphUNetSmallLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_Small_LessLayers_no_last_norm")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,12,16, GNNData)
        self.in1 = nn.InstanceNorm1d(16)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.in2 = nn.InstanceNorm1d(16)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.in3 = nn.InstanceNorm1d(16)
        self.convN_4 = _ConvBlock4LessLayers(16, 32, GNNData)
        self.in4 = nn.InstanceNorm1d(32)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(48, 32, GNNData)
        self.in7 = nn.InstanceNorm1d(32)
        self.convN_8 = _ConvBlock8(48, 16, GNNData)
        self.in8 = nn.InstanceNorm1d(16)
        self.convN_9 = _ConvBlock9(32, 16, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
        self.final = _ConvBlock10(16, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward,weights_dict):
        # print(xCellCenters.is_contiguous())
        # print(xFace.is_contiguous())


        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0],weights_dict['convN_1'])
        # print(convN_1out.is_contiguous())
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)
        # print(convN_1out.is_contiguous())
        # convN_1out = convN_1out.squeeze(-1)
        # print(convN_1out.is_contiguous())


        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"],weights_dict['convN_2'])
        # print(convN_2out.is_contiguous())

        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"],weights_dict['convN_3'])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2],weights_dict['convN_4'])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        # convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        # convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"],weights_dict['convN_7'])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"],weights_dict['convN_8'])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0],weights_dict['convN_9'])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out


class GraphUNetVerySmallLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_VerySmall_LessLayers_no_last_norm")
        # print(GNNData)

        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,8,12, GNNData)
        self.in1 = nn.InstanceNorm1d(12)
        self.convN_2 = _ConvBlock2(12, 12, GNNData)
        self.in2 = nn.InstanceNorm1d(12)
        self.convN_3 = _ConvBlock3(12, 12, GNNData)
        self.in3 = nn.InstanceNorm1d(12)
        self.convN_4 = _ConvBlock4LessLayers(12, 24, GNNData)
        self.in4 = nn.InstanceNorm1d(24)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(36, 24, GNNData)
        self.in7 = nn.InstanceNorm1d(24)
        self.convN_8 = _ConvBlock8(36, 12, GNNData)
        self.in8 = nn.InstanceNorm1d(12)
        self.convN_9 = _ConvBlock9(24, 12, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
        self.final = _ConvBlock10(12, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward,weights_dict):
        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0],weights_dict['convN_1'])
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"],weights_dict['convN_2'])
        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"],weights_dict['convN_3'])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2],weights_dict['convN_4'])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        # convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        # convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"],weights_dict['convN_7'])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"],weights_dict['convN_8'])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0],weights_dict['convN_9'])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out


class GraphUNetVeryVerySmallLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_VeryVerySmall_LessLayers_no_last_norm")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,4,8, GNNData)
        self.in1 = nn.InstanceNorm1d(8)
        self.convN_2 = _ConvBlock2(8, 8, GNNData)
        self.in2 = nn.InstanceNorm1d(8)
        self.convN_3 = _ConvBlock3(8, 8, GNNData)
        self.in3 = nn.InstanceNorm1d(8)
        self.convN_4 = _ConvBlock4LessLayers(8, 16, GNNData)
        self.in4 = nn.InstanceNorm1d(16)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(24, 16, GNNData)
        self.in7 = nn.InstanceNorm1d(16)
        self.convN_8 = _ConvBlock8(24, 8, GNNData)
        self.in8 = nn.InstanceNorm1d(8)
        self.convN_9 = _ConvBlock9(16, 8, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
        self.final = _ConvBlock10(8, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward,weights_dict):
        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0],weights_dict['convN_1'])
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"],weights_dict['convN_2'])
        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"],weights_dict['convN_3'])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2],weights_dict['convN_4'])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        # convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        # convN_5out = self.in5(convN_5out.transpose(1, 2)).transpose(1, 2)

        # convN_6out = self.convN_6(torch.cat((convN_5out, convN_4out), dim=2), graphData[3], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        # convN_6out = self.in6(convN_6out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"],weights_dict['convN_7'])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"],weights_dict['convN_8'])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0],weights_dict['convN_9'])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out

class GraphUNetVeryVeryVerySmallLessLayers(nn.Module):
    def __init__(self, GNNData):
        print("GraphUNet_BC1Gradient0_VeryVeryVerySmall_LessLayers_no_last_norm")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(4)

        self.convN_1 = _ConvBlock1(2, 4,2,6, GNNData)
        self.in1 = nn.InstanceNorm1d(6)
        self.convN_2 = _ConvBlock2(6, 6, GNNData)
        self.in2 = nn.InstanceNorm1d(6)
        self.convN_3 = _ConvBlock3(6, 6, GNNData)
        self.in3 = nn.InstanceNorm1d(6)
        self.convN_4 = _ConvBlock4LessLayers(6, 12, GNNData)
        self.in4 = nn.InstanceNorm1d(12)
        # self.convN_5 = _ConvBlock5(48, 48, GNNData)
        # self.in5 = nn.InstanceNorm1d(48)
        # self.convN_6 = _ConvBlock6(96, 48, GNNData)
        # self.in6 = nn.InstanceNorm1d(48)
        self.convN_7 = _ConvBlock7(18, 12, GNNData)
        self.in7 = nn.InstanceNorm1d(12)
        self.convN_8 = _ConvBlock8(18, 6, GNNData)
        self.in8 = nn.InstanceNorm1d(6)
        self.convN_9 = _ConvBlock9(12, 6, GNNData)
        # self.in9 = nn.InstanceNorm1d(24)
        self.final = _ConvBlock10(6, 1)

    def forward(self, xCellCenters,xFace, graphData, graphPoolData, GNNDataForward, poolDataForward,weights_dict):
        xCellCenters = self.inInputCellCenters(xCellCenters.transpose(1, 2)).transpose(1, 2)
        xFace = self.inInputFace(xFace.transpose(1, 2)).transpose(1, 2)

        convN_1out = self.convN_1(xCellCenters,xFace, graphData[0], GNNDataForward[0],weights_dict['convN_1'])
        convN_1out = self.in1(convN_1out.transpose(1, 2)).transpose(1, 2)

        convN_2out = self.convN_2(convN_1out, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"],weights_dict['convN_2'])
        convN_2out = self.in2(convN_2out.transpose(1, 2)).transpose(1, 2)
        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"],weights_dict['convN_3'])
        convN_3out = self.in3(convN_3out.transpose(1, 2)).transpose(1, 2)

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2], poolDataForward[2],weights_dict['convN_4'])
        convN_4out = self.in4(convN_4out.transpose(1, 2)).transpose(1, 2)

        convN_7out = self.convN_7(torch.cat((convN_4out, convN_3out), dim=2), graphData[2], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"],weights_dict['convN_7'])
        convN_7out = self.in7(convN_7out.transpose(1, 2)).transpose(1, 2)

        convN_8out = self.convN_8(torch.cat((convN_7out, convN_2out), dim=2), graphData[1], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"],weights_dict['convN_8'])
        convN_8out = self.in8(convN_8out.transpose(1, 2)).transpose(1, 2)

        convN_9out = self.convN_9(torch.cat((convN_8out, convN_1out), dim=2), graphData[0], GNNDataForward[0],weights_dict['convN_9'])
        # convN_9out = self.in9(convN_9out.transpose(1, 2)).transpose(1, 2)

        final_out = self.final(convN_9out)

        return final_out

def print_tree(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_tree(value, indent + 1)
        else:
            print('  ' * (indent + 1) + str(value.shape))



def get_edgeAttrPrep(model,GNNdataDict):
    weights_dict = {}

    level_dict = {"convN_1":0,
                  "convN_2":1,
                  "convN_3":2,
                  "convN_4":3,
                  "convN_7":2,
                  "convN_8":1,
                  "convN_9":0}

    print(GNNdataDict["graphData"][0].keys())
    print(GNNdataDict["graphData"][0].values())
    print(GNNdataDict["GNNDataForward"][0]['facePoint'])

    aggrTypeLink = {"layerCenterFace": "centerFace",
                    "layerFacePoint": "facePoint"}

    # Populate the dictionary with model parameters (only weights)
    for name, param in model.named_parameters():
        # print(name)
        if 'weight_matrix' in name:
            keys = name.split('.')
            d = weights_dict
            for key in keys[:-3]:
                if key not in d:
                    d[key] = {}
                d = d[key]

            # print(keys)
            level = level_dict.get(keys[0], None)
            # print(level)

            if keys[2] == "pointPoint":
                aggrType = "pointPoint"
            elif keys[2] == "pointCenter":
                if keys[0] == 'convN_9':
                    aggrType = "pointCenter"
                else:
                    aggrType = aggrTypeLink[keys[1]]

            tensor = GNNdataDict["graphData"][level][aggrType].edge_attr
            d[keys[-3]] = torch.einsum('ij,jkl->ikl', tensor, param.data).contiguous()


            # d[keys[-3]] = torch.einsum('ij,jkl->ikl', tensor, param.data).transpose(1, 2).unsqueeze(0).expand(1, -1, -1,-1).contiguous()

    return weights_dict

if __name__ == '__main__':
    import pickle
    from anytree import Node, RenderTree
    import os
    import yaml
    import sys

    sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    config_file = '/home/justinbrusche/scripts_final/trainTestModel/config_files_step1/small.yaml'

    with open(config_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    test_case_path = "/home/justinbrusche/datasets/step1/case_0"

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
    print(nFaces)

    batchSize = 1
    inputChannels = 2
    device = "cuda"
    # device = "cpu"

    x_tensor = torch.randn((batchSize, nNodes,inputChannels)).to(device)
    face_tensor = torch.randn((batchSize, nFaces,3)).to(device)

    print("x_tensor.shape",x_tensor.shape)

    file_path = f"{test_case_path}/pointTypes.pt"
    pointTypes = torch.load(file_path).float()


    model = GraphUNetSmallLessLayers(GNNdataDict["GNNData"]).to(device)

    weights_dict = get_edgeAttrPrep(model, GNNdataDict)
    # # print_tree(weights_dict)
    # print("test1")
    # output1 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
    #                 GNNdataDict["poolDataForward"],weights_dict)
    # print("test2")
    #
    # output2 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
    #                 GNNdataDict["poolDataForward"],weights_dict)
    #
    import torch
    import time

    import torch.autograd.profiler as profiler

    output = model(x_tensor, face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"],
                   GNNdataDict["GNNDataForward"],
                   GNNdataDict["poolDataForward"], weights_dict)
    output = model(x_tensor, face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"],
                   GNNdataDict["GNNDataForward"],
                   GNNdataDict["poolDataForward"], weights_dict)

    # with profiler.profile(use_cuda=True) as prof:
    #     output1 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
    #                 GNNdataDict["poolDataForward"],weights_dict)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profiler.profile(use_cuda=True) as prof:
        output1 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
                    GNNdataDict["poolDataForward"],weights_dict)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # for _ in range(100):  # Run in a loop to see GPU utilization
    #     output = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
    #                 GNNdataDict["poolDataForward"],weights_dict)
    #     print(f"GPU Utilization: {torch.cuda.utilization()}")
    #     time.sleep(0.5)

    print("field fast")

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


