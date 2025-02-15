import torch
import torch.nn as nn
import sys

sys.path.append("/home/justinbrusche/scripts_final/trainTestModel/Models/magBase")
from GNN_parts.convolution_layers import *
from GNN_parts.pooling_layers import *
from GNN_parts.unpooling_layers import *


class _ConvBlock1(nn.Module):

    def __init__(self, in_channels_centers,in_channels_faces, out_channels_centerFace, out_channels, GNNData):
        super(_ConvBlock1, self).__init__()
        self.layerCenterFace = CustomGraphConv(in_channels_centers, out_channels_centerFace,GNNData["centerFace"])
        self.layerFacePoint = CustomGraphConv(out_channels_centerFace+in_channels_faces, out_channels, GNNData["facePoint"])

        self.layerPointPoint = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, xCellCenters, xFace, graphData, GNNDataForward):
        xCellCenters = self.layerCenterFace(xCellCenters, graphData["centerFace"].edge_index,
                                            graphData["centerFace"].edge_attr, GNNDataForward["centerFace"])

        x = torch.cat([xCellCenters, xFace], dim=-1)

        x = self.layerFacePoint(x, graphData["facePoint"].edge_index, graphData["facePoint"].edge_attr,
                                     GNNDataForward["facePoint"])

        x = self.layerPointPoint(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        return x


class _ConvBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock2, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData, poolDataForward):


        x = self.pooling(x, graphPoolData.edge_index, graphPoolData.edge_attr, poolDataForward)

        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        return x

class _ConvBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock3, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)

        return x

class _ConvBlock4LessLayers(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock4LessLayers, self).__init__()
        self.pooling = CustomPoolingAdd()
        self.layer1 = CustomGraphConvPointPoint(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.in4 = nn.InstanceNorm1d(out_channels)
        self.layer3 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.layer4 = CustomGraphConvPointPoint(out_channels, out_channels, GNNData["pointPoint"])
        self.in42 = nn.InstanceNorm1d(out_channels)

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData["pooling"].edge_index,graphPoolData["pooling"].edge_attr,poolDataForward["pooling"])
        # print("sum",torch.sum(x).item())
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.in4(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer3(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer4(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.in42(x.transpose(1, 2)).transpose(1, 2)

        # print("sum",torch.sum(x).item())
        return x

class _ConvBlock10(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        self.layer1 = SelfLoopLayer( in_channels, 3*in_channels)
        # self.layer2 = SelfLoopLayer( in_channels, in_channels)
        self.layer3 = SelfLoopLayer( 3*in_channels, out_channels)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)

        return x


class VeryVeryVerySmall(nn.Module):
    def __init__(self, GNNData):
        print("VeryVeryVerySMALL MAG")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(3)

        self.convN_1 = _ConvBlock1(2, 3,3,6, GNNData)
        self.in1 = nn.InstanceNorm1d(6)
        self.convN_2 = _ConvBlock2(6, 6, GNNData)
        self.in2 = nn.InstanceNorm1d(6)
        self.convN_3 = _ConvBlock3(6, 6, GNNData)
        self.in3 = nn.InstanceNorm1d(6)
        self.convN_4 = _ConvBlock4LessLayers(6, 12, GNNData)
        self.in4 = nn.InstanceNorm1d(12)
        self.final = _ConvBlock10(12, 1)

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

        # mean_convN_4out = torch.max(convN_4out, dim=1).values
        mean_convN_4out = torch.std(convN_4out,dim=1)

        final_out = self.final(mean_convN_4out)

        return final_out

class VeryVerySmall(nn.Module):
    def __init__(self, GNNData):
        print("VeryVerySMALL MAG")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(3)

        self.convN_1 = _ConvBlock1(2, 3,5,8, GNNData)
        self.in1 = nn.InstanceNorm1d(8)
        self.convN_2 = _ConvBlock2(8, 8, GNNData)
        self.in2 = nn.InstanceNorm1d(8)
        self.convN_3 = _ConvBlock3(8, 8, GNNData)
        self.in3 = nn.InstanceNorm1d(8)
        self.convN_4 = _ConvBlock4LessLayers(8, 16, GNNData)
        self.in4 = nn.InstanceNorm1d(16)
        self.final = _ConvBlock10(16, 1)

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

        mean_convN_4out = torch.std(convN_4out,dim=1)

        final_out = self.final(mean_convN_4out)

        return final_out


class VerySmall(nn.Module):
    def __init__(self, GNNData):
        print("VerySMALL MAG")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(3)

        self.convN_1 = _ConvBlock1(2, 3,9,12, GNNData)
        self.in1 = nn.InstanceNorm1d(12)
        self.convN_2 = _ConvBlock2(12, 12, GNNData)
        self.in2 = nn.InstanceNorm1d(12)
        self.convN_3 = _ConvBlock3(12, 12, GNNData)
        self.in3 = nn.InstanceNorm1d(12)
        self.convN_4 = _ConvBlock4LessLayers(12, 24, GNNData)
        self.in4 = nn.InstanceNorm1d(24)
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

        mean_convN_4out = torch.std(convN_4out,dim=1)

        final_out = self.final(mean_convN_4out)

        return final_out

class Small(nn.Module):
    def __init__(self, GNNData):
        print("SMALL MAG")
        super().__init__()

        self.inInputCellCenters = nn.InstanceNorm1d(2)
        self.inInputFace = nn.InstanceNorm1d(3)

        self.convN_1 = _ConvBlock1(2, 3,13,16, GNNData)
        self.in1 = nn.InstanceNorm1d(16)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.in2 = nn.InstanceNorm1d(16)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.in3 = nn.InstanceNorm1d(16)
        self.convN_4 = _ConvBlock4LessLayers(16, 32, GNNData)
        self.in4 = nn.InstanceNorm1d(32)
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

        mean_convN_4out = torch.std(convN_4out,dim=1)

        final_out = self.final(mean_convN_4out)

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


    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    # edgeAttrName = "MeshGraphNetCNNWithFace"
    edgeAttrName = "FVMModelConv"

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

    batchSize = 3
    inputChannels = 2
    device = "cuda"
    # device = "cpu"

    x_tensor = torch.randn((batchSize, nNodes,inputChannels)).to(device)
    face_tensor = torch.randn((batchSize, nFaces,4)).to(device)

    print("x_tensor.shape",x_tensor.shape)

    file_path = f"{test_case_path}/pointTypes.pt"
    pointTypes = torch.load(file_path).float()


    model = GraphUNetMediumLessLayers(GNNdataDict["GNNData"]).to(device)
    output1 = model(x_tensor,face_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"],
                    GNNdataDict["poolDataForward"])

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


