import torch
import torch.nn as nn
import sys

sys.path.append("/home/justinbrusche/scripts_FVM_2/trainTestModel/Models/Pooling")
from GNN_parts.convolution_layers import *
from GNN_parts.pooling_layers import *
from GNN_parts.unpooling_layers import *

# class _ConvBlock1(nn.Module):
#
#     def __init__(self, in_channels, out_channels, GNNData):
#         super(_ConvBlock1, self).__init__()
#         self.layerPre = CustomGraphConv(in_channels, in_channels+1, GNNData["centerPoint"], GNNData["pointCenter"],pointType=True)
#         self.layer1 = CustomGraphConv(in_channels+1, out_channels, GNNData["centerPoint"], GNNData["pointCenter"])
#         self.layer2 = CustomGraphConv(out_channels, out_channels, GNNData["centerPoint"], GNNData["pointCenter"])
#
#     def forward(self, x,pointTypes, graphData, GNNDataForward):
#         x = self.layerPre(x, graphData["centerPoint"].edge_index, graphData["centerPoint"].edge_attr,GNNDataForward["centerPoint"],
#                             graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"],pointTypes=pointTypes)
#         x = self.layer1(x, graphData["centerPoint"].edge_index, graphData["centerPoint"].edge_attr,GNNDataForward["centerPoint"],
#                             graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"])
#         x = self.layer2(x, graphData["centerPoint"].edge_index, graphData["centerPoint"].edge_attr,GNNDataForward["centerPoint"],
#                             graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"])
#         return x

class _ConvBlock1(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock1, self).__init__()

    def forward(self, x,pointTypes, graphData, GNNDataForward):
        return x

class _ConvBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock2, self).__init__()
        self.pooling = CustomPoolingAdd()

    def forward(self, x, graphData,graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return x

class _ConvBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock3, self).__init__()
        self.pooling = CustomPoolingAdd()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return x

class _ConvBlock4(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock4, self).__init__()
        self.pooling = CustomPoolingAdd()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return x

class _ConvBlock5(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock5, self).__init__()
        self.pooling = CustomPoolingAdd()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.pooling(x, graphPoolData["pooling"].edge_index,graphPoolData["pooling"].edge_attr,poolDataForward["pooling"])

        return x

class _ConvBlock5_5(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock5_5, self).__init__()
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        # x = self.unpooling(x, graphPoolData["unpooling"].edge_index,graphPoolData["unpooling"].edge_attr,poolDataForward["unpooling"])
        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)

        return x

class _ConvBlock6(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock6, self).__init__()

        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):

        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock7(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock7, self).__init__()

        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData, graphPoolData,poolDataForward):
        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock8(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock8, self).__init__()
        self.unpooling = CustomUnpooling()

    def forward(self, x, graphData,graphPoolData,poolDataForward):

        x = self.unpooling(x, graphPoolData.edge_index,graphPoolData.edge_attr,poolDataForward)
        return x

class _ConvBlock9(nn.Module):

    def __init__(self, in_channels, out_channels, GNNData):
        super(_ConvBlock9, self).__init__()
        self.layer1 = CustomGraphConv(in_channels, out_channels, GNNData["pointPoint"])
        self.layer2 = CustomGraphConvEnd(out_channels, out_channels, GNNData["pointCenter"])

    def forward(self, x, graphData, GNNDataForward):
        x = self.layer1(x, graphData["pointPoint"].edge_index, graphData["pointPoint"].edge_attr)
        x = self.layer2(x,graphData["pointCenter"].edge_index, graphData["pointCenter"].edge_attr,GNNDataForward["pointCenter"])
        return x

class _ConvBlock10(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        self.layer1 = SelfLoopLayer( in_channels, out_channels)

    def forward(self, x):
        x = self.layer1(x)

        return x

class GraphUNetSimpleInstanceNorm(nn.Module):
    def __init__(self, GNNData):
        print("Pooling")
        super().__init__()
        self.convN_1 = _ConvBlock1(1, 16, GNNData)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.convN_4 = _ConvBlock4(16, 32, GNNData)
        self.convN_5 = _ConvBlock5(32, 32, GNNData)
        self.convN_5_5 = _ConvBlock5_5(32, 32, GNNData)
        self.convN_6 = _ConvBlock6(32, 32, GNNData)
        self.convN_7 = _ConvBlock7(32, 32, GNNData)
        self.convN_8 = _ConvBlock8(32, 32, GNNData)





    def forward(self, x, graphData, graphPoolData, GNNDataForward, poolDataForward, pointTypes):

        # convN_1out = self.convN_1(x, pointTypes, graphData[0], GNNDataForward[0])
        # x1 = self.convN_8(convN_1out, graphData[0], GNNDataForward[0])

        convN_2out = self.convN_2(x, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        x2 = self.convN_8(convN_2out, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        x3 = self.convN_7(convN_3out, graphData[1], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        x3 = self.convN_8(x3, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])
        x4 = self.convN_6(convN_4out, graphData[2], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        x4 = self.convN_7(x4, graphData[1], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        x4 = self.convN_8(x4, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])


        convN_5out = self.convN_5(convN_4out, graphData[4], graphPoolData[3], poolDataForward[3])
        x5 = self.convN_5_5(convN_5out, graphData[3], graphPoolData[3]["unpooling"], poolDataForward[3]["unpooling"])
        x5 = self.convN_6(x5, graphData[2], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        x5 = self.convN_7(x5, graphData[1], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        x5 = self.convN_8(x5, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])
        # except:
        #     x5 = torch.zeros_like(x4).to(device)

        return x, x2, x3 ,x4 , x5

class GraphUNetSimpleInstanceNorm4layers(nn.Module):
    def __init__(self, GNNData):
        print("Pooling")
        super().__init__()
        self.convN_1 = _ConvBlock1(1, 16, GNNData)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.convN_4 = _ConvBlock4(16, 32, GNNData)
        self.convN_5 = _ConvBlock5(32, 32, GNNData)
        self.convN_5_5 = _ConvBlock5_5(32, 32, GNNData)
        self.convN_6 = _ConvBlock6(32, 32, GNNData)
        self.convN_7 = _ConvBlock7(32, 32, GNNData)
        self.convN_8 = _ConvBlock8(32, 32, GNNData)





    def forward(self, x, graphData, graphPoolData, GNNDataForward, poolDataForward, pointTypes):

        # convN_1out = self.convN_1(x, pointTypes, graphData[0], GNNDataForward[0])
        # x1 = self.convN_8(convN_1out, graphData[0], GNNDataForward[0])
        convN_2out = self.convN_2(x, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        x2 = self.convN_8(convN_2out, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])

        convN_3out = self.convN_3(convN_2out, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        x3 = self.convN_7(convN_3out, graphData[1], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        x3 = self.convN_8(x3, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])

        convN_4out = self.convN_4(convN_3out, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])
        x4 = self.convN_6(convN_4out, graphData[2], graphPoolData[2]["unpooling"], poolDataForward[2]["unpooling"])
        x4 = self.convN_7(x4, graphData[1], graphPoolData[1]["unpooling"], poolDataForward[1]["unpooling"])
        x4 = self.convN_8(x4, graphData[0], graphPoolData[0]["unpooling"], poolDataForward[0]["unpooling"])

        # except:
        #     x5 = torch.zeros_like(x4).to(device)

        return x, x2, x3 ,x4

class GraphUNetSimpleInstanceNorm4layers_raw(nn.Module):
    def __init__(self, GNNData):
        print("Pooling")
        super().__init__()
        self.convN_1 = _ConvBlock1(1, 16, GNNData)
        self.convN_2 = _ConvBlock2(16, 16, GNNData)
        self.convN_3 = _ConvBlock3(16, 16, GNNData)
        self.convN_4 = _ConvBlock4(16, 32, GNNData)
        self.convN_5 = _ConvBlock5(32, 32, GNNData)
        self.convN_5_5 = _ConvBlock5_5(32, 32, GNNData)
        self.convN_6 = _ConvBlock6(32, 32, GNNData)
        self.convN_7 = _ConvBlock7(32, 32, GNNData)
        self.convN_8 = _ConvBlock8(32, 32, GNNData)





    def forward(self, x, graphData, graphPoolData, GNNDataForward, poolDataForward, pointTypes):

        # convN_1out = self.convN_1(x, pointTypes, graphData[0], GNNDataForward[0])
        # x1 = self.convN_8(convN_1out, graphData[0], GNNDataForward[0])
        x2 = self.convN_2(x, graphData[1], graphPoolData[0]["pooling"], poolDataForward[0]["pooling"])
        x3 = self.convN_2(x, graphData[2], graphPoolData[1]["pooling"], poolDataForward[1]["pooling"])
        x4 = self.convN_2(x, graphData[3], graphPoolData[2]["pooling"], poolDataForward[2]["pooling"])


        # except:
        #     x5 = torch.zeros_like(x4).to(device)

        return x, x2, x3 ,x4


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


    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    edgeAttrName = "MeshGraphNetCNN"
    # edgeAttrName = "test"

    file_path = f"{test_case_path}/embedding_{edgeAttrName}"
    with open(file_path, 'rb') as file:
        GNNdataDict = pickle.load(file)

    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    nNodes = dataDictMeshes[0]["nNodesDualMesh"]
    batchSize = 3
    inputChannels = 2
    device = "cuda"
    x_tensor = torch.randn((batchSize, nNodes,inputChannels)).to(device)
    print("x_tensor.shape",x_tensor.shape)

    file_path = f"{test_case_path}/pointTypes.pt"
    pointTypes = torch.load(file_path).float()


    model = GraphUNetSimpleInstanceNorm(GNNdataDict["GNNData"]).to(device)
    import time
    t_tab = []
    for i in range(1):
        print(i)
        x_tensor = torch.randn((batchSize, nNodes, inputChannels)).to(device)
        print("x_tensor: ", x_tensor.shape)
        t1 = time.time()
        output1 = model(x_tensor, GNNdataDict["graphData"], GNNdataDict["graphPoolData"], GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"],pointTypes.to(device))
        t_tab.append(time.time()-t1)
        print(output1.shape)
        del output1
        del x_tensor
        torch.cuda.empty_cache()

    plt.plot(t_tab)
    # plt.show()
    # print("output1.shape",output1.shape)


