import torch
from torch_geometric.data import Data
import numpy as np
import pickle

class EmbedGraph:

    def __init__(self,test_case_path,model,device, model_group,show_pooling = False,faceToCell=False,faceToPoint=False,faceToEdgeCenters=False,direct_pooling=False):
        self.test_case_path = test_case_path
        self.model = model
        self.device = device
        self.faceToCell = faceToCell
        self.faceToPoint = faceToPoint
        self.faceToEdgeCenters = faceToEdgeCenters
        self.model_group = model_group
        self.show_pooling = show_pooling
        self.direct_pooling = direct_pooling

        file_path = f"{test_case_path}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            self.dataDictMeshes = pickle.load(file)

        if self.model_group == "pointPoint" or self.model_group == "FVM":
            file_path = f"{test_case_path}/dataDictPoolMeshPoints.pkl"
            with open(file_path, 'rb') as file:
                self.dataDictPool = pickle.load(file)
        else:
            file_path = f"{test_case_path}/dataDictPoolCellCenters.pkl"
            with open(file_path, 'rb') as file:
                self.dataDictPool = pickle.load(file)

        if self.show_pooling:
            if self.direct_pooling:
                file_path = f"{test_case_path}/dataDictPoolCellCenters_direct.pkl"
                with open(file_path, 'rb') as file:
                    self.dataDictPool = pickle.load(file)
                    # print("direct")
            else:
                file_path = f"{test_case_path}/dataDictPoolCellCenters.pkl"
                with open(file_path, 'rb') as file:
                    self.dataDictPool = pickle.load(file)
                    # print("direct")

        file_path = f"{test_case_path}/dataDictEdgeAttr_{model}.pkl"
        with open(file_path, 'rb') as file:
            self.dataDictEdgeAttr = pickle.load(file)

        # print(self.dataDictEdgeAttr[4])
        # a

        # print(self.dataDictEdgeAttr[4]['centerPoint'])
        # b = self.dataDictEdgeAttr[4]['centerPoint']
        # print(np.sum(b,axis=1))
        #
        # a

        self.nLayers = len(self.dataDictMeshes)

    def embedMeshes(self):
        self.graphData = {}

        for i in range(self.nLayers):
            self.graphData[i] = {}
            #centerPoint

            if self.model_group == "FVM":
                if i == 0:
                    # edge_index = torch.stack(
                    #     (torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["sources"]).long(),
                    #      torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["targets"]).long()), dim=0).to(
                    #     self.device)
                    # edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["centerPoint"]).float().to(self.device)
                    # self.graphData[i]["centerPoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                    edge_index = torch.stack((torch.from_numpy(self.dataDictMeshes[i]["faceCenters"]["targets"]).long(),
                                              torch.from_numpy(self.dataDictMeshes[i]["faceCenters"]["sources"]).long()),dim=0).to(self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["centerFace"]).float().to(self.device)
                    self.graphData[i]["centerFace"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                    edge_index = torch.stack((torch.from_numpy(self.dataDictMeshes[i]["facePoints"]["sources"]).long(),
                                              torch.from_numpy(self.dataDictMeshes[i]["facePoints"]["targets"]).long()),dim=0).to(self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["facePoint"]).float().to(self.device)
                    self.graphData[i]["facePoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                    edge_index = torch.stack((torch.from_numpy(self.dataDictMeshes[i]["pointPoints"]["sources"]).long(),
                                torch.from_numpy(self.dataDictMeshes[i]["pointPoints"]["targets"]).long()),dim=0).to(self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["pointPoint"]).float().to(self.device)
                    self.graphData[i]["pointPoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                    edge_index = torch.stack(
                        (torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["targets"]).long(),
                         torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["sources"]).long()), dim=0).to(self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["pointCenter"]).float().to(self.device)
                    self.graphData[i]["pointCenter"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                else:
                    edge_index = torch.stack((torch.from_numpy(self.dataDictMeshes[i]["pointPoints"]["sources"]).long(),
                                torch.from_numpy(self.dataDictMeshes[i]["pointPoints"]["targets"]).long()),dim=0).to(self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["pointPoint"]).float().to(self.device)
                    self.graphData[i]["pointPoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

            else:
                if self.model_group == "pointPoint" and i != 0:
                    # print(i)
                    pass
                else:
                    edge_index = torch.stack(
                        (torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["sources"]).long(),
                         torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["targets"]).long()), dim=0).to(
                        self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["centerPoint"]).float().to(self.device)
                    self.graphData[i]["centerPoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

                    edge_index = torch.stack(
                        (torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["targets"]).long(),
                         torch.from_numpy(self.dataDictMeshes[i]["centerPoints"]["sources"]).long()), dim=0).to(
                        self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["pointCenter"]).float().to(self.device)
                    self.graphData[i]["pointCenter"] = Data(edge_index=edge_index, edge_attr=edge_attr)
                    # print(i)

                if self.model_group == "pointPoint":
                    edge_index = torch.stack((torch.from_numpy(self.dataDictMeshes[i]["pointPoints"]["targets"]).long(),
                                              torch.from_numpy(
                                                  self.dataDictMeshes[i]["pointPoints"]["sources"]).long()), dim=0).to(
                        self.device)
                    edge_attr = torch.from_numpy(self.dataDictEdgeAttr[i]["pointPoint"]).float().to(self.device)
                    self.graphData[i]["pointPoint"] = Data(edge_index=edge_index, edge_attr=edge_attr)

        return self.graphData

    def embedPooling(self):
        self.graphPoolData = {}
        self.poolDataFoward = {}
        for i in range(self.nLayers-1):
            self.graphPoolData[i] = {}
            edge_index = torch.stack((torch.from_numpy(self.dataDictPool[i]["pooling"]["sources"]).long(),
                        torch.from_numpy(self.dataDictPool[i]["pooling"]["targets"]).long()),dim=0).to(self.device)
            edge_attr = torch.from_numpy(self.dataDictPool[i]["pooling"]["attr"]).float().to(self.device)

            self.graphPoolData[i]["pooling"] = Data(edge_index=edge_index, edge_attr=edge_attr)

            edge_index = torch.stack((torch.from_numpy(self.dataDictPool[i]["unpooling"]["sources"]).long(),
                        torch.from_numpy(self.dataDictPool[i]["unpooling"]["targets"]).long()), dim=0).to(self.device)
            edge_attr = torch.from_numpy(self.dataDictPool[i]["unpooling"]["attr"]).float().to(self.device)

            self.graphPoolData[i]["unpooling"] = Data(edge_index=edge_index, edge_attr=edge_attr)
            self.poolDataFoward[i] = {}

            if self.model_group == "pointPoint" or self.model_group == "FVM":
                self.poolDataFoward[i]["pooling"] = {"inputSize":self.dataDictMeshes[i]["nNodesPrimalMesh"],
                                                      "outputSize":self.dataDictMeshes[i+1]["nNodesPrimalMesh"]}
                self.poolDataFoward[i]["unpooling"] = {"inputSize": self.dataDictMeshes[i + 1]["nNodesPrimalMesh"],
                                                           "outputSize": self.dataDictMeshes[i]["nNodesPrimalMesh"]}
            else:
                self.poolDataFoward[i]["pooling"] = {"inputSize":self.dataDictMeshes[i]["nNodesDualMesh"],
                                                      "outputSize":self.dataDictMeshes[i+1]["nNodesDualMesh"]}
                self.poolDataFoward[i]["unpooling"] = {"inputSize": self.dataDictMeshes[i + 1]["nNodesDualMesh"],
                                                           "outputSize": self.dataDictMeshes[i]["nNodesDualMesh"]}

            if self.show_pooling:
                if self.direct_pooling:
                    self.poolDataFoward[i]["pooling"] = {"inputSize":self.dataDictMeshes[0]["nNodesDualMesh"],
                                                          "outputSize":self.dataDictMeshes[i+1]["nNodesDualMesh"]}
                    self.poolDataFoward[i]["unpooling"] = {"inputSize": self.dataDictMeshes[i + 1]["nNodesDualMesh"],
                                                               "outputSize": self.dataDictMeshes[0]["nNodesDualMesh"]}
                else:
                    self.poolDataFoward[i]["pooling"] = {"inputSize": self.dataDictMeshes[i]["nNodesDualMesh"],
                                                         "outputSize": self.dataDictMeshes[i + 1]["nNodesDualMesh"]}
                    self.poolDataFoward[i]["unpooling"] = {"inputSize": self.dataDictMeshes[i + 1]["nNodesDualMesh"],
                                                           "outputSize": self.dataDictMeshes[i]["nNodesDualMesh"]}


        return self.graphPoolData

    def embedGNNDataForward(self):
        self.GNNDataForward = {}
        for i in range(self.nLayers):
            nNodesPrimalMesh = self.dataDictMeshes[i]["nNodesPrimalMesh"]
            nNodesDualMesh = self.dataDictMeshes[i]["nNodesDualMesh"]
            Nfaces = self.dataDictMeshes[i]["Nfaces"].item()
            self.GNNDataForward[i] = {}
            if i == 0:
                self.GNNDataForward[i]["centerFace"] = {"inputSize":nNodesDualMesh,
                                                      "outputSize":Nfaces}
                self.GNNDataForward[i]["facePoint"] = {"inputSize":Nfaces,
                                                      "outputSize":nNodesPrimalMesh}
                self.GNNDataForward[i]["pointCenter"] = {"inputSize":nNodesPrimalMesh,
                                                      "outputSize":nNodesDualMesh}
                self.GNNDataForward[i]["centerPoint"] = {"inputSize":nNodesDualMesh,
                                                      "outputSize":nNodesPrimalMesh}

            else:
                pass

    def embedGNNData(self):
        if self.model_group == "FVM":
            self.GNNData = {}
            # print(self.dataDictEdgeAttr["nAttributesDict"])
            self.GNNData["centerFace"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["centerFace"]}
            self.GNNData["centerPoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["centerPoint"]}
            self.GNNData["facePoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["facePoint"]}
            self.GNNData["pointPoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["pointPoint"]}
            self.GNNData["pointCenter"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["pointCenter"]}

            print(self.GNNData)

        else:
            self.GNNData ={}
            self.GNNData["centerPoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["centerPoint"]}
            self.GNNData["pointCenter"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["pointCenter"]}
            if self.model_group == "pointPoint":
                self.GNNData["pointPoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["pointPoint"]}
                # print(self.GNNData["pointPoint"])

            self.GNNData["facePoint"] = {"nEdgeAttr": self.dataDictEdgeAttr["nAttributesDict"]["facePoint"]}


    def getData(self):
        self.embedMeshes()
        self.embedPooling()
        self.embedGNNDataForward()
        self.embedGNNData()
        dataDict = {"poolDataForward":self.poolDataFoward,
                    "graphPoolData":self.graphPoolData,
                    "graphData": self.graphData,
                    "GNNDataForward":self.GNNDataForward,
                    "GNNData":self.GNNData}

        print(dataDict["GNNData"])

        if self.show_pooling:
            # print("POOLNG")
            if self.direct_pooling:
                file_path = f"{self.test_case_path}/embedding_show_pooling_direct"
                with open(file_path, 'wb') as file:
                    pickle.dump(dataDict, file)
            else:
                file_path = f"{self.test_case_path}/embedding_show_pooling"
                with open(file_path, 'wb') as file:
                    pickle.dump(dataDict, file)

        else:
            file_path = f"{self.test_case_path}/embedding_{self.model}"
            with open(file_path, 'wb') as file:
                pickle.dump(dataDict, file)


        return dataDict


if __name__ == '__main__':
    import pickle

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"

    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p4"
    #
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    test_case_path = "/home/justinbrusche/test_cases/cylinder"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"
    #

    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"

    model = "slope"
    model = "test4"
    model = "normConnectedNodes"
    # model = "type"
    # model = "test"
    model = "MeshGraphNetCNN"
    model = "MeshGraphNetCNNWithFace"
    model = "FVMModel"
    # model = "FVMModel2"
    model = "FVMModelConv"




    model_group = "pointPoint"
    model_group = "FVM"


    device = 'cuda'
    # device = 'cpu'

    a = EmbedGraph(test_case_path,model,device,model_group,show_pooling=True,direct_pooling=True)
    # a = EmbedGraph(test_case_path,model,device,model_group,show_pooling=False)

    # data_dict = a.embedMeshes()
    # data_dictPool = a.embedPooling()
    dataDict = a.getData()
    # dataDict = a.getData(show_pooling=True)

    # print(data_dict[4]["pointCenter"].edge_index)