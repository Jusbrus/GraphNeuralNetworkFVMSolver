import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import sys
import pickle
class LossClass:
    def __init__(self, test_case_path,conf, device='cuda',t=0):
        self.device = device
        self.test_case_path = os.path.join(test_case_path, "level0")
        self.indices = self.get_indices()
        self.prepare_FVM_loss()

        file_path = f"{test_case_path}/embedding_show_pooling_direct"
        with open(file_path, 'rb') as file:
            self.GNNdataDict_pooling = pickle.load(file)

    def prepare_FVM_loss(self):
        sparce_matrix_data = np.loadtxt(os.path.join(self.test_case_path, "constant/polyMesh/faceCells.txt")).astype(int)
        self.num_nodes = np.max(sparce_matrix_data) + 1

        i_boundary_cells = np.any(sparce_matrix_data == -1, axis=1)

        filtered_data = sparce_matrix_data[~i_boundary_cells]
        self.FVM_face_indices_0 = filtered_data[:,0]
        self.FVM_face_indices_1 = filtered_data[:,1]

    def get_indices(self):
        sparce_matrix_data = np.loadtxt(os.path.join(self.test_case_path, "constant/polyMesh/faceCells.txt")).astype(int)
        self.num_nodes = np.max(sparce_matrix_data) + 1

        i_boundary_cells = np.any(sparce_matrix_data == -1, axis=1)

        filtered_data = sparce_matrix_data[~i_boundary_cells]

        x = np.concatenate((np.arange(self.num_nodes), filtered_data[:, 0], filtered_data[:, 1]))
        y = np.concatenate((np.arange(self.num_nodes), filtered_data[:, 1], filtered_data[:, 0]))

        return torch.LongTensor(np.array([y, x])).to(self.device)

    def getMatrixValues(self,d,face):
        # return torch.FloatTensor(np.concatenate((d, face, face))).to(self.device)
        return torch.concatenate((d, face, face))

    def setup_sparse_matrix(self,d,face):
        values = self.getMatrixValues(d,face)
        return torch.sparse_coo_tensor(self.indices, values.to(self.device), (self.num_nodes, self.num_nodes)).to(self.device)

    # def getPredictedSources(self, outP, dArray, faceArray):
    #     predictedSources = []
    #     for i in range(len(outP)):
    #         sparse_matrix = self.setup_sparse_matrix(dArray[i].view(-1), faceArray[i].view(-1))
    #         predicted_source = torch.matmul(sparse_matrix, outP[i])
    #         predictedSources.append(predicted_source)
    #
    #     predictedSources = torch.stack(predictedSources)
    #
    #     return predictedSources.to(self.device)
    #
    # def get_scale(self, outP, dArray, faceArray, sourceArray,p):
    #     sys.path.append("/home/justinbrusche/scripts_FVM_2/trainTestModel/Models")
    #     from Models.Pooling.MeshGraphNetCNN import GraphUNetSimpleInstanceNorm4layers_raw
    #
    #     model = GraphUNetSimpleInstanceNorm4layers_raw(self.GNNdataDict_pooling["GNNData"]).to(self.device)
    #
    #     for i in range(len(outP)):
    #         sparse_matrix = self.setup_sparse_matrix(dArray[i].view(-1).to(self.device), faceArray[i].view(-1).to(self.device))
    #         predicted_source = torch.matmul(sparse_matrix, outP[i])
    #
    #         s1, s2, s3, s4 = model(sourceArray.to(self.device), self.GNNdataDict_pooling["graphData"],
    #                                self.GNNdataDict_pooling["graphPoolData"],
    #                                self.GNNdataDict_pooling["GNNDataForward"], self.GNNdataDict_pooling["poolDataForward"],
    #                                None)
    #
    #         # print(s1.shape, s2.shape, s3.shape, s4.shape," fdsafdsaafdsafdsafdsaafdasafdsaafdasafdsaf")
    #
    #         sp1, sp2, sp3, sp4 = model(predicted_source.to(self.device), self.GNNdataDict_pooling["graphData"],
    #                                self.GNNdataDict_pooling["graphPoolData"],
    #                                self.GNNdataDict_pooling["GNNDataForward"], self.GNNdataDict_pooling["poolDataForward"],
    #                                None)
    #
    #         p1, p2, p3, p4 = model(p.unsqueeze(-1).to(self.device), self.GNNdataDict_pooling["graphData"],
    #                                self.GNNdataDict_pooling["graphPoolData"],
    #                                self.GNNdataDict_pooling["GNNDataForward"], self.GNNdataDict_pooling["poolDataForward"],
    #                                None)
    #
    #         # print(s1.shape, s2.shape, s3.shape, s4.shape," fdsafdsaafdsafdsafdsaafdasafdsaafdasafdsaf")
    #
    #         pp1, pp2, pp3, pp4 = model(outP.to(self.device), self.GNNdataDict_pooling["graphData"],
    #                                self.GNNdataDict_pooling["graphPoolData"],
    #                                self.GNNdataDict_pooling["GNNDataForward"], self.GNNdataDict_pooling["poolDataForward"],
    #                                None)
    #
    #         # print(sp1.shape, sp2.shape, sp3.shape, sp4.shape," fdsafdsaafdsafdsafdsaafdasafdsaafdasafdsaf")
    #
    #
    #         print("dfsaafdsaafdsaa")
    #         # print(torch.std(s1)/torch.std(sp1))
    #         # print(torch.std(s2)/torch.std(sp2))
    #         # print(torch.std(s3)/torch.std(sp3))
    #         # print(torch.std(s4)/torch.std(sp4))
    #         # print(s1.shape,sp1.shape,sp2.shape,sp3.shape,sp4.shape)
    #
    #         print(torch.dot(sp1.squeeze(-1), s1[0].squeeze(-1)) / torch.dot(sp1.squeeze(-1), sp1.squeeze(-1)))
    #         print(torch.dot(sp2.squeeze(-1), s2[0].squeeze(-1)) / torch.dot(sp2.squeeze(-1), sp2.squeeze(-1)))
    #         print(torch.dot(sp3.squeeze(-1), s3[0].squeeze(-1)) / torch.dot(sp3.squeeze(-1), sp3.squeeze(-1)))
    #         print(torch.dot(sp4.squeeze(-1), s4[0].squeeze(-1)) / torch.dot(sp4.squeeze(-1), sp4.squeeze(-1)))
    #
    #     return sp1,sp2,sp3,sp4,s1.squeeze(-1),s2.squeeze(-1),s3.squeeze(-1),s4.squeeze(-1),p4,pp4
    #
    def computeLossSource(self, outP, pArray, dArray, faceArray,sourceArray):
        loss = 0

        # outP = outP * pressure_factor.to(self.device)
        # pArray = pArray * pressure_factor.to(self.device)

        # print(faceArray.shape)
        # source_mag_list = torch.mean(abs(sourceArray), axis=(1,2))
        # source_mag_list = torch.std(sourceArray, axis=(1,2))
        # print(outP.shape)
        # print(torch.std(outP))
        # sourceArray = sourceArray / source_mag_list.unsqueeze(-1).unsqueeze(-1) / pressure_factor.to(self.device)
        # sourceArray = sourceArray / source_mag_list.unsqueeze(-1).unsqueeze(-1) / pressure_factor.to(self.device)

        for i in range(len(outP)):
            sparse_matrix = self.setup_sparse_matrix(dArray[i].view(-1), faceArray[i].view(-1))
            # predicted_source = torch.matmul(sparse_matrix, outP[i].to(self.device))

            gt_source_expl = torch.matmul(sparse_matrix.to("cpu"), (pArray[i].squeeze(-1)))

            diff_gt = gt_source_expl - sourceArray[0,:,0]
            print(diff_gt)
            # diff_gt = gt_source_expl

            plt.figure()
            plt.plot(diff_gt)
            plt.title("diff")
            plt.figure()
            plt.plot(sourceArray[0])
            plt.title("source")
            plt.show()

            # diff_source = torch.matmul(sparse_matrix, pArray[i]) - torch.matmul(sparse_matrix, outP[i])
            #
            # if countCellweights:
            #     loss += torch.sum(cellWeights.unsqueeze(1) * ((diff_source) ** 2)) / outP.numel()
            # else:
            #     loss += torch.sum(((diff_source) ** 2)) / outP.numel()

        # predictedSources = self.getPredictedSources(outP, dArray, faceArray)
        # # print(torch.sum((sourceArray - predictedSources) ** 2))

        return loss
    #
    # def checkcomputeLossSource(self, outP, dArray, faceArray, sourceArray,cellWeights,countCellweights,pressure_factor):
    #     loss = 0
    #
    #     outP = outP * pressure_factor.to(self.device)
    #
    #     # print(faceArray.shape)
    #     # source_mag_list = torch.mean(abs(sourceArray), axis=(1,2))
    #     # source_mag_list = torch.std(sourceArray, axis=(1,2))
    #     # print(outP.shape)
    #     # print(torch.std(outP))
    #     # sourceArray = sourceArray / source_mag_list.unsqueeze(-1).unsqueeze(-1) / pressure_factor.to(self.device)
    #     # sourceArray = sourceArray / source_mag_list.unsqueeze(-1).unsqueeze(-1) / pressure_factor.to(self.device)
    #
    #     for i in range(len(outP)):
    #         sparse_matrix = self.setup_sparse_matrix(dArray[i].view(-1), faceArray[i].view(-1))
    #         predicted_source = torch.matmul(sparse_matrix, outP[i])
    #         # a = predicted_source.squeeze(-1)
    #         # b = sourceArray[i].squeeze(-1).squeeze(-1)
    #         # print(torch.mean(abs(a)))
    #         # print(torch.mean(abs(b)))
    #         # print(torch.dot(a,b) / torch.dot(a,a))
    #         # print(torch.std(sourceArray[i])/torch.std(predicted_source))
    #
    #         # predicted_source = predicted_source / source_mag_list[i]
    #         print(torch.mean(abs(predicted_source-sourceArray[i])))
    #         # print(source_mag_list[i],torch.std(outP[i]))
    #         # print("source", sourceArray.shape, predicted_source.shape)
    #         # print(cellWeights.shape, predicted_source.shape)
    #         if countCellweights:
    #             loss += torch.sum(cellWeights.unsqueeze(1) * ((sourceArray[i] - predicted_source) ** 2)) / outP.numel()
    #         else:
    #             loss += torch.sum(((sourceArray[i] - predicted_source) ** 2)) / outP.numel()
    #
    #     # predictedSources = self.getPredictedSources(outP, dArray, faceArray)
    #     # # print(torch.sum((sourceArray - predictedSources) ** 2))
    #
    #     return loss


    def computeLossPressure(self, outP, pArray):
        shape_p = np.shape(outP)

        # print("pressure", outP.shape, pArray.shape)
        # pressure_mag_list = torch.mean(abs(pArray), axis=1).unsqueeze(-1)
        pressure_mag_list = torch.std(pArray, axis=1).unsqueeze(-1)

        # print("fsdfs",pressure_mag_list)

        # print(pressure_mag_list)

        pArray = pArray/pressure_mag_list
        outP = outP / pressure_mag_list.unsqueeze(-1)
        # print(torch.mean(abs(outP.squeeze(-1)-pArray)),torch.std(outP))


        return torch.sum(((((outP.squeeze(-1)) - (pArray))) ** 2)) / shape_p[1]
    #
    # def computeLossFVM(self, outP, pArray,faceArray):
    #     # print("fdsgfd")
    #     # print(outP.shape)
    #     # print(faceArray.shape)
    #     # print(pArray.shape)
    #
    #     p_factor = torch.std(pArray, axis=1)
    #     face_factor = torch.mean((faceArray), axis=(1,2))
    #
    #     faceArray = faceArray.squeeze(-1) / face_factor.unsqueeze(-1)
    #     pArray = pArray / p_factor.unsqueeze(-1)
    #     outP = outP.squeeze(-1) / p_factor.unsqueeze(-1)
    #     # print(face_factor)
    #
    #     flux_ground_truth = (pArray[:,self.FVM_face_indices_1] - pArray[:,self.FVM_face_indices_0]) * faceArray
    #     flux_prediction = (outP[:,self.FVM_face_indices_1] - outP[:,self.FVM_face_indices_0]) * faceArray
    #     # print("asdfghjk")
    #     # print(torch.sum((flux_prediction - flux_ground_truth) ** 2) / faceArray.numel())
    #     # print(torch.mean((flux_prediction - flux_ground_truth) ** 2))
    #
    #     return torch.mean((flux_prediction - flux_ground_truth) ** 2)
    #
    #     # print(flux_ground_truth.shape)
    #     # print(flux_prediction.shape)

if __name__ == '__main__':
    from embed_data import *
    # Define the paths
    import time

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    device = 'cuda'
    iTab = np.arange(128)

    embedClass = EmbedData(test_case_path, device)
    t1 = time.time()
    dArray, faceArray, sourceArray = embedClass.getData(iTab,getPressure=False)
    print("embedClass: ",time.time() - t1)

    pNetwork = torch.randn((sourceArray.shape), dtype=torch.float, device=device)

    lossClass = LossClass(test_case_path, device)
    t1 = time.time()
    loss = lossClass.computeLossSource(pNetwork, dArray, faceArray, sourceArray,1,False)
    loss = lossClass.computeLossFVM(pNetwork, dArray, faceArray, sourceArray,1,False)

    print("lossClass: ",time.time() - t1)
    print("loss: ", loss)

    # Print the recognized device
    if device == 'cuda':
        print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Running on CPU")
