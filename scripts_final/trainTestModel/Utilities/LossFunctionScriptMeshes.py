import numpy as np
import os
import torch
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import sys
import pickle
class LossClass:
    def __init__(self, test_case_path,train_cases, device='cuda',t=0):
        self.device = device
        self.FVM_face_indices_0_tab = []
        self.FVM_face_indices_1_tab = []
        self.indices_tab = []
        self.num_nodes_tab = []
        for i_case in train_cases:
            casePath = os.path.join(test_case_path,f"case_{i_case}" ,"level0")
            self.prepare_FVM_loss(casePath)
            self.get_indices(casePath)

    def prepare_FVM_loss(self,casePath):
        sparce_matrix_data = np.loadtxt(os.path.join(casePath, "constant/polyMesh/faceCells.txt")).astype(int)
        i_boundary_cells = np.any(sparce_matrix_data == -1, axis=1)

        filtered_data = sparce_matrix_data[~i_boundary_cells]
        self.FVM_face_indices_0_tab.append(filtered_data[:,0])
        self.FVM_face_indices_1_tab.append(filtered_data[:,1])

    def get_indices(self,casePath):
        sparce_matrix_data = np.loadtxt(os.path.join(casePath, "constant/polyMesh/faceCells.txt")).astype(int)
        self.num_nodes_tab.append(np.max(sparce_matrix_data) + 1)
        print(self.num_nodes_tab[-1])

        i_boundary_cells = np.any(sparce_matrix_data == -1, axis=1)

        filtered_data = sparce_matrix_data[~i_boundary_cells]

        x = np.concatenate((np.arange(self.num_nodes_tab[-1]), filtered_data[:, 0], filtered_data[:, 1]))
        y = np.concatenate((np.arange(self.num_nodes_tab[-1]), filtered_data[:, 1], filtered_data[:, 0]))

        self.indices_tab.append(torch.LongTensor(np.array([y, x])))

    def getMatrixValues(self,d,face):
        return torch.concatenate((d, face, face))

    def setup_sparse_matrix(self,d,face,iSet):
        values = self.getMatrixValues(d,face)
        return torch.sparse_coo_tensor(self.indices_tab[iSet].to(self.device), values, (self.num_nodes_tab[iSet], self.num_nodes_tab[iSet])).to(self.device)

    def computeLossSource(self, outP, pArray, dArray, faceArray, sourceArray,pressure_factor,iSet):
        loss = 0

        shape_p = np.shape(outP)

        for i in range(len(outP)):
            sparse_matrix = self.setup_sparse_matrix(dArray[i].view(-1), faceArray[i].view(-1),iSet)
            # predicted_source = torch.matmul(sparse_matrix, outP[i])

            diff_source = torch.matmul(sparse_matrix, pArray[i]-outP[i])
            loss += torch.sum(((diff_source) ** 2)) / shape_p[1]

        return loss


    def computeLossPressure(self, outP, pArray):
        shape_p = np.shape(outP)

        pressure_mag_list = torch.std(pArray, axis=1).unsqueeze(-1)
        pArray = pArray/pressure_mag_list
        outP = outP / pressure_mag_list.unsqueeze(-1)

        return torch.sum(((((outP.squeeze(-1)) - (pArray))) ** 2)) / shape_p[1]

    def computeLossFVM(self, outP, pArray,faceArray,iSet):
        shape_p = np.shape(outP)

        p_factor = torch.std(pArray, axis=1)
        face_factor = torch.mean((faceArray), axis=(1,2))

        faceArray = faceArray.squeeze(-1) / face_factor.unsqueeze(-1)
        pArray = pArray / p_factor.unsqueeze(-1)
        outP = outP.squeeze(-1) / p_factor.unsqueeze(-1)

        flux_ground_truth = (pArray[:,self.FVM_face_indices_1_tab[iSet]] - pArray[:,self.FVM_face_indices_0_tab[iSet]]) * faceArray
        flux_prediction = (outP[:,self.FVM_face_indices_1_tab[iSet]] - outP[:,self.FVM_face_indices_0_tab[iSet]]) * faceArray

        return torch.sum((flux_prediction - flux_ground_truth) ** 2) / shape_p[1]


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
