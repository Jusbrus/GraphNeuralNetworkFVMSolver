import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from torch.utils.data import TensorDataset, DataLoader


# def loadDataLoader(test_case_path,iTab,batchSize,getPressure=True):
#     embedder = EmbedData(test_case_path, 'cpu')
#     if getPressure:
#         pArray ,dArray, faceArray, sourceArray = embedder.getData(iTab,getPressure)
#         dataset = TensorDataset(pArray ,dArray, faceArray, sourceArray)
#     else:
#         dArray, faceArray, sourceArray = embedder.getData(iTab,getPressure)
#         dataset = TensorDataset(dArray, faceArray, sourceArray)
#     return DataLoader(dataset, batch_size=batchSize, shuffle=False)

class EmbedData:

    def __init__(self, test_case_path, device,is_cfd_data):
        self.test_case_path = os.path.join(test_case_path, "level0")
        self.test_case_path_datadict = test_case_path
        self.device = device
        self.is_cfd_data = is_cfd_data
        if self.is_cfd_data:
            self.tag = ''

        else:
            self.tag = '_saved'

    def read_file_lines(self, filepath, start, end):
        with open(filepath, 'r') as f:
            lines = f.readlines()[start:end]
        return np.fromstring(''.join(lines), sep='\n')

    def load_data(self, localPath):
        pEqnD = self.read_file_lines(os.path.join(localPath, "pEqnD"), 21, -4)
        pEqnFace = self.read_file_lines(os.path.join(localPath, "pEqnFace"), 21, -4)
        pEqnSource = self.read_file_lines(os.path.join(localPath, "pEqnSource"), 21, -4)
        nCells = len(pEqnSource)
        return pEqnD, pEqnFace, pEqnSource

    def load_p(self, localPath,nCells,name="p"):
        return self.read_file_lines(os.path.join(localPath, name), 23, 23 + nCells)

    def load_pInitial(self, localPath,nCells):
        return self.read_file_lines(os.path.join(localPath, "pInitial"), 23, 23 + nCells)

    def getData(self, iTab,getPressure=True):
        pArray = []
        dArray = []
        faceArray = []
        sourceArray = []

        for i in iTab:
            try:
                if i%50==0:
                    print(i)
            except:
                pass
            localPath = os.path.join(self.test_case_path, f"{i}{self.tag}")
            pEqnD, pEqnFace, pEqnSource = self.load_data(localPath)

            dArray.append(pEqnD)
            faceArray.append(pEqnFace)
            sourceArray.append(pEqnSource)

            if getPressure:
                nCells = len(pEqnSource)
                p = self.load_p(localPath,nCells)
                pArray.append(p)

        # np.expand_dims(dArray, axis=-1)
        dArray = torch.FloatTensor(np.expand_dims(dArray, axis=-1)).to(self.device)
        faceArray = torch.FloatTensor(np.expand_dims(faceArray, axis=-1)).to(self.device)
        sourceArray = torch.FloatTensor(np.expand_dims(sourceArray, axis=-1)).to(self.device)

        if getPressure:
            pArray = torch.FloatTensor(np.array(pArray)).to(self.device)
            return pArray, dArray, faceArray, sourceArray
        else:
            return dArray, faceArray, sourceArray

    def getp(self, iTab,nCells,name="p"):
        pArray = []

        for i in iTab:
            try:
                if i%50==0:
                    print(i)
            except:
                pass
            localPath = os.path.join(self.test_case_path, f"{i}{self.tag}")

            p = self.load_p(localPath,nCells,name=name)
            pArray.append(p)

        pArray = torch.FloatTensor(np.array(pArray)).to(self.device)
        return pArray


    def getpEqnD(self, iTab):
        dArray = []

        for i in iTab:
            try:
                if i%50==0:
                    print(i)
            except:
                pass
            localPath = os.path.join(self.test_case_path, f"{i}{self.tag}")
            pEqnD = self.read_file_lines(os.path.join(localPath, "pEqnD"), 21, -4)

            dArray.append(pEqnD)

        # np.expand_dims(dArray, axis=-1)
        dArray = torch.FloatTensor(np.expand_dims(dArray, axis=-1)).to(self.device)

        return dArray

    def getpEqnFace(self, iTab):
        faceArray = []

        for i in iTab:
            try:
                if i%50==0:
                    print(i)
            except:
                pass
            localPath = os.path.join(self.test_case_path, f"{i}{self.tag}")
            pEqnFace = self.read_file_lines(os.path.join(localPath, "pEqnFace"), 21, -4)

            faceArray.append(pEqnFace)

        # np.expand_dims(dArray, axis=-1)
        faceArray = torch.FloatTensor(np.expand_dims(faceArray, axis=-1)).to(self.device)

        return faceArray

    def getpEqnSource(self, iTab):
        sourceArray = []

        for i in iTab:
            try:
                if i%50==0:
                    print(i)
            except:
                pass
            localPath = os.path.join(self.test_case_path, f"{i}{self.tag}")
            pEqnSource = self.read_file_lines(os.path.join(localPath, "pEqnSource"), 21, -4)

            sourceArray.append(pEqnSource)

        sourceArray = torch.FloatTensor(np.expand_dims(sourceArray, axis=-1)).to(self.device)
        nCells = len(pEqnSource)

        return sourceArray, nCells

    # def getBoundaryCondition(self, iTab):
    #     file_path = f"{test_case_path}/dataDictMeshes.pkl"
    #     with open(file_path, 'rb') as file:
    #         dataDictMeshes = pickle.load(file)
    #
    #     boundaryFacesDict = dataDictMeshes[0]["boundaryFacesDict"]
    #     print(dataDictMeshes[0]["boundaryFaces"])
    #
    #     dataArray = np.zeros((len(iTab), dataDictMeshes[0]["Nfaces"],4))
    #
    #     for i in range(len(iTab)):
    #         if i%50==0:
    #             print("BC",i)
    #         localPath = os.path.join(self.test_case_path, f"{iTab[i]}{self.tag}/p")
    #         p_file = ParsedParameterFile(localPath)
    #         # Access the data
    #         p_data = p_file.content
    #
    #         # print(p_data["boundaryField"].keys())
    #         for boundary in p_data["boundaryField"].keys():
    #             if p_data["boundaryField"][boundary]["type"] == "empty":
    #                 pass
    #                 # print("this is a Side")
    #             elif p_data["boundaryField"][boundary]["type"] == "fixedValue":
    #                 # print(p_data["boundaryField"][boundary]["value"])
    #                 # print(p_data["boundaryField"][boundary]["value"][:])
    #                 dataArray[iTab[i], boundaryFacesDict[boundary], 1] = 1
    #                 dataArray[iTab[i], boundaryFacesDict[boundary], 3] = p_data["boundaryField"][boundary]["value"][:]
    #
    #             elif p_data["boundaryField"][boundary]["type"] == "fixedGradient":
    #                 # print(p_data["boundaryField"][boundary])
    #                 dataArray[iTab[i], boundaryFacesDict[boundary], 0] = 1
    #                 # print(p_data["boundaryField"][boundary])
    #                 dataArray[iTab[i], boundaryFacesDict[boundary], 2] = p_data["boundaryField"][boundary]["gradient"][:]
    #
    #
    #             else:
    #                 dataArray[iTab[i], boundaryFacesDict[boundary], 0] = 1
    #                 # print("testss")
    #
    #             # print(p_data["boundaryField"][boundary]["type"])
    #             # print(p_data["boundaryField"][boundary])
    #
    #             # if p_data["boundaryField"][boundary]["type"]=="neumann":
    #         # print(p_data.keys())
    #
    #     dataArray = dataArray[:,dataDictMeshes[0]["boundaryFaces"],:]
    #
    #     return torch.FloatTensor(dataArray).to(self.device)


    def getBoundaryCondition(self, iTab,name="p"):
        file_path = f"{self.test_case_path_datadict}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshes = pickle.load(file)

        boundaryFacesDict = dataDictMeshes[0]["boundaryFacesDict"]
        # print(dataDictMeshes[0]["boundaryFaces"])

        dataArray = np.zeros((len(iTab), dataDictMeshes[0]["Nfaces"],4))

        for i in range(len(iTab)):
            if i%50==0:
                print("BC",i)
            localPath = os.path.join(self.test_case_path, f"{iTab[i]}{self.tag}/{name}")
            p_file = ParsedParameterFile(localPath)
            # Access the data
            p_data = p_file.content

            # print(p_data["boundaryField"].keys())

            for boundary in p_data["boundaryField"].keys():
                # print(p_data["boundaryField"][boundary]["type"])
                # print(p_data["boundaryField"][boundary].keys())
                # print(p_data["boundaryField"][boundary]["value"])
                if p_data["boundaryField"][boundary]["type"] == "empty":
                    pass
                    # print("this is a Side")
                elif p_data["boundaryField"][boundary]["type"] == "fixedValue" or p_data["boundaryField"][boundary]["type"] == "inletOutlet":
                    # print(p_data["boundaryField"][boundary]["value"])
                    # print(p_data["boundaryField"][boundary]["value"][:])
                    dataArray[i, boundaryFacesDict[boundary], 1] = 1
                    dataArray[i, boundaryFacesDict[boundary], 3] = p_data["boundaryField"][boundary]["value"][:]

                elif p_data["boundaryField"][boundary]["type"] == "fixedGradient":
                    # print(p_data["boundaryField"][boundary])
                    dataArray[i, boundaryFacesDict[boundary], 0] = 1
                    # print(p_data["boundaryField"][boundary])
                    dataArray[i, boundaryFacesDict[boundary], 2] = p_data["boundaryField"][boundary]["gradient"][:]

                else:
                    dataArray[i, boundaryFacesDict[boundary], 0] = 1
                    # print("testss")

                # print(p_data["boundaryField"][boundary]["type"])
                # print(p_data["boundaryField"][boundary])

                # if p_data["boundaryField"][boundary]["type"]=="neumann":
            # print(p_data.keys())

        dataArray = dataArray[:,dataDictMeshes[0]["boundaryFaces"],:]

        return torch.FloatTensor(dataArray).to(self.device)


    def getBoundaryCondition_cfd(self, iTab):
        file_path = f"{self.test_case_path_datadict}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshes = pickle.load(file)

        boundaryFacesDict = dataDictMeshes[0]["boundaryFacesDict"]
        # print(dataDictMeshes[0]["boundaryFaces"])

        dataArray = np.zeros((len(iTab), dataDictMeshes[0]["Nfaces"],4))

        p_data = {}
        p_data["boundaryField"] = {"Left": {"type":"zeroGradient"},
                                   "Upper": {"type":"zeroGradient"},
                                   "Right": {"type":"fixedValue"},
                                   "Lower": {"type":"zeroGradient"},
                                   "Object": {"type":"zeroGradient"},}

        for i in range(len(iTab)):
            if i%50==0:
                print("BC",i)
            # localPath = os.path.join(self.test_case_path, f"{iTab[i]}{self.tag}/pInitial")
            # p_file = ParsedParameterFile(localPath)
            # # Access the data
            # p_data = p_file.content
            #
            # print(p_data["boundaryField"].keys())

            # print(p_data["boundaryField"].keys())
            for boundary in p_data["boundaryField"].keys():
                if p_data["boundaryField"][boundary]["type"] == "empty":
                    pass
                    # print("this is a Side")
                elif p_data["boundaryField"][boundary]["type"] == "fixedValue" or p_data["boundaryField"][boundary]["type"] == "inletOutlet":
                    # print(p_data["boundaryField"][boundary]["value"])
                    # print(p_data["boundaryField"][boundary]["value"][:])
                    dataArray[i, boundaryFacesDict[boundary], 1] = 1
                    # dataArray[i, boundaryFacesDict[boundary], 3] = p_data["boundaryField"][boundary]["value"][:]

                elif p_data["boundaryField"][boundary]["type"] == "fixedGradient":
                    # print(p_data["boundaryField"][boundary])
                    dataArray[i, boundaryFacesDict[boundary], 0] = 1
                    # print(p_data["boundaryField"][boundary])
                    # dataArray[i, boundaryFacesDict[boundary], 2] = p_data["boundaryField"][boundary]["gradient"][:]

                else:
                    dataArray[i, boundaryFacesDict[boundary], 0] = 1
                    # print("testss")

                # print(p_data["boundaryField"][boundary]["type"])
                # print(p_data["boundaryField"][boundary])

                # if p_data["boundaryField"][boundary]["type"]=="neumann":
            # print(p_data.keys())

        dataArray = dataArray[:,dataDictMeshes[0]["boundaryFaces"],:]

        return torch.FloatTensor(dataArray).to(self.device)


def save_tensors(test_case_path, nDatasets,getPressure=True,device='cpu',is_cfd_data=False,itab=None):
    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    if itab is None:
        itab = np.arange(nDatasets)+1

    file_path = f"{test_case_path}/itab"
    with open(file_path, 'wb') as file:
        pickle.dump(itab, file)


    embedder = EmbedData(test_case_path, 'cpu',is_cfd_data)
    if is_cfd_data:
        dataArray_initial = embedder.getBoundaryCondition_cfd(itab)
        torch.save(dataArray_initial, os.path.join(test_case_path, 'boundaryConditionCfd.pt'))
        torch.save(dataArray_initial, os.path.join(test_case_path, 'boundaryConditionArray.pt'))

    else:
        dataArray = embedder.getBoundaryCondition(itab)
        torch.save(dataArray, os.path.join(test_case_path, 'boundaryConditionArray.pt'))

    dArray = embedder.getpEqnD(itab)
    print(dArray.shape)

    torch.save(dArray, os.path.join(test_case_path, 'dArray.pt'))

    faceArray = embedder.getpEqnFace(itab)
    torch.save(faceArray, os.path.join(test_case_path, 'faceArray.pt'))

    sourceArray, nCells = embedder.getpEqnSource(itab)
    torch.save(sourceArray, os.path.join(test_case_path, 'sourceArray.pt'))

    if getPressure:
        if is_cfd_data:
            pInitialArray = embedder.getp(itab, nCells,name="pInitial")
            torch.save(pInitialArray, os.path.join(test_case_path, 'pInitialArray.pt'))

            pAfter = embedder.getp(itab, nCells,name="pAfter")
            torch.save(pAfter, os.path.join(test_case_path, 'pArray.pt'))

        else:
            pArray = embedder.getp(itab, nCells, name="p")
            torch.save(pArray, os.path.join(test_case_path, 'pArray.pt'))


def get_faceArray_incl_boundaries(test_case_path,device='cpu'):
    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    dArray = torch.load(os.path.join(test_case_path, 'dArray.pt'))
    faceArray = torch.load(os.path.join(test_case_path, 'faceArray.pt'))

    faceArray_shape = faceArray.shape

    faceArray_zeros = np.zeros((faceArray_shape[0], len(dataDictMeshes[0]["boundaryCells"]), 1))
    faceArray_incl_zeros = np.concatenate((faceArray, faceArray_zeros), axis=1)
    cell_faces_boundary_cells = dataDictMeshes[0]["cellFaces"][dataDictMeshes[0]["boundaryCells"]]

    print(cell_faces_boundary_cells)

    face_values_boundaries_sum = torch.FloatTensor(
        np.sum(faceArray_incl_zeros[:, cell_faces_boundary_cells, :], axis=2)).to(device)

    boundary_values = - dArray[:, dataDictMeshes[0]["boundaryCells"], :] - face_values_boundaries_sum
    # print(boundary_values)
    faceArray_incl_boundaries = torch.FloatTensor(np.concatenate((faceArray, boundary_values), axis=1)).to(device)

    # Deleting large arrays to free up memory
    del dArray
    del faceArray

    print(faceArray_incl_boundaries.shape)

    torch.save(faceArray_incl_boundaries, os.path.join(test_case_path, 'faceArray_incl_boundaries.pt'))



def loadDataLoader(test_case_path,iTab,batchSize,dataDictMeshes,BCInput=True,gradientBCInput=False,doShuffling=True, normalize_data=False,compute_diff_cfd=False):
    pArray = torch.load(os.path.join(test_case_path, 'pArray.pt'))[iTab]
    # print("pArray.shape",pArray.shape)

    dArray = torch.load(os.path.join(test_case_path, 'dArray.pt'))[iTab]
    faceArray_incl_boundaries = torch.load(os.path.join(test_case_path, 'faceArray_incl_boundaries.pt'))[iTab]
    faceArray = torch.load(os.path.join(test_case_path, 'faceArray.pt'))[iTab]
    sourceArray = torch.load(os.path.join(test_case_path, 'sourceArray.pt'))[iTab]
    # print("sourceArray.shape",sourceArray.shape)
    # print(sourceArray.shape)

    if compute_diff_cfd:
        BCArray = torch.load(os.path.join(test_case_path, 'boundaryConditionCfd.pt'))[iTab]
    else:
        BCArray = torch.load(os.path.join(test_case_path, 'boundaryConditionArray.pt'))[iTab]

    if compute_diff_cfd == False:
        sparce_matrix_data = dataDictMeshes[0]["faceCells"]
        boundaryCells = dataDictMeshes[0]["boundaryCells"]

        print(sparce_matrix_data)
        print(len(sparce_matrix_data))



        print(dataDictMeshes[0].keys())

        # boundaryCells = np.loadtxt(
        #     os.path.join(test_case_path, "level0/constant/polyMesh/boundaryCells.txt")).astype(int)

        num_nodes = np.max(sparce_matrix_data) + 1

        i_boundary_faces = np.any(sparce_matrix_data == -1, axis=1)

        boundary_cells = sparce_matrix_data[i_boundary_faces][:,0]

        filtered_data = sparce_matrix_data[~i_boundary_faces]

        x = np.concatenate((np.arange(num_nodes), filtered_data[:, 0], filtered_data[:, 1]))
        y = np.concatenate((np.arange(num_nodes), filtered_data[:, 1], filtered_data[:, 0]))

        indices = torch.LongTensor(np.array([y, x]))

        print(faceArray_incl_boundaries.shape)
        print(i_boundary_faces.shape)

        faceArray_only_boundaries= faceArray_incl_boundaries[:,i_boundary_faces,0]

        for i in range(sourceArray.shape[0]):

            values = torch.concatenate((dArray[i], faceArray[i], faceArray[i])).squeeze(-1)

            sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

            # dense_matrix = sparse_matrix.to_dense()

            a = torch.matmul(sparse_matrix, (pArray[i]))


            b_dirichlet = BCArray[i,:,3]*faceArray_only_boundaries[i]
            # print(b_dirichlet.shape)
            # plt.plot(b_dirichlet,'o')

            # print(boundaryCells)
            a[boundaryCells] = a[boundaryCells] + b_dirichlet

            sourceArray[i,:,0] = a+0

            # plt.figure()
            # plt.plot(BCArray[i,:,3],'o')
            # plt.plot(pArray[i,boundary_cells]*BCArray[i,:,1],'o')
            #
            # plt.plot((pArray[i,boundary_cells]-BCArray[i,:,3])*BCArray[i,:,1],'o')
            #
            # plt.show()

            # # Calculate row sums
            # row_sums = torch.sum(dense_matrix, dim=1)
            #
            # row_sums[boundaryCells] += faceArray_only_boundaries[i]
            # plt.plot(row_sums,"o")
            # plt.plot(boundaryCells,faceArray_only_boundaries[i],"o")

        #     plt.show()
        #
        #     threshold = 0.1
        #     rows_above_threshold = torch.nonzero(abs(row_sums) > threshold).squeeze()
        #
        #     # Print the row sums
        #     print("Row sums:", row_sums)
        #     print(torch.sum(abs(sourceArray[i,:,0]-a)))
        #     # Create a single figure with subplots
        #     plt.figure(figsize=(10, 8))
        #
        #     # First plot
        #     plt.subplot(2, 2, 1)
        #     plt.plot(a, 'o')
        #     plt.plot(boundaryCells,a[boundaryCells],'o')
        #     plt.plot(rows_above_threshold,a[rows_above_threshold],'o')
        #     plt.title('Plot of a')
        #
        #     # Second plot
        #     plt.subplot(2, 2, 2)
        #     plt.plot(sourceArray[i, :, 0], 'o')
        #     plt.plot(boundaryCells,sourceArray[i, :, 0][boundaryCells],'o')
        #     plt.plot(rows_above_threshold,sourceArray[i, :, 0][rows_above_threshold],'o')
        #
        #     plt.title('Plot of sourceArray[i, :, 0]')
        #
        #     # Third plot
        #     plt.subplot(2, 2, 3)
        #     plt.plot(sourceArray[i, :, 0] / torch.std(sourceArray[i, :, 0]), 'o')
        #     plt.plot(boundaryCells,(sourceArray[i, :, 0] / torch.std(sourceArray[i, :, 0]))[boundaryCells],'o')
        #     plt.plot(rows_above_threshold,(sourceArray[i, :, 0] / torch.std(sourceArray[i, :, 0]))[rows_above_threshold],'o')
        #
        #     plt.title('Plot of sourceArray[i, :, 0] / std')
        #
        #     # Fourth plot
        #     plt.subplot(2, 2, 4)
        #     plt.plot(sourceArray[i, :, 0] - a, 'o')
        #     plt.plot(boundaryCells,(sourceArray[i, :, 0] - a)[boundaryCells],'o')
        #     plt.plot(rows_above_threshold,(sourceArray[i, :, 0] - a)[rows_above_threshold],'o')
        #
        #     plt.title('Plot of sourceArray[i, :, 0] - a')
        #
        #     # Adjust layout and show the plot
        #     plt.tight_layout()
        #
        #     # plt.figure(figsize=(10, 8))
        #     # plt.plot(row_sums,"o")
        #     plt.show()
        #
        #     # # Create the histogram
        #     # plt.hist(sourceArray[i,:,0], bins=30, edgecolor='black')  # 30 bins, black edges
        #     #
        #     # # Add labels and title
        #     # plt.xlabel('Value')
        #     # plt.ylabel('Frequency')
        #     # plt.title('Histogram of Data')
        #     #
        #     # # Show the plot
        #     # plt.show()
        #     print("fsf")
        # gfd



    if compute_diff_cfd:
        pInitialArray = torch.load(os.path.join(test_case_path, 'pInitialArray.pt'))[iTab]

        sparce_matrix_data = np.loadtxt(
            os.path.join(test_case_path, "level0/constant/polyMesh/faceCells.txt")).astype(int)
        num_nodes = np.max(sparce_matrix_data) + 1

        i_boundary_faces = np.any(sparce_matrix_data == -1, axis=1)

        filtered_data = sparce_matrix_data[~i_boundary_faces]

        x = np.concatenate((np.arange(num_nodes), filtered_data[:, 0], filtered_data[:, 1]))
        y = np.concatenate((np.arange(num_nodes), filtered_data[:, 1], filtered_data[:, 0]))

        indices = torch.LongTensor(np.array([y, x]))

        sourceInitial = torch.zeros_like(sourceArray)
        # print(sourceInitial.shape)
        for i in range(sourceArray.shape[0]):
            values = torch.concatenate((dArray[i], faceArray[i], faceArray[i])).squeeze(-1)
            # print("dsfsfsfsf")
            # print(values.shape)

            sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

            sourceArray[i,:,0] = torch.matmul(sparse_matrix, (pArray[i]-pInitialArray[i]))

            # print(torch.mean(abs(sourceArray[i]-torch.matmul(sparse_matrix, pArray[i]))))
            # print(torch.mean(abs(sourceArray[i])))
        # sourceArray = sourceArray-sourceInitial
        pArray = pArray-pInitialArray
        BCArray[:,:,[2,3]] = 0

    if normalize_data:
        A_factor = torch.mean(abs(dArray),axis=(1,2)).unsqueeze(-1).unsqueeze(-1)
        # source_factor = torch.mean(abs(sourceArray),axis=(1,2)).unsqueeze(-1).unsqueeze(-1)
        source_factor = torch.std(sourceArray,axis=(1,2)).unsqueeze(-1).unsqueeze(-1)

        dArray = dArray / A_factor
        faceArray = faceArray / A_factor
        faceArray_incl_boundaries = faceArray_incl_boundaries /  A_factor

        sourceArray = sourceArray / source_factor

        pressure_factor_factorSourceA = source_factor / A_factor
        # print(sourceArray.shape)
        # print(pressure_factorSourceA.shape)
        # print(pArray.shape)
        # print(pressure_factor.unsqueeze(-1).shape)
        pArray = pArray / pressure_factor_factorSourceA.squeeze(-1)

        BCArray[:,:,[2,3]] = BCArray[:,:,[2,3]] / pressure_factor_factorSourceA

        pressure_factor_norm = torch.std(pArray,axis=1).unsqueeze(-1)
        pArray = pArray / pressure_factor_norm
        print("factorrrsss")
        print(torch.mean(pressure_factor_norm),torch.mean(pressure_factor_factorSourceA),torch.mean(source_factor),torch.mean(A_factor))

        print(pArray.shape)
        print(pressure_factor_norm.shape)

        print(BCArray[:,:,[2,3]].shape)
        print("fdsaafdsafdsas")
        print(pressure_factor_norm.shape)
        print(pressure_factor_factorSourceA.shape)
        BCArray[:,:,[2,3]] = BCArray[:,:,[2,3]] / pressure_factor_norm.unsqueeze(-1)

    if compute_diff_cfd:
        BCArray = BCArray[:,:,np.array([0,1])]

    elif gradientBCInput == False:
        BCArray = BCArray[:,:,np.array([0,1,3])]

    inputCellArray = torch.cat((sourceArray,dArray), dim=2)

    # print(pArray.shape)
    # print(sourceArray.shape)
    # print(dArray.shape)
    # print(faceArray.shape)
    # p_avg = torch.mean(abs(faceArray),axis=1)
    # import matplotlib.pyplot as plt
    # plt.scatter(p_avg.detach().numpy(),np.ones(len(p_avg.detach().numpy())))
    # plt.show()
    if normalize_data:
        print("NORMALIZE DATA")

        if compute_diff_cfd:
            dataset = TensorDataset(pArray, dArray, faceArray, sourceArray,inputCellArray, faceArray_incl_boundaries, BCArray,A_factor.squeeze(-1).squeeze(-1),
                                    source_factor.squeeze(-1).squeeze(-1),
                                    pressure_factor_factorSourceA.squeeze(-1).squeeze(-1),pressure_factor_norm.squeeze(-1),pInitialArray)
        else:
            dataset = TensorDataset(pArray, dArray, faceArray, sourceArray,inputCellArray, faceArray_incl_boundaries, BCArray,A_factor.squeeze(-1).squeeze(-1),
                                    source_factor.squeeze(-1).squeeze(-1),
                                    pressure_factor_factorSourceA.squeeze(-1).squeeze(-1),pressure_factor_norm.squeeze(-1))
        # dataset = TensorDataset(pArray, dArray, faceArray, sourceArray, inputCellArray, faceArray_incl_boundaries, BCArray)

    else:
        dataset = TensorDataset(pArray, dArray, faceArray, sourceArray,inputCellArray, faceArray_incl_boundaries, BCArray)

    return DataLoader(dataset, batch_size=batchSize, shuffle=doShuffling)


def test_data(test_case_path,i,is_cfd_data=False,max_pos_list = [],max_mag_list = []):

    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    # print(dataDictMeshes[0]["faceCoordinates"])
    # print(dataDictMeshes[0]["boundaryFaces"])
    coordinates_boundary_faces = dataDictMeshes[0]["faceCoordinates"][dataDictMeshes[0]["boundaryFaces"]]
    # print(coordinates_boundary_faces)
    embedder = EmbedData(test_case_path, 'cpu',is_cfd_data)
    dArray = embedder.getpEqnD([i])
    faceArray = embedder.getpEqnFace([i])
    sourceArray, nCells = embedder.getpEqnSource([i])
    pArray = embedder.getp([i], nCells)
    pAfter = embedder.getp([i], nCells,name="pAfter")
    pInitial = embedder.getp([i], nCells,name="pInitial")

    cellPoints = dataDictMeshes[0]["cellPoints"]
    x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
    y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]

    sparce_matrix_data = np.loadtxt(
        os.path.join(test_case_path, "level0/constant/polyMesh/faceCells.txt")).astype(int)
    num_nodes = np.max(sparce_matrix_data) + 1

    # print(dataDictMeshes[0]["cellFaces"][166])
    faces = dataDictMeshes[0]["cellFaces"][166]
    # print(dataDictMeshes[0]["cellCenters"][166])

    i_boundary_cells = np.any(sparce_matrix_data == -1, axis=1)
    # print(np.where(i_boundary_cells)[0])

    filtered_data = sparce_matrix_data[~i_boundary_cells]

    x = np.concatenate((np.arange(num_nodes), filtered_data[:, 0], filtered_data[:, 1]))
    y = np.concatenate((np.arange(num_nodes), filtered_data[:, 1], filtered_data[:, 0]))

    indices = torch.LongTensor(np.array([y, x]))

    values = torch.concatenate((dArray[0], faceArray[0], faceArray[0])).squeeze(-1)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    sourcediff = torch.matmul(sparse_matrix, pArray[0]-pInitial[0])

    boundary_condition = embedder.getBoundaryCondition([i],name="p")
    boundary_condition_initial = embedder.getBoundaryCondition_initial([i])

    # print(boundary_condition.shape)

    f = -dArray[0,166]-torch.sum(faceArray[0,faces[:-1]])
    flux_boundary = boundary_condition[0,0,3] * f

    # print(flux_boundary)

    # print(flux_boundary,sourceRec[166],sourceArray[0,166],flux_boundary+sourceRec[166])
    import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(boundary_condition[0, :, 3],"o")
    # plt.plot(boundary_condition_initial[0, :, 3],"o")
    # plt.plot(boundary_condition[0, :, 3]-boundary_condition_initial[0, :, 3],"o")
    # plt.show()

    i_high_diff = np.where(abs(boundary_condition[0, :, 3]-boundary_condition_initial[0, :, 3])>0.003)[0]
    # print(max(abs(boundary_condition[0, :, 3]-boundary_condition_initial[0, :, 3])))
    pDiff = abs(pArray[0]-pInitial[0])

    max_mag_list.append(max(abs(boundary_condition[0, :, 3]-boundary_condition_initial[0, :, 3])))
    # max_mag_list.append(torch.mean((pArray)))
    # print(pDiff.shape)
    # print(torch.argmax(pDiff))
    # print(dataDictMeshes[0]["cellCenters"][torch.argmax(pDiff).item()])

    # max_pos = dataDictMeshes[0]["cellCenters"][torch.argmax(pDiff).item()]
    # if max_pos[1]>7 or max_pos[1]<-7:
    #     if max_pos[0]<-7:
    #         max_pos_list.append(max_pos)
    #         print(dataDictMeshes[0]["cellCenters"][torch.argmax(pDiff).item()])
    #         print(i)

            # fig = plt.figure(figsize=(24, 13))
            # ax = fig.add_subplot(2, 1, 1)
            # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pAfter[0]-pInitial[0], "pressure")
            # ax = fig.add_subplot(2, 1, 2)
            # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, sourcediff, "source")
            #
            # plt.show()

    fig = plt.figure(figsize=(24, 13))
    ax = fig.add_subplot(2, 2, 1)
    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pAfter[0]-pInitial[0], "pressure diff")
    ax = fig.add_subplot(2, 2, 2)
    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, sourcediff, "source")
    ax = fig.add_subplot(2, 2, 3)
    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pAfter[0], "pressure")

    plt.show()

    return max_pos_list, max_mag_list


if __name__ == '__main__':
    import time
    from plotScripts import *

    import yaml

    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p4"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"

    # test_case_path = "/home/justinbrusche/test_cases/cylinder"
    test_case_path = "/home/justinbrusche/test_cases/cylinder_0"
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_1"
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_2_deg"

    test_case_path = "/home/justinbrusche/datasets/step4/case_0"

    # test_case_path = "/home/justinbrusche/test_cases/cylinder_fast"


    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"

    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"

    # test_case_path = "/home/justinbrusche/old_test_custom_solver_cases/explicit_small_poissonfoam"
    #
    itab = np.arange(1, 100.5, 1)
    #
    # a

    device = 'cpu'
    # iTab = np.arange(240)
    model = "MeshGraphNetCNN"
    # save_tensors(test_case_path, 16000, getPressure=True,is_cfd_data=None)
    # itab = np.arange(0.00001,0.00150,0.00001)

    # Correctly format the numbers in iTab

    # itab = np.arange(0.5, 60, 0.5)
    itab = [str(format(num, '.5f')).rstrip('0').rstrip('.') for num in itab]
    print(itab)

    save_tensors(test_case_path, 16000, getPressure=True,is_cfd_data=True,itab=itab)
    get_faceArray_incl_boundaries(test_case_path)



    # # # Create a list of formatted strings without trailing zeros
    # itab = [str(format(num, '.5f')).rstrip('0').rstrip('.') for num in itab]
    # #
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_0"
    # max_pos_list = []
    # max_mag_list = []
    # for i in itab:
    #     max_pos_list,max_mag_list = test_data(test_case_path,i,is_cfd_data=True,max_pos_list=max_pos_list,max_mag_list=max_mag_list)
    # plt.plot(max_mag_list,'o',label="alpha = 0 deg")
    # plt.show()
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"


    #
    # test_case_path = "/home/justinbrusche/test_cases/cylinder"
    # max_pos_list = []
    # max_mag_list = []
    # for i in itab:
    #     max_pos_list,max_mag_list = test_data(test_case_path,i,is_cfd_data=True,max_pos_list=max_pos_list,max_mag_list=max_mag_list)
    # plt.plot(max_mag_list,'o',label="alpha = 1 deg")

    # test_case_path = "/home/justinbrusche/test_cases/cylinder_2_deg"
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_1"
    #
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_3_deg"
    #
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_0"



    # max_pos_list = []
    # max_mag_list = []
    # for i in itab:
    #     max_pos_list,max_mag_list = test_data(test_case_path,i,is_cfd_data=True,max_pos_list=max_pos_list,max_mag_list=max_mag_list)
    # plt.plot(max_mag_list,'o',label="alpha = 2 deg")
    # plt.grid()
    # plt.xlabel("sample")
    # plt.ylabel(r"$\Delta P$")
    # plt.legend()
    # plt.title("Maximum pressure difference between two time steps at the domain boundaries")
    #
    # plt.show()
    # print(max_pos_list)
    # a
    # # Output the result
    # print(itab)

    # print(itab)
    # save_tensors(test_case_path, 16000, getPressure=True,is_cfd_data=True,itab=itab)
    # save_tensors(test_case_path, 16000, getPressure=True)

    # save_tensors(test_case_path, 16000, getPressure=True,is_cfd_data=False)

    # save_tensors(test_case_path, 1000, getPressure=True)

    # get_faceArray_incl_boundaries(test_case_path)

    # config_file = '/home/justinbrusche/scripts_FVM_2/trainTestModel/config_files/trainConfig_step_3_var_conv_medium.yaml'
    #
    # #
    # countCellweights = False
    # save_every_model = True
    # device = 'cuda'
    #
    # with open(config_file, 'r') as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)
    #
    # test_case_path = conf['test_case_path']
    #
    # file_path = f"{test_case_path}/dataDictMeshes.pkl"
    # with open(file_path, 'rb') as file:
    #     dataDictMeshes = pickle.load(file)
    #
    # # Load train / test data
    # testDataLoader = loadDataLoader(test_case_path,
    #                                 np.arange(conf['nTrainData'], conf['nTrainData'] + conf['nTestData']),
    #                                 conf['batchSize'], dataDictMeshes,
    #                                 BCInput=conf["modelParam"]["inputChannels"]["BCInput"],
    #                                 gradientBCInput=conf["modelParam"]["inputChannels"]["gradientBCInput"],
    #                                 normalize_data=True,compute_diff_cfd=True)

    # a = EmbedData(test_case_path, device)
    # t1 = time.time()
    # # dArray, faceArray, sourceArray = a.getData(iTab,getPressure=False)
    # print(dArray.shape)
    # # pArray, dArray, faceArray, sourceArray = a.getData(iTab,getPressure=False)
    # print(time.time() - t1)
    #
    # from torch.utils.data import TensorDataset, DataLoader
    #
    # dataset = TensorDataset(dArray, faceArray, sourceArray)
    # print(dataset)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # print(dataloader)
    # config_file = 'config_files/trainConfig_FVMSimpleMedium.yaml'

    # config_file = '/home/justinbrusche/scripts_FVM_2/trainTestModel/config_files/trainConfig_FVMSimpleMedium.yaml'
    # # device = 'cuda'
    # #
    # #
    # with open(config_file, 'r') as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)
    # #
    # test_case_path = conf['test_case_path']
    #
    # file_path = f"{test_case_path}/dataDictMeshes.pkl"
    # with open(file_path, 'rb') as file:
    #     dataDictMeshes = pickle.load(file)

    #
    # # ************************** Load train / test data *********************************
    # trainDataLoader = loadDataLoader(test_case_path, np.arange(conf['nTrainData']), conf['batchSize'],dataDictMeshes)
    # testDataLoader = loadDataLoader(test_case_path,np.arange(conf['nTrainData'], conf['nTrainData'] + conf['nTestData']),conf['batchSize'])
    #
    # for d, face, source in dataloader:
    #     # Your training code here
    #     print(d.shape, face.shape, source.shape)



