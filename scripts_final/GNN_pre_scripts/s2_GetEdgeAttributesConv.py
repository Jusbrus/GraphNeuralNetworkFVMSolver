import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt

class EdgeAttributes:
    def __init__(self,nAttributesDict,test_case_path):
        self.nAttributesDict = nAttributesDict
        self.test_case_path = test_case_path
        file_path = f"{test_case_path}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            self.dataDictMeshes = pickle.load(file)

        self.dataDict = {}
        self.dataDict["nAttributesDict"] = nAttributesDict

        for i in range(len(self.dataDictMeshes)):
            self.dataDict[i] = {}

    def _edgeAttributesAngles(self,pos_sources,pos_targets,nWeights):
        delta_distance = pos_sources - pos_targets

        # Get the normalized angle
        angle_norm = (np.arctan2(delta_distance[:, 1], delta_distance[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

        # Get the indices of the nonzero weights
        index_up = np.ceil(angle_norm).astype(int)
        index_down = np.floor(angle_norm).astype(int)

        # Check if the indices are not the same
        check_difference = index_up - index_down
        # Map the last column on the first one
        index_up = index_up % nWeights
        index_down = index_down % nWeights

        # Compute the weights
        weight_up = angle_norm % 1
        weight_down = 1 - weight_up

        # Correct the weights if the indices are the same
        i_correct = np.where(check_difference == 0)[0]
        weight_up[i_correct] = 0.5
        weight_down[i_correct] = 0.5

        EdgeAttributes = np.zeros((len(pos_sources), nWeights))

        arange_len_sources = np.arange(len(pos_sources))

        EdgeAttributes[arange_len_sources, index_down] = weight_down
        EdgeAttributes[arange_len_sources, index_up] = weight_up
        # print(EdgeAttributes.shape)
        return EdgeAttributes


    def get_edgeAttr(self, angles, must_switch_condition, nWeights):
        # print(np.shape(angles))
        edgeAttr = np.zeros((len(angles), nWeights))

        x1_index_up = np.ceil(angles[:,0]).astype(int)%nWeights
        x1_index_down = np.floor(angles[:,0]).astype(int)%nWeights
        x2_index_up = np.ceil(angles[:,1]).astype(int)%nWeights
        x2_index_down = np.floor(angles[:,1]).astype(int)%nWeights

        x1_index_up = np.where(x1_index_up==x1_index_down,x1_index_up+1,x1_index_up)%nWeights
        x2_index_up = np.where(x2_index_up==x2_index_down,x2_index_up+1,x2_index_up)%nWeights

        x1_weight_up = angles[:,0] % 1
        x1_weight_down = 1 - x1_weight_up
        x2_weight_up = angles[:,1] % 1
        x2_weight_down = 1 - x2_weight_up

        for i in range(len(angles)):
            if x1_index_down[i] == x2_index_down[i]:
                dx = angles[i,1] - angles[i,0]
                # print(dx,angles[i])

                edgeAttr[i,x1_index_up[i]] += 0.5 * x1_weight_up[i] * dx
                edgeAttr[i,x1_index_down[i]] += 0.5 * x1_weight_down[i] * dx
                edgeAttr[i,x2_index_up[i]] += 0.5 * x2_weight_up[i] * dx
                edgeAttr[i,x2_index_down[i]] += 0.5 * x2_weight_down[i] * dx

            else:
                # print(edgeAttr[i])

                dx1 = x1_weight_down[i]+0
                dx2 = x2_weight_up[i]+0

                # print(dx1,dx2)

                edgeAttr[i,x1_index_up[i]] += 0.5 * x1_weight_up[i] * dx1 + 0.5 * dx1
                edgeAttr[i,x1_index_down[i]] += 0.5 * x1_weight_down[i] * dx1

                edgeAttr[i,x2_index_up[i]] += 0.5 * x2_weight_up[i] * dx2
                edgeAttr[i,x2_index_down[i]] += 0.5 * x2_weight_down[i] * dx2 + 0.5 *dx2

                if (x2_index_down[i] - x1_index_up[i])>1:
                    # print(edgeAttr[i])

                    edgeAttr[i, x1_index_up[i]] += 0.5
                    edgeAttr[i, x2_index_down[i]] += 0.5

                    full_list = np.arange(x1_index_up[i]+1,x2_index_down[i])
                    edgeAttr[i,full_list] = 1

                elif (x2_index_down[i] - x1_index_up[i])>0:
                    edgeAttr[i, x1_index_up[i]] += 0.5
                    edgeAttr[i, x2_index_down[i]] += 0.5

            if must_switch_condition[i]:
                edgeAttr[i] = 1-edgeAttr[i]

            # print(edgeAttr[i])

        return edgeAttr

    def _centerFaceConvIntegral(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["faceCenters"]["targets"]+0
            targets = self.dataDictMeshes[i]["faceCenters"]["sources"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["cellCenters"][sources]
            pos_targets = self.dataDictMeshes[i]["faceCoordinates_from_cells"][targets]

            delta_distance = pos_sources - pos_targets

            nWeights = self.nAttributesDict["centerFace"]
            # Get the normalized angle
            angle_norm = (np.arctan2(delta_distance[:, 1], delta_distance[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            # print(angle_norm)

            angles = np.vstack(((angle_norm-nWeights/4)%nWeights, (angle_norm+nWeights/4)%nWeights)).transpose()

            must_switch_condition = angles[:, 1] < angles[:, 0]
            angles = np.sort(angles, axis=1)
            edge_attr = self.get_edgeAttr(angles,must_switch_condition, nWeights)
            self.dataDict[i]["centerFace"] = edge_attr/nWeights

            # n_points = np.max(self.dataDictMeshes[i]["faceCenters"]["sources"]) + 1
            # check_tab = np.zeros(n_points)
            # for j in range(len(edge_attr)):
            #     check_tab[targets[j]] += np.sum(self.dataDict[i]["centerFace"][j])
            #
            # plt.plot(check_tab,"o")
            # plt.show()

    def _facePointConvIntegral(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["facePoints"]["sources"]+0
            targets = self.dataDictMeshes[i]["facePoints"]["targets"]+0

            pos_mid = self.dataDictMeshes[i]["pointCoordinates"][targets]
            cellCenters = np.concatenate((self.dataDictMeshes[i]["cellCenters"],[[np.inf,np.inf]]))
            neighboring_cells = self.dataDictMeshes[i]["faceCells"][sources]
            pos_neighboring_cells_1 = cellCenters[neighboring_cells[:, 0]]
            pos_neighboring_cells_2 = cellCenters[neighboring_cells[:, 1]]

            pos_neighboring_cells_2 = np.where(pos_neighboring_cells_2==np.inf,self.dataDictMeshes[i]["faceCoordinates_from_points"][sources],pos_neighboring_cells_2)
            # pos_cell_1 = cellCenters[]

            delta_distance_1 = pos_neighboring_cells_1 - pos_mid
            delta_distance_2 = pos_neighboring_cells_2 - pos_mid

            nWeights = self.nAttributesDict["facePoint"]
            # Get the normalized angle
            angle_norm_1 = (np.arctan2(delta_distance_1[:, 1], delta_distance_1[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            angle_norm_2 = (np.arctan2(delta_distance_2[:, 1], delta_distance_2[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            angles = np.vstack((angle_norm_1, angle_norm_2)).transpose()

            angles = np.sort(angles, axis=1)

            must_switch_condition = (angles[:, 1] - angles[:, 0]) > nWeights/2

            edge_attr = self.get_edgeAttr(angles,must_switch_condition, nWeights)

            self.dataDict[i]["facePoint"] = edge_attr/nWeights



    def _pointPointConvIntegral(self):
        for i in range(len(self.dataDictMeshes.keys())):
            n_points = np.max(self.dataDictMeshes[i]["pointPoints"]["sources"]) + 1
            sources = self.dataDictMeshes[i]["pointPoints"]["sources"][:-n_points]+0
            targets = self.dataDictMeshes[i]["pointPoints"]["targets"][:-n_points]+0
            faces = self.dataDictMeshes[i]["pointPoints"]["faces"]+0

            pos_mid = self.dataDictMeshes[i]["pointCoordinates"][targets]
            cellCenters = np.concatenate((self.dataDictMeshes[i]["cellCenters"],[[np.inf,np.inf]]))
            neighboring_cells = self.dataDictMeshes[i]["faceCells"][faces]
            pos_neighboring_cells_1 = cellCenters[neighboring_cells[:, 0]]
            pos_neighboring_cells_2 = cellCenters[neighboring_cells[:, 1]]

            pos_neighboring_cells_2 = np.where(pos_neighboring_cells_2==np.inf,self.dataDictMeshes[i]["pointCoordinates"][sources],pos_neighboring_cells_2)
            # pos_cell_1 = cellCenters[]

            delta_distance_1 = pos_neighboring_cells_1 - pos_mid
            delta_distance_2 = pos_neighboring_cells_2 - pos_mid

            nWeights = self.nAttributesDict["pointPoint"]
            # Get the normalized angle
            angle_norm_1 = (np.arctan2(delta_distance_1[:, 1], delta_distance_1[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            angle_norm_2 = (np.arctan2(delta_distance_2[:, 1], delta_distance_2[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            angles = np.vstack((angle_norm_1, angle_norm_2)).transpose()
            angles = np.sort(angles, axis=1)

            must_switch_condition = (angles[:, 1] - angles[:, 0]) > nWeights/2

            edge_attr = self.get_edgeAttr(angles,must_switch_condition, nWeights)

            edge_attr = edge_attr/nWeights

            # check_tab = np.zeros(n_points)
            # for j in range(len(edge_attr)):
            #     check_tab[targets[j]] += np.sum(edge_attr[j])
            #
            # plt.plot(check_tab,"o")
            # plt.show()

            edge_attr = np.hstack((edge_attr, np.zeros((len(sources),1))))

            edge_attr_self_loops = np.zeros((n_points,self.nAttributesDict["pointPoint"]+1))

            edge_attr_self_loops[:,-1] = 1/6
            self.dataDict[i]["pointPoint"] = np.concatenate((edge_attr,edge_attr_self_loops),axis=0)

    def _pointCenterConvIntegral(self):
        for i in range(len(self.dataDictMeshes.keys())):
            targets = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            sources = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["pointCoordinates"][sources]
            pos_targets = self.dataDictMeshes[i]["cellCenters"][targets]

            nWeights = self.nAttributesDict["pointCenter"]

            delta_distance = pos_sources - pos_targets

            angle_norm = (np.arctan2(delta_distance[:, 1], delta_distance[:, 0]) * (nWeights) / (
                    2 * np.pi)) % (nWeights)

            angles_cell = np.reshape(angle_norm, (-1, 3))

            sort_indices = np.argsort(angles_cell, axis=1)
            angles_sorted = np.sort(angles_cell, axis=1)
            angles_tiled = np.concatenate((angles_sorted,angles_sorted+nWeights),axis=1)
            angles_1_sorted = (0.5*(angles_tiled[:,3:]+angles_tiled[:,2:5]))%nWeights
            angles_2_sorted = (0.5*(angles_tiled[:,:3]+angles_tiled[:,1:4]))%nWeights

            angles_cell_1 = np.zeros_like(angles_1_sorted)
            angles_cell_2 = np.zeros_like(angles_2_sorted)

            angles_cell_1[targets,sort_indices.flatten()] = angles_1_sorted.flatten()
            angles_cell_2[targets, sort_indices.flatten()] = angles_2_sorted.flatten()

            angles = np.vstack((angles_cell_1.flatten(), angles_cell_2.flatten())).transpose()
            must_switch_condition = angles[:, 1] < angles[:, 0]
            angles = np.sort(angles, axis=1)
            edge_attr = self.get_edgeAttr(angles,must_switch_condition, nWeights)

            self.dataDict[i]["pointCenter"] = edge_attr/nWeights

            n_points = np.max(self.dataDictMeshes[i]["centerPoints"]["sources"]) + 1
            check_tab = np.zeros(n_points)
            for j in range(len(edge_attr)):
                check_tab[targets[j]] += np.sum(self.dataDict[i]["pointCenter"][j])

            # plt.plot(check_tab,"o")
            # plt.show()



    def _centerPointNoDistance(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            targets = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["cellCenters"][sources]
            pos_targets = self.dataDictMeshes[i]["pointCoordinates"][targets]

            self.dataDict[i]["centerPoint"] = self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["centerPoint"]) + 0

    def _centerFace(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["faceCenters"]["targets"]+0
            targets = self.dataDictMeshes[i]["faceCenters"]["sources"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["cellCenters"][sources]
            pos_targets = self.dataDictMeshes[i]["faceCoordinates_from_cells"][targets]
            # print("dssd",sources)

            self.dataDict[i]["centerFace"] = 0.5 * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["centerFace"]) + 0
            # print(self.dataDict[i]["centerFace"])

    def _centerPointType(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            targets = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["cellCenters"][sources]
            pos_targets = self.dataDictMeshes[i]["pointCoordinates"][targets]
            edgeAttr = self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["centerPoint"]) + 0

            extra_attr = self.dataDictMeshes[i]["pointTypes"][targets]

            self.dataDict[i]["centerPoint"] = np.concatenate((edgeAttr, extra_attr.reshape(-1, 1)), axis=1)

    def _centerPointNormConnectedNodes(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            targets = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["cellCenters"][sources]
            pos_targets = self.dataDictMeshes[i]["pointCoordinates"][targets]

            max_value = np.max(targets)
            nodeDegree = np.zeros(max_value + 1, dtype=int)
            # Count occurrences
            unique, counts = np.unique(targets, return_counts=True)
            nodeDegree[unique] = counts
            # print(nodeDegree)
            # print(self.dataDictMeshes[i]["boundaryPoints"])
            nodeDegree[self.dataDictMeshes[i]["boundaryPoints"]] = 6
            # print(nodeDegree)

            edgeDegree = nodeDegree[targets]

            self.dataDict[i]["centerPoint"] = (1/edgeDegree).reshape(-1, 1) * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["centerPoint"]) + 0

    def _facePointNormConnectedNodes(self):
        for i in range(len(self.dataDictMeshes.keys())):
            sources = self.dataDictMeshes[i]["facePoints"]["sources"]+0
            targets = self.dataDictMeshes[i]["facePoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["faceCoordinates_from_points"][sources]
            pos_targets = self.dataDictMeshes[i]["pointCoordinates"][targets]

            max_value = np.max(targets)
            nodeDegree = np.zeros(max_value + 1, dtype=int)
            # Count occurrences
            unique, counts = np.unique(targets, return_counts=True)
            nodeDegree[unique] = counts
            # print(nodeDegree)
            # print(self.dataDictMeshes[i]["boundaryPoints"])
            nodeDegree[self.dataDictMeshes[i]["boundaryPoints"]] = 6
            # print(nodeDegree)

            edgeDegree = nodeDegree[targets]
            # print("fdsfdsfs")
            self.dataDict[i]["facePoint"] = (1/edgeDegree).reshape(-1, 1) * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["facePoint"]) + 0


    def _pointPointNormConnectedNodes(self):
        for i in range(len(self.dataDictMeshes.keys())):
            n_points = np.max(self.dataDictMeshes[i]["pointPoints"]["sources"]) + 1
            sources = self.dataDictMeshes[i]["pointPoints"]["sources"][:-n_points]+0
            targets = self.dataDictMeshes[i]["pointPoints"]["targets"][:-n_points]+0
            # print("gfgfdd")
            # plt.plot(sources)
            # plt.show()
            # print(sources)
            # print(self.dataDictMeshes[i]["pointPoints"]["targets"][-n_points:])

            pos_sources = self.dataDictMeshes[i]["pointCoordinates"][sources]
            pos_targets = self.dataDictMeshes[i]["pointCoordinates"][targets]

            max_value = np.max(targets)
            nodeDegree = np.zeros(max_value + 1, dtype=int)
            # Count occurrences
            unique, counts = np.unique(targets, return_counts=True)
            nodeDegree[unique] = counts + 1
            # print(nodeDegree)
            # print(self.dataDictMeshes[i]["boundaryPoints"])
            nodeDegree[self.dataDictMeshes[i]["boundaryPoints"]] = 7
            # print(nodeDegree)

            edgeDegree = nodeDegree[targets]

            edge_attr = (1/edgeDegree).reshape(-1, 1) * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["pointPoint"]) + 0
            # print(edge_attr)
            # print(np.shape(edge_attr))
            edge_attr = np.hstack((edge_attr, np.zeros((len(sources),1))))

            edge_attr_self_loops = np.zeros((n_points,self.nAttributesDict["pointPoint"]+1))
            edgeDegree_self_loops = nodeDegree[self.dataDictMeshes[i]["pointPoints"]["targets"][-n_points:]]
            # print(1/edgeDegree_self_loops)
            edge_attr_self_loops[:,-1] = 1/edgeDegree_self_loops
            self.dataDict[i]["pointPoint"] = np.concatenate((edge_attr,edge_attr_self_loops),axis=0)
            # print(self.dataDict[i]["pointPoint"])

    def _pointCenterNoDistance(self):
        for i in range(len(self.dataDictMeshes.keys())):
            targets = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            sources = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["pointCoordinates"][sources]
            pos_targets = self.dataDictMeshes[i]["cellCenters"][targets]

            print("fdsfds", targets)

            self.dataDict[i]["pointCenter"] = (1/3) * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["pointCenter"]) + 0
            # print(np.shape(self.dataDict[i]["pointCenter"]))

    def _pointCenterMean(self):
        for i in range(len(self.dataDictMeshes.keys())):
            n_edges = len(self.dataDictMeshes[i]["centerPoints"]["sources"])
            self.dataDict[i]["pointCenter"] = (1/3) * np.ones((n_edges,1))
            # print(np.shape(self.dataDict[i]["pointCenter"]))

    def _pointCenterType(self):
        for i in range(len(self.dataDictMeshes.keys())):
            targets = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            sources = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["pointCoordinates"][sources]
            pos_targets = self.dataDictMeshes[i]["cellCenters"][targets]
            edgeAttr = (1/3) * self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["pointCenter"]) + 0

            extra_attr = self.dataDictMeshes[i]["pointTypes"][sources]

            self.dataDict[i]["pointCenter"] = np.concatenate((edgeAttr, extra_attr.reshape(-1, 1)), axis=1)

    def _pointCenterSlope(self):
        for i in range(len(self.dataDictMeshes.keys())):
            targets = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            sources = self.dataDictMeshes[i]["centerPoints"]["targets"]+0

            posCellPoints = self.dataDictMeshes[i]["pointCoordinates"][self.dataDictMeshes[i]["cellPoints"]]
            deltaDistances = posCellPoints - self.dataDictMeshes[i]["cellCenters"][:, np.newaxis, :]

            ones_column = np.ones((len(sources), 1))/3

            self.dataDict[i]["pointCenter"] = np.hstack((deltaDistances.reshape(len(sources), 2), ones_column))

    def _pointCenterNorm(self):
        for i in range(len(self.dataDictMeshes.keys())):
            targets = self.dataDictMeshes[i]["centerPoints"]["sources"]+0
            sources = self.dataDictMeshes[i]["centerPoints"]["targets"]+0
            # Get coordinates of the nodes
            pos_sources = self.dataDictMeshes[i]["pointCoordinates"][sources]
            pos_targets = self.dataDictMeshes[i]["cellCenters"][targets]

            edgeAttrRaw = self._edgeAttributesAngles(pos_sources,pos_targets,self.nAttributesDict["pointCenter"])
            edgeAttrRawReshaped = edgeAttrRaw.reshape(int(round(len(sources)/3)), 3,3)
            sum = np.sum(edgeAttrRawReshaped, axis=1)
            edgeAttrNorm = edgeAttrRawReshaped/sum[:,np.newaxis,:]
            self.dataDict[i]["pointCenter"] = edgeAttrNorm.reshape((len(sources), 3))


    def centerPointcenter(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNoDistance()
            self._pointCenterNoDistance()

    def centerPointcenterType(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointType()
            self._pointCenterType()
            self.nAttributesDict["centerPoint"] += 1
            self.nAttributesDict["pointCenter"] += 1


    def centerPointCenterNorm(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNoDistance()
            self._pointCenterNorm()

    def centerPointCenterNormConnectedNodes(self):
        print("test")
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNormConnectedNodes()
            self._pointCenterNoDistance()

    def centerPointcenterSlope(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNoDistance()
            self._pointCenterSlope()

    def MeshGraphNetCNN(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNormConnectedNodes()
            self._pointCenterNoDistance()
            # self._pointCenterMean()
            self._pointPointNormConnectedNodes()

            self.nAttributesDict["pointPoint"] += 1

    def MeshGraphNetCNNWithFace(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNormConnectedNodes()
            self._pointCenterNoDistance()
            # self._pointCenterMean()
            self._pointPointNormConnectedNodes()
            self._facePointNormConnectedNodes()

            self.nAttributesDict["pointPoint"] += 1

    def FVMModel(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerFace()
            self._facePointNormConnectedNodes()
            self._pointPointNormConnectedNodes()
            self._pointCenterNoDistance()

            self.nAttributesDict["pointPoint"] += 1

    def FVMModel2(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerPointNoDistance()
            self._facePointNormConnectedNodes()
            self._pointPointNormConnectedNodes()
            self._pointCenterNoDistance()

            self.nAttributesDict["pointPoint"] += 1

    def FVMModelConv(self):
        if self.nAttributesDict["includeDistance"]:
            pass
        else:
            self._centerFaceConvIntegral()
            self._facePointConvIntegral()
            self._pointPointConvIntegral()
            self._pointCenterConvIntegral()

            print(self.nAttributesDict)

            self.nAttributesDict["pointPoint"] += 1


    def getData(self,filename,edgeType):
        #Only FVMModelConv is used during this research.

        if edgeType == "centerPointCenter":
            self.centerPointcenter()
        elif edgeType == "centerPointCenterNorm":
            self.centerPointCenterNorm()
        elif edgeType == "centerPointCenterNormConnectedNodes":
            self.centerPointCenterNormConnectedNodes()
        elif edgeType == "centerPointCenterSlope":
            self.centerPointcenterSlope()
        elif edgeType == "centerPointCenterType":
            self.centerPointcenterType()
        elif edgeType == "MeshGraphNetCNN":
            self.MeshGraphNetCNN()
        elif edgeType == "MeshGraphNetCNNWithFace":
            self.MeshGraphNetCNNWithFace()
        elif edgeType == "FVMModel":
            self.FVMModel()
        elif edgeType == "FVMModel2":
            self.FVMModel2()
        elif edgeType == "FVMModelConv":
            self.FVMModelConv()

        # print(self.dataDict.keys())
        # print(self.dataDict['nAttributesDict'])

        file_path = f"{self.test_case_path}/dataDictEdgeAttr_{filename}.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(self.dataDict, file)


# import sys
# sys.path.append("/home/justinbrusche/scripts_2/GNN_pre_scripts")
# from _GetGraphData import *

if __name__ == '__main__':
    # Define the paths
    test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    #
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p2"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    #
    test_case_path = "/home/justinbrusche/test_cases/cylinder"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"
    #
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"

    # test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"
    #

    # nAttributesDict = {"centerPoint": 6,
    #                    "pointPoint": 6,
    #                    "facePoint": 6,
    #                    "pointCenter": 3,
    #                    "includeDistance": False}

    nAttributesDict = {"centerFace": 4,
                       "facePoint": 6,
                       "pointPoint": 6,
                       "pointCenter": 3,
                       "centerPoint": 6,
                       "includeDistance": False}

    nAttributesDict = {"centerFace": 4,
                       "facePoint": 6,
                       "pointPoint": 6,
                       "pointCenter": 3,
                       "includeDistance": False}

    a = EdgeAttributes(nAttributesDict,test_case_path)
    # model = "slope"
    # model = "test4"
    model = "test"
    # model = "norm"
    model = "normConnectedNodes"
    # model = "type"
    model = "MeshGraphNetCNN"
    model = "MeshGraphNetCNNWithFace"
    model = "FVMModel"
    # model = "FVMModel2"
    model = "FVMModelConv"


    # edgeType = "centerPointCenterSlope"
    edgeType = "centerPointCenter"
    # edgeType = "centerPointCenterType"

    # edgeType = "centerPointCenterNorm"
    edgeType = "centerPointCenterNormConnectedNodes"
    edgeType = "MeshGraphNetCNN"
    edgeType = "MeshGraphNetCNNWithFace"
    edgeType = "FVMModel"
    # edgeType = "FVMModel2"
    edgeType = "FVMModelConv"

    a.getData(model,edgeType)
    angles = np.array([[0,5]])
    a.get_edgeAttr(angles,[False],6)

    # file_path = f"{test_case_path}/dataDictEdgeAttr_slope.pkl"
    # with open(file_path, 'rb') as file:
    #     dataDictEdgeAttr = pickle.load(file)

    # print(dataDictEdgeAttr)





