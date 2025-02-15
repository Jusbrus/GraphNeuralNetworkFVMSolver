import shutil
import os
import numpy as np
import pickle
import sys
sys.path.append("/home/justinbrusche/scripts_final")
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

from createMeshes import getMeshes
from GNN_pre_scripts.generateMeshPerlinDataset import *
from GNN_pre_scripts.s1_setup_openfoam import SetupOpenfoam
from GNN_pre_scripts.s2_GetEdgeAttributesConv import EdgeAttributes
from GNN_pre_scripts.s2_setup_pool_data_cellCenters_not_used import SetupPoolDataCellCenters
from GNN_pre_scripts.s2_setup_pool_data_meshPoints import SetupPoolDataMeshPoints
from GNN_pre_scripts.s3_embed_graph import EmbedGraph
from trainingDataScripts.openfoam_inputs.Generate_input_data_openfoam import getFieldDirect
from trainingDataScripts.s2_Run_poissonfoam_explicit import set_initial_conditions_direct, modify_control_dict, run_openfoam, setup_source
from trainTestModel.Utilities.embed_data import save_tensors, get_faceArray_incl_boundaries, loadDataLoader

class PerlinNoiseDataset:
    def __init__(self, basePath, datasetName,nSets,nPerCase):
        self.basePath = basePath
        self.datasetName = datasetName
        self.nSets = nSets
        self.nPerCase = nPerCase

        self.datasetPath = os.path.join(self.basePath, self.datasetName)

        print(self.datasetPath)
        print(datasetName)

    def copy_and_rename_map(self, source_path, destination_directory, new_name):
        destination_path = os.path.join(destination_directory, new_name)

        # Check if the destination directory exists
        if os.path.exists(destination_path):
            print(f"Directory '{destination_path}' already exists. Skipping copy.")
        else:
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Copy the source directory to the destination path
            shutil.copytree(source_path, destination_path)
            print(f"Copied from {source_path} to {destination_path}")

    def createDirectoriesAndDatadictMesh(self):
        # Copy and rename the base directory only if it doesn't exist
        self.copy_and_rename_map(os.path.join(self.basePath, "perlinNoiseBase"), self.basePath, self.datasetName)

        containsObject, variable_dict_invariant_list, variable_dict_variant_list, boxListList, objectListList = getMeshes(21)
        mesh_name = "generatedMesh"
        openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"

        for i in range(self.nSets):
            if 1==1:

                if containsObject[i]:
                    self.copy_and_rename_map(os.path.join(self.datasetPath, "baseCaseObject"), self.datasetPath,
                                             f"case_{i}")
                else:
                    self.copy_and_rename_map(os.path.join(self.datasetPath, "baseCaseNoObject"), self.datasetPath,
                                             f"case_{i}")


            casePath = os.path.join(self.datasetPath, f"case_{i}")

            datadictMeshBuilder = SetupOpenfoam(casePath, casePath, mesh_name, 4, 1, faceToCell=True, faceToPoint=True,
                              faceToEdgeCenters=True)

            datadictMeshBuilder.prepareFilesWithObject(openfoam_source_path,variable_dict_variant_list[i],variable_dict_invariant_list[i],objectListList[i],moreObjects=True,box_list=boxListList[i])

            datadictMeshBuilder.getMeshData()

    def embedGraph(self,fileName_embedding,nAttributesDict,device,edgeType,model_group):
        for i in range(self.nSets):
            if 1==1:
                print(i)
                casePath = os.path.join(self.datasetPath, f"case_{i}")

                edgeAttrBuilder = EdgeAttributes(copy.deepcopy(nAttributesDict),casePath)
                edgeAttrBuilder.getData(fileName_embedding, edgeType)

                poolCellCenters = SetupPoolDataCellCenters(casePath)
                poolCellCenters.getPoolData()
                poolCellCenters.getPoolData_direct()

                poolMeshPoints = SetupPoolDataMeshPoints(casePath)
                poolMeshPoints.getPoolData()

                embedder = EmbedGraph(casePath,fileName_embedding,device,model_group,show_pooling=False)
                embedder.getData()

                embedderPool = EmbedGraph(casePath,fileName_embedding,device,model_group,show_pooling=True,direct_pooling=False)
                embedderPool.getData()

                embedderPoolDirect = EmbedGraph(casePath,fileName_embedding,device,model_group,show_pooling=True,direct_pooling=True)
                embedderPoolDirect.getData()

    def getPerlinFields(self):
        openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"
        seeds_dict_path = "/home/justinbrusche/scripts_final/trainingDataScripts/data_input_openfoam/seeds_dict.pkl"
        setup_source(openfoam_source_path)

        def get_Face_centers(path, sides_list_boundaries):
            x_tab = []
            y_tab = []
            # Load the text file into a numpy array
            for i in range(len(sides_list_boundaries)):
                file_path = path + '/constant/polyMesh/faceCenters_' + sides_list_boundaries[i] + '.txt'
                face_centers = np.loadtxt(file_path)
                x_tab.append(face_centers[:, 0])
                y_tab.append(face_centers[:, 1])

            return x_tab, y_tab

        with open(seeds_dict_path, 'rb') as f:
            seeds_dict = pickle.load(f)

        octaves_list = [1, 2, 4, 6, 8, 10]
        persistences = [0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
        lacunarities = [2, 2.2, 2.2, 2.2, 2.2, 3.3]
        scale_range = [0.3, 8]
        center_domain = [0, 0]
        sides_list_boundaries = ["Left", "Upper", "Right", "Lower"]

        i = 0
        for i_case in range(self.nSets):
            if i_case>-1:
                casePath = os.path.join(self.datasetPath, f"case_{i_case}")
                casePathLevel0 = os.path.join(casePath, "level0")
                file_path = f"{casePath}/dataDictMeshes.pkl"
                with open(file_path, 'rb') as file:
                    dataDictMeshes = pickle.load(file)

                x_coords = dataDictMeshes[0]['cellCenters'][:, 0]
                y_coords = dataDictMeshes[0]['cellCenters'][:, 1]
                sides_list = dataDictMeshes[0]["boundaryNames"]

                print(len(x_coords))

                x_tab_face_centers_boundaries, y_tab_face_centers_boundaries = get_Face_centers(casePathLevel0, sides_list_boundaries)
                x_tab_face_centers, y_tab_face_centers = get_Face_centers(casePathLevel0, sides_list)

                for i_field in range(self.nPerCase):
                    if i_case>-1:
                        smoothness = seeds_dict["smoothness"][i]
                        orientation = seeds_dict["orientation"][i]
                        p_seed = seeds_dict["p_seeds"][i]
                        tau_seed = seeds_dict["tau_seeds"][i]
                        p_BC_type = seeds_dict["p_BC_type"][i]
                        # if i_field<101:
                        if 1==1:

                            p_field, p_boundary_dict, tau_field, tau_boundary_dict = getFieldDirect(center_domain,x_coords.copy(), y_coords.copy(), x_tab_face_centers_boundaries.copy(), y_tab_face_centers_boundaries.copy(), x_tab_face_centers.copy(), y_tab_face_centers.copy(), scale_range, octaves_list, persistences, lacunarities,smoothness,orientation,p_seed,tau_seed,p_BC_type,sides_list,sides_list_boundaries)
                            # print(tau_boundary_dict.keys())
                            # print(p_boundary_dict.keys())
                            # print(orientation)
                            # print(p_boundary_dict)
                            # print(x_tab_face_centers_boundaries[0][:5])

                            # fdsafsd
                            set_initial_conditions_direct(i, casePathLevel0, p_field, p_boundary_dict, tau_field, tau_boundary_dict, p_BC_type,
                                                   sides_list, p_bcs_set_to_zero=False, only_diriglet=False, tau_1=False,
                                                   p_bias=True)

                            modify_control_dict(casePathLevel0, i_field)
                            time_elapsed = run_openfoam(casePathLevel0)
                            print(i_case,i_field, time_elapsed)
                            if os.path.exists(os.path.join(casePathLevel0, str(i_field + 1) + "_saved")):
                                shutil.rmtree(os.path.join(casePathLevel0, str(i_field + 1) + "_saved"))
                            os.rename(os.path.join(casePathLevel0, str(i_field + 1)), os.path.join(casePathLevel0, str(i_field + 1) + "_saved"))
                        if i_field==0:
                            import matplotlib.pyplot as plt
                            from trainTestModel.Utilities.plotScripts import plot_mesh_triangles,plot_mesh_triangles_mesh
                            cellPoints = dataDictMeshes[0]["cellPoints"]
                            x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
                            y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]
                            fig = plt.figure(figsize=(20, 10))
                            ax = fig.add_subplot(1, 1, 1)
                            plot_mesh_triangles_mesh(fig, ax, cellPoints, x_mesh, y_mesh, np.zeros(len(cellPoints)),"mesh")
                            fig.savefig(f"/home/justinbrusche/datasetMeshes/mesh_{i_case}.png", dpi=300, bbox_inches='tight')
                            # plt.show()

                        # print(seeds_dict.keys())

            i+=1

    def embedPerlinFields(self):
        for i_case in range(self.nSets):
            print(i_case)
            if i_case>-1:
                casePath = os.path.join(self.datasetPath, f"case_{i_case}")

                save_tensors(casePath, self.nPerCase, getPressure=True, device='cpu', is_cfd_data=False)
                get_faceArray_incl_boundaries(casePath)

        # print(seeds_dict.keys())


if __name__ == '__main__':
    basePath = "/home/justinbrusche/datasets"
    datasetName = "foundationalParameters22"
    edgeType = "FVMModelConv"
    model_group = "FVM"
    device = 'cuda'
    fileName_embedding = "conv_4666"
    nSets = 21
    nPerCase = 1000


    nAttributesDict = {"centerFace": 4,
                       "facePoint": 6,
                       "pointPoint": 6,
                       "pointCenter": 6,
                       "centerPoint": 6,
                       "includeDistance": False}

    builder = PerlinNoiseDataset(basePath,datasetName, nSets,nPerCase)
    builder.createDirectoriesAndDatadictMesh()
    builder.embedGraph(fileName_embedding,nAttributesDict,device,edgeType,model_group)
    # builder.getPerlinFields()
    # builder.embedPerlinFields()
