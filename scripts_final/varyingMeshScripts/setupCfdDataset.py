import copy
import shutil
import os
import numpy as np
import pickle
import subprocess
import time
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import sys

sys.path.append("/home/justinbrusche/scripts_final")
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

from createMeshes import getMeshes
from GNN_pre_scripts.generateMeshCFDDataset import *
from GNN_pre_scripts.s1_setup_openfoam import SetupOpenfoam
from GNN_pre_scripts.s2_GetEdgeAttributesConv import EdgeAttributes
from GNN_pre_scripts.s2_setup_pool_data_cellCenters_not_used import SetupPoolDataCellCenters
from GNN_pre_scripts.s2_setup_pool_data_meshPoints import SetupPoolDataMeshPoints
from GNN_pre_scripts.s3_embed_graph import EmbedGraph
from trainingDataScripts.openfoam_inputs.Generate_input_data_openfoam import getFieldDirect
from trainingDataScripts.s2_Run_poissonfoam_explicit import set_initial_conditions_direct, modify_control_dict, run_openfoam, setup_source
from trainTestModel.Utilities.embed_data import save_tensors, get_faceArray_incl_boundaries, loadDataLoader

sys.path.append("/home/justinbrusche/scripts_final")
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

class cfdDataset:
    def __init__(self, basePath, datasetName,openfoam_source_path,nSets):
        self.basePath = basePath
        self.datasetName = datasetName
        self.nSets = nSets
        self.openfoam_source_path = openfoam_source_path

        self.datasetPath = os.path.join(self.basePath, self.datasetName)

        print(self.datasetPath)
        print(datasetName)

    def setup_source(self):
        source_commands = f"""
        cd /
        source {self.openfoam_source_path}/etc/bashrc
        env
        """

        result = subprocess.run(source_commands, shell=True, capture_output=True, text=True, executable="/bin/bash")

        if result.returncode == 0:
            print("Environment sourced successfully.")
            env_vars = result.stdout.splitlines()
            env_dict = dict(line.split("=", 1) for line in env_vars if "=" in line)
        else:
            print("Sourcing environment failed.")
            print(result.stderr)
            exit(1)

        os.environ.update(env_dict)

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

    def createDirectoriesAndDatadictMesh(self,Re_tab,writeInterval,endTime,y_plus_target=1,use_ref_Re=False,object="circle",AoA_tab=[0]):
        # Copy and rename the base directory only if it doesn't exist
        self.copy_and_rename_map(os.path.join(self.basePath, "cfdBase"), self.basePath, self.datasetName)

        variable_dict_invariant_list, variable_dict_variant_list, objectList = self.getCfdMeshInputs(Re_tab,y_plus_target=y_plus_target,use_ref_Re=use_ref_Re,object=object,AoA_tab=AoA_tab)
        mesh_name = "generatedMesh"
        openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"

        for i in range(self.nSets):
            if Re_tab[i]>5000:
                self.copy_and_rename_map(os.path.join(self.datasetPath, "baseCaseHighRe"), self.datasetPath,
                                         f"case_{i}")
            else:
                self.copy_and_rename_map(os.path.join(self.datasetPath, "baseCaseLowRe"), self.datasetPath,
                                         f"case_{i}")

            casePath = os.path.join(self.datasetPath, f"case_{i}")

            datadictMeshBuilder = SetupOpenfoam(casePath, casePath, mesh_name, 4, 1, faceToCell=True, faceToPoint=True,
                              faceToEdgeCenters=True)

            datadictMeshBuilder.prepareFilesWithObject(openfoam_source_path,variable_dict_variant_list[i],variable_dict_invariant_list[i],[objectList[i]],moreObjects=False)

            datadictMeshBuilder.getMeshData()

            nu_target = 1/Re_tab[i]
            self.modify_nu_value(os.path.join(self.datasetPath,f"case_{i}/level0/constant/transportProperties"),nu_target)
            self.modify_write_interval(os.path.join(self.datasetPath,f"case_{i}/level0/system/controlDict"),writeInterval)
            self.modifyendTime(os.path.join(self.datasetPath,f"case_{i}/level0/system/controlDict"),endTime)

    def modify_nu_value(self,file_path, new_nu_value):
        # Open the file for reading
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Look for the line containing 'nu' and modify its value
        for i, line in enumerate(lines):
            if line.strip().startswith('nu'):
                # Replace the existing value of nu with the new one
                lines[i] = f'nu              {new_nu_value};\n'
                break

        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"The value of 'nu' has been changed to {new_nu_value}")

    def modifyendTime(self,file_path,new_endTime):
        # Open the file for reading
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Look for the line containing 'writeInterval' and modify its value
        for i, line in enumerate(lines):
            if line.strip().startswith('endTime'):
                # Replace the existing value of writeInterval with the new one
                lines[i] = f'endTime         {new_endTime};\n'
                break

        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"The value of 'endTime' has been changed to {new_endTime}")

    def modify_write_interval(self,file_path, new_write_interval):
        # Open the file for reading
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Look for the line containing 'writeInterval' and modify its value
        for i, line in enumerate(lines):
            if line.strip().startswith('writeInterval'):
                # Replace the existing value of writeInterval with the new one
                lines[i] = f'writeInterval   {new_write_interval};\n'
                break

        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"The value of 'writeInterval' has been changed to {new_write_interval}")

    def embedGraph(self,fileName_embedding,nAttributesDict,device,edgeType,model_group):
        for i in range(self.nSets):
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

            embedderPool = EmbedGraph(casePath,fileName_embedding,device,model_group,show_pooling=True,direct_pooling=True)
            embedderPool.getData()

    def runCFDs(self):
        setup_source(self.openfoam_source_path)

        for i_case in range(self.nSets):
            print(f"RUN CASE {i_case}")
            casePath = os.path.join(self.datasetPath, f"case_{i_case}")
            casePathLevel0 = os.path.join(casePath, "level0")
            self.run_openfoam(casePathLevel0)

    def run_openfoam(self,casePathLevel0):
        # Run the poissonFoam solver directly using subprocess
        poissonFoam_command = f"pimpleFoam_save_matrices -case {casePathLevel0}"
        t1 = time.time()
        poissonFoam_result = subprocess.run(poissonFoam_command, shell=True, capture_output=True, text=True,
                                            executable="/bin/bash")
        # print(time.time() - t1)
        time_elapsed = time.time() - t1

        # Print the results of the poissonFoam run
        if poissonFoam_result.returncode == 0:
            # print("Solver run was successful.")
            # print(poissonFoam_result.stdout)
            pass
        else:
            print("Solver run failed.")
            print(poissonFoam_result.stderr)

        return time_elapsed


    def embedCfdFields(self,writeInterval,endTime):
        itab = np.arange(writeInterval,endTime+0.5*writeInterval,writeInterval)
        print(itab)
        print(len(itab))

        itab = [str(format(num, '.5f')).rstrip('0').rstrip('.') for num in itab]

        for i_case in range(self.nSets):
            if i_case in [5,15]:
                print(i_case)
                casePath = os.path.join(self.datasetPath, f"case_{i_case}")

                save_tensors(casePath, 1, getPressure=True, device='cpu', is_cfd_data=True, itab=itab)
                get_faceArray_incl_boundaries(casePath)

    def getCfdMeshInputs(self,Re_tab,use_ref_Re=False,ref_Re=1000,y_plus_target=1,object="circle",AoA_tab=[0]):

        def get_ncyl(d, v, Re, y_plus_target):
            if Re>5000:
                nu = (d * v) / Re
                Cf = (2 * np.log10(Re) - 0.65) ** (-2.3)
                u_T = v * np.sqrt(Cf / 2)

                l_viscous = nu / u_T

                ncyl = np.pi * d / l_viscous / y_plus_target
                print(ncyl)
            else:
                boundary_thickness = (5*d)/np.sqrt(Re)
                t_cell = boundary_thickness/20

                ncyl = (np.pi * d) / t_cell
                print("Low Re, ",ncyl)

            # ncyl = ncyl * 2.1 / np.pi
            return int(np.ceil(ncyl))

        variable_dict_variant_base = {"nx": 32,
                                 "ny": 32,
                                 "ncyl": 200,
                                 "nx_box": 200,
                                 "ny_box": 25,
                                 }

        variable_dict_invariant_base = {
            "y": 16,
            "x": 28,
        }
        if object == "circle":
            object_base = {"type":"circle", "x":8, "y": 8, "r": 0.5, "n":"ncyl"}
        else:
            object_base = {"type":"airfoil","naca": object, "x":8, "y": 8, "c": 1,"alpha":20, "n":"ncyl","box": True}

        variable_dict_invariant_list = []
        variable_dict_variant_list = []
        objectList = []

        for i in range(self.nSets):
            if use_ref_Re:
                ncylTarget = get_ncyl(1,1,ref_Re,y_plus_target)
            else:
                ncylTarget = get_ncyl(1,1,Re_tab[i],y_plus_target)

            factor = ncylTarget/variable_dict_variant_base["ncyl"]

            variable_dict_variant = copy.deepcopy(variable_dict_variant_base)

            variable_dict_variant["ncyl"] = int(round(variable_dict_variant["ncyl"] * factor))

            # for key in  variable_dict_variant.keys():
            #     if key
            #     if key == "nx" or key == "ny":
            #         pass
            #     else:
            #         if key == "nx_box" or key == "ny_box":
            #             if factor > 1:
            #                 variable_dict_variant[key] = int(round(variable_dict_variant[key] * factor))
            #
            #         else:
            #             variable_dict_variant[key] = int(round(variable_dict_variant[key] * factor))

            variable_dict_variant_list.append(variable_dict_variant)
            variable_dict_invariant_list.append(copy.deepcopy(variable_dict_invariant_base))

            if object != "circle":
                object_base["alpha"] = AoA_tab[i]

            objectList.append(copy.deepcopy(object_base))

        return variable_dict_invariant_list, variable_dict_variant_list, objectList

if __name__ == '__main__':
    basePath = "/home/justinbrusche/datasets"
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"
    edgeType = "FVMModelConv"
    model_group = "FVM"
    device = 'cuda'
    fileName_embedding = "conv_4666"

    #Define dataset name
    step = 4
    datasetName = f"step{step}"

    if step != 4:
        endTime = 100
    else:
        endTime = 40

    if step == 1:
        nSets = 1
        nPerCase = 20000

    else:
        nSets = 22
        nPerCase = 1000

    writeInterval = endTime/nPerCase

    #Define Reynolds numbers
    Re_tab = np.arange(250, 1325, 50)
    nSets = len(Re_tab)

    #Define Object
    object = "circle"
    object = "2412"

    #Define AoA
    AoA_tab = np.arange(-5,17,1)



    nAttributesDict = {"centerFace": int(fileName_embedding[5]),
                       "facePoint": int(fileName_embedding[6]),
                       "pointPoint": int(fileName_embedding[7]),
                       "pointCenter": int(fileName_embedding[8]),
                       "centerPoint": 3,
                       "includeDistance": False}

    builder = cfdDataset(basePath,datasetName, openfoam_source_path,nSets)
    builder.createDirectoriesAndDatadictMesh(Re_tab,writeInterval,endTime,y_plus_target=1,use_ref_Re=False,object=object,AoA_tab=AoA_tab)
    builder.embedGraph(fileName_embedding,nAttributesDict,device,edgeType,model_group)
    builder.runCFDs()
    builder.embedCfdFields(writeInterval,endTime)
