import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch
import time
from Utilities.embed_data import *

sys.path.append("/home/justinbrusche/scripts_final")
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

from Utilities._loadModel import *
from Utilities.LossFunctionScript import *
from Utilities.plotScripts import *

class testNIterationsMultipleMeshes():
    def __init__(self, basePathDatasets,basePathLinearTest,datasetName,openfoam_source_path,config_file,device,iteration=None):
        self.basePathDatasets = basePathDatasets
        self.basePathLinearTest = basePathLinearTest
        self.datasetName = datasetName
        self.openfoam_source_path = openfoam_source_path
        self.config_file = config_file
        self.device = device
        self.iteration = iteration

        self.datasetPath = os.path.join(self.basePathDatasets, self.datasetName)
        self.linearTestPath = os.path.join(self.basePathLinearTest, self.datasetName)



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

    def load_data(self, i,name):
        p_path = os.path.join(self.path_data, str(i) + "_saved", name)
        with open(p_path, 'r') as file:
            p_data = file.readlines()

        return p_data

    def modify_internalField_zero(self, data, new_value):
        modified_data = []
        inside_internalField = False

        for line in data:
            if "internalField" in line:
                inside_internalField = True
                modified_data.append(f"internalField   {new_value};\n")
            elif inside_internalField and line.strip() == ";":
                inside_internalField = False
            elif not inside_internalField:
                modified_data.append(line)

        return modified_data

    def modify_internalField_model(self, data,p_model):
        modified_data = []
        inside_internalField = False
        almost_inside_internalField = False
        i = 0
        for line in data:
            if "internalField" in line:
                almost_inside_internalField = True
                # modified_data.append(f"internalField   {new_value};\n")
            if "(" in line and almost_inside_internalField == True:
                inside_internalField = True
                modified_data.append(line)
            elif inside_internalField and ")" in line:
                modified_data.append(line)
                inside_internalField = False
                almost_inside_internalField = False
            elif not inside_internalField:
                modified_data.append(line)
            elif inside_internalField:
                modified_data.append(f"{p_model[0,i,0]}\n")
                i+=1

        return modified_data

    def setup_test_conditions(self, i,p_model):
        p_data = self.load_data(i,"p")
        files_to_create = ["p_model_GAMG", "p_model_jacobi", "p_zero_GAMG", "p_zero_jacobi"]

        temp_dir = os.path.join(self.path_model, "0")
        os.makedirs(temp_dir, exist_ok=True)

        for file_name in files_to_create:
            # print(file_name)

            if "zero" in file_name:
                new_p_data = self.modify_internalField_zero(p_data, "uniform 0")
            else:
                new_p_data = self.modify_internalField_model(p_data, p_model)

            final_file_path = os.path.join(temp_dir, file_name)
            with open(final_file_path, 'w') as file:
                file.writelines(new_p_data)

            # print(f"File {file_name} written successfully.")

        source = self.load_data(i,"source_fvc")
        final_file_path = os.path.join(temp_dir, "source")
        with open(final_file_path, 'w') as file:
            file.writelines(source)

        tau = self.load_data(i,"tau")
        final_file_path = os.path.join(temp_dir, "tau")
        with open(final_file_path, 'w') as file:
            file.writelines(tau)

    def load_model(self,i_first_case):
        with open(config_file, 'r') as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

        sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

        # self.test_case_path = self.conf['test_case_path']

        file_path = f"{self.datasetPath}/case_{i_first_case}/embedding_{self.conf['edgeAttrName']}"
        with open(file_path, 'rb') as file:
            GNNdataDict_first = pickle.load(file)

        print(GNNdataDict_first["GNNData"])

        self.model = loadModel(self.conf, GNNdataDict_first["GNNData"], self.device)
        if self.iteration == None:
            print("iteration == None")
            model_path = os.path.join(self.conf["modelDir"], f'{self.conf["modelFilename"]}.pth')
        else:
            print("iteration = ", self.iteration)
            model_path = os.path.join(self.conf["modelDir"], f'{self.conf["modelFilename"]}_{str(self.iteration)}.pth')
        if os.path.exists(model_path):

            print("Pre-trained model found. Loading model.")
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])


    def extract_iterations_and_times(self, poissonFoam_output):

        iteration_patterns = {
            'p_zero_jacobi': r'DICPCG:\s+Solving for p_zero_jacobi,.*No Iterations (\d+)',
            'p_model_jacobi': r'DICPCG:\s+Solving for p_model_jacobi,.*No Iterations (\d+)',
            'p_zero_GAMG': r'GAMG:\s+Solving for p_zero_GAMG,.*No Iterations (\d+)',
            'p_model_GAMG': r'GAMG:\s+Solving for p_model_GAMG,.*No Iterations (\d+)'
        }

        # iteration_patterns = {
        #     'p_zero_jacobi': r'nonePCG:\s+Solving for p_zero_jacobi,.*No Iterations (\d+)',
        #     'p_model_jacobi': r'nonePCG:\s+Solving for p_model_jacobi,.*No Iterations (\d+)',
        #     'p_zero_GAMG': r'GAMG:\s+Solving for p_zero_GAMG,.*No Iterations (\d+)',
        #     'p_model_GAMG': r'GAMG:\s+Solving for p_model_GAMG,.*No Iterations (\d+)'
        # }

        time_patterns = {
            'p_zero_jacobi': r'Time taken to solve pEqn_zero_jacobi: ([\d.]+) seconds',
            'p_model_jacobi': r'Time taken to solve pEqn_model_jacobi: ([\d.]+) seconds',
            'p_zero_GAMG': r'Time taken to solve pEqn_zero_GAMG: ([\d.]+) seconds',
            'p_model_GAMG': r'Time taken to solve pEqn_model_GAMG: ([\d.]+) seconds'
        }

        iterations = {}
        times = {}

        for key, pattern in iteration_patterns.items():
            match = re.search(pattern, poissonFoam_output)
            if match:
                iterations[key] = int(match.group(1))
            else:
                raise ValueError(f"Expected iteration key {key} not found in output.")

        for key, pattern in time_patterns.items():
            match = re.search(pattern, poissonFoam_output)
            if match:
                times[key] = float(match.group(1))
            else:
                raise ValueError(f"Expected time key {key} not found in output.")

        return iterations, times

    def run_openfoam(self):
        # Run the poissonFoam solver directly using subprocess
        # test_case_path = os.path.join(self.path_model, "0")
        poissonFoam_command = f"poissonFoamLinearSolverTest -case {self.path_model}"
        poissonFoam_result = subprocess.run(poissonFoam_command, shell=True, capture_output=True, text=True,
                                            executable="/bin/bash")
        # print(time.time() - t1)

        # Print the results of the poissonFoam run
        if poissonFoam_result.returncode == 0:
            # print("Solver run was successful.")
            print(poissonFoam_result.stdout)
            results, times = self.extract_iterations_and_times(poissonFoam_result.stdout)
            # print(results)

            return results, times

        else:
            print("Solver run failed.")
            print(poissonFoam_result.stderr)

            return None

    def read_file_lines(self, filepath, start, end):
        with open(filepath, 'r') as f:
            lines = f.readlines()[start:end]
        return np.fromstring(''.join(lines), sep='\n')

    def runOneCase(self,i_case, i_start, i_end, results_dict, normalize_data=True):

        self.path_model = os.path.join(self.linearTestPath, f"case_{i_case}")
        self.path_data = os.path.join(self.datasetPath, f"case_{i_case}/level0")

        file_path = f"/home/justinbrusche/scripts_final/trainingDataScripts/data_input_openfoam/seeds_dict.pkl"
        with open(file_path, 'rb') as file:
            seeds_dict = pickle.load(file)

        octaves_list = [1, 2, 4, 6, 8, 10]

        i_seeds_base = i_case*1000


        file_path = f"{self.datasetPath}/case_{i_case}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshes = pickle.load(file)

        file_path = f"{self.datasetPath}/case_{i_case}/embedding_{self.conf['edgeAttrName']}"
        with open(file_path, 'rb') as file:
            GNNdataDict = pickle.load(file)

        lossFunction = LossClass(os.path.join(self.datasetPath, f"case_{i_case}"), self.device)

        testDataLoader = loadDataLoader(f"{self.datasetPath}/case_{i_case}",
                                        np.arange(i_start, i_end),
                                        1, dataDictMeshes,
                                        BCInput=self.conf["modelParam"]["inputChannels"]["BCInput"],
                                        gradientBCInput=self.conf["modelParam"]["inputChannels"]["gradientBCInput"],doShuffling=False,normalize_data=normalize_data)

        cellPoints = dataDictMeshes[0]["cellPoints"]
        x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
        y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]

        x_coords = dataDictMeshes[0]["cellCenters"][:, 0]
        y_coords = dataDictMeshes[0]["cellCenters"][:, 1]

        results_dict_sub = {
            "p_zero_jacobi_iterations": [],
            "p_model_jacobi_iterations": [],
            "p_zero_GAMG_iterations": [],
            "p_model_GAMG_iterations": [],
            "p_zero_jacobi_times": [],
            "p_model_jacobi_times": [],
            "p_zero_GAMG_times": [],
            "p_model_GAMG_times": [],
            "rme": [],
            "t_model": []
        }

        with torch.no_grad():
            time_list = []
            # for i in range(i_start, i_end + 1):
            i = i_start+1
            for p, pEqnD, pEqnFace, pEqnSource, inputCellArray, faceArray_incl_boundaries, BCArray, *factors in testDataLoader:
                i_seed = i_seeds_base + i - 1

                t1 = time.time()

                if self.conf["modelParam"]["inputChannels"]["BCInput"] == True:

                    if self.conf["modelParam"]["inputChannels"]["gradientBCInput"] == True:
                        inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 4))
                        inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                    else:
                        inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 3))
                        inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                    inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)
                else:
                    inputFaceArray = faceArray_incl_boundaries + 0

                torch.cuda.synchronize()

                t1 = time.time()

                outP = self.model(inputCellArray.to(device), inputFaceArray.to(device), GNNdataDict["graphData"],
                                 GNNdataDict["graphPoolData"],
                                 GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"])

                torch.cuda.synchronize()
                t_process = time.time() - t1

                lossPressure = lossFunction.computeLossPressure(outP, p.to(device))
                # print("pressure loss: ", lossPressure.item())

                #outP = outP + 0.9*(p.unsqueeze(-1).to(device) - outP)
                # outP = p.unsqueeze(-1).to(device)

                # outP = outP * 2/3
                if normalize_data:
                    outP = outP.to("cpu") * factors[3] * factors[2] * 1

                # p = p.unsqueeze(-1)
                # lossFunction.computeLossSource(outP, p.unsqueeze(-1), pEqnD, pEqnFace, pEqnSource)

                # print(outP[0,0,0])
                time_list.append(time.time() - t1)
                # print(time_list[-1])
                self.setup_test_conditions(i, outP.to("cpu"))
                # self.setup_test_conditions(i, p.unsqueeze(-1).to("cpu"))

                iterations, times = self.run_openfoam()
                plt.show()

                print(iterations,i)

                if 1 == 2:
                    # if torch.max(abs(p[0]))<10:

                    # fig = plt.figure(figsize=(24, 13))
                    # ax = fig.add_subplot(1, 1, 1)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "pressure")
                    # plt.tight_layout()
                    # # fig = plt.figure(figsize=(24, 13))
                    # # ax = fig.add_subplot(1, 1, 1)
                    # # plot_mesh_triangles_mesh(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "Mesh")
                    # # plt.tight_layout()
                    # plt.show()

                    zmin = min(np.min(p[0].numpy()), np.min(outP[0, :, 0].cpu().numpy()),
                               np.min(p[0].numpy() - outP[0, :, 0].cpu().numpy()))
                    zmax = max(np.max(p[0].numpy()), np.max(outP[0, :, 0].cpu().numpy()),
                               np.max(p[0].numpy() - outP[0, :, 0].cpu().numpy()))
                    zlim = [zmin, zmax]

                    fig = plt.figure(figsize=(15, 10))

                    ax = fig.add_subplot(2, 2, 1)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "Ground truth",zlim=zlim)
                    # plot_2d(fig, ax, x_coords, y_coords, p[0], "Ground truth",zlim=zlim)
                    plt.xlim([0,5.5])
                    plt.ylim([0,4])
                    ax = fig.add_subplot(2, 2, 2)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0, :, 0].cpu(), "Prediction",zlim=zlim)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0], "Prediction",zlim=zlim)
                    plt.xlim([0,5.5])
                    plt.ylim([0,4])
                    ax = fig.add_subplot(2, 2, 3)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0, :, 0].cpu() - p[0], "Diff",zlim=zlim,centered_colorbar=True)
                    plt.xlim([0,5.5])
                    plt.ylim([0,4])
                    ax = fig.add_subplot(2, 2, 4)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0], "Source term",centered_colorbar=True)
                    # plot_2d(fig, ax, x_coords, y_coords, pEqnSource[0], "Source term",zlim=zlim)
                    plt.xlim([0,5.5])
                    plt.ylim([0,4])
                    plt.tight_layout()

                    # fig, ax = plt.subplots(figsize=(8, 6))
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "Pressure field",zlim=zlim)
                    # plt.tight_layout()

                    plt.show()


                try:

                    results_dict_sub["p_zero_jacobi_iterations"].append(iterations["p_zero_jacobi"])
                    results_dict_sub["p_model_jacobi_iterations"].append(iterations["p_model_jacobi"])
                    results_dict_sub["p_zero_GAMG_iterations"].append(iterations["p_zero_GAMG"])
                    results_dict_sub["p_model_GAMG_iterations"].append(iterations["p_model_GAMG"])

                    results_dict_sub["p_zero_jacobi_times"].append(times["p_zero_jacobi"])
                    results_dict_sub["p_model_jacobi_times"].append(times["p_model_jacobi"])
                    results_dict_sub["p_zero_GAMG_times"].append(times["p_zero_GAMG"])
                    results_dict_sub["p_model_GAMG_times"].append(times["p_model_GAMG"])
                    results_dict_sub["t_model"].append(t_process)

                    results_dict_sub["rme"].append(lossPressure.item())
                except:
                    print("code does not work")

                i += 1

            if len(time_list) > 0:
                print("average time: for model: ",np.mean(time_list[1:]))

        try:

            results_dict["p_zero_jacobi_iterations"].append(results_dict_sub["p_zero_jacobi_iterations"])
            results_dict["p_model_jacobi_iterations"].append(results_dict_sub["p_model_jacobi_iterations"])
            results_dict["p_zero_GAMG_iterations"].append(results_dict_sub["p_zero_GAMG_iterations"])
            results_dict["p_model_GAMG_iterations"].append(results_dict_sub["p_model_GAMG_iterations"])

            results_dict["p_zero_jacobi_times"].append(results_dict_sub["p_zero_jacobi_times"])
            results_dict["p_model_jacobi_times"].append(results_dict_sub["p_model_jacobi_times"])
            results_dict["p_zero_GAMG_times"].append(results_dict_sub["p_zero_GAMG_times"])
            results_dict["p_model_GAMG_times"].append(results_dict_sub["p_model_GAMG_times"])

            results_dict["rme"].append(results_dict_sub["rme"])
            results_dict["t_model"].append(results_dict_sub["t_model"])

        except:
            print("code does not work")

        return results_dict

    def run(self,test_cases, i_start, i_end,main_modeldir,save_name,conf):
        self.load_model(test_cases[0])
        self.model.eval()
        self.setup_source()

        results_dict = {
            "p_zero_jacobi_iterations": [],
            "p_model_jacobi_iterations": [],
            "p_zero_GAMG_iterations": [],
            "p_model_GAMG_iterations": [],
            "p_zero_jacobi_times": [],
            "p_model_jacobi_times": [],
            "p_zero_GAMG_times": [],
            "p_model_GAMG_times": [],
            "rme": [],
            "t_model": []

        }

        for i_case in test_cases:
            results_dict = self.runOneCase(i_case, i_start, i_end, results_dict)

        save_path = os.path.join(main_modeldir, save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        path = os.path.join(save_path, f"conv_{conf}")

        with open(path, 'wb') as f:
            pickle.dump(results_dict, f)

        print(f"-------------RESULTS {self.config_file}, iteration: {self.iteration}--------------------")
        print("RME MEAN: ",np.mean(results_dict["rme"]))
        print("RME MEAN PER CASE: ", np.mean(results_dict["rme"],axis=1))
        iterations_PCG = np.array(results_dict["p_zero_jacobi_iterations"]) - np.array(results_dict["p_model_jacobi_iterations"])
        iterations_GAMG = np.array(results_dict["p_zero_GAMG_iterations"]) - np.array(results_dict["p_model_GAMG_iterations"])
        print("")
        print("PCG ITERATIONS SAVED: ", np.mean(iterations_PCG))
        print("PCG ITERATIONS SAVED PER CASE: ", np.mean(iterations_PCG,axis=1))
        print("")
        print("GAMG ITERATIONS SAVED: ", np.mean(iterations_GAMG))
        print("GAMG ITERATIONS SAVED PER CASE: ", np.mean(iterations_GAMG,axis=1))
        print("")
        ratios_PCG = np.array(results_dict["p_model_jacobi_iterations"]) / np.array(results_dict["p_zero_jacobi_iterations"])
        ratios_GAMG = np.array(results_dict["p_model_GAMG_iterations"]) / np.array(results_dict["p_zero_GAMG_iterations"])
        print("PCG RATIOS: ", np.mean(ratios_PCG))
        print("PCG RATIOS PER CASE: ", np.mean(ratios_PCG,axis=1))
        print("")
        print("GAMG RATIOS: ", np.mean(ratios_GAMG))
        print("GAMG RATIOS PER CASE: ", np.mean(ratios_GAMG,axis=1))
        print("")
        a = np.array(results_dict["p_zero_jacobi_iterations"]).flatten()
        b = np.array(results_dict["p_model_jacobi_iterations"]).flatten()

        plt.figure(figsize=(12,12))
        plt.plot(a,a-b,'o')
        plt.grid()
        plt.figure(figsize=(12,12))
        plt.plot(a,(a-b)/a,'o')
        plt.grid()
        plt.figure(figsize=(12,12))
        plt.plot(a,b/a,'o')
        plt.grid()

        plt.show()

        return results_dict


if __name__ == '__main__':
    # Define dataset to evaluate
    basePathDatasets = "/home/justinbrusche/datasets"
    datasetName = "foundationalParameters"

    ### Do not touch
    basePathLinearTest = "/home/justinbrusche/linearSolverTests"
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"
    device = 'cuda'
    ###

    # Define model
    conf = "4666_e3_100"
    config_file = f'config_files/foundParam_conv_{conf}_small.yaml'

    # define which cases must be evaluated
    test_cases = [11,12]

    #Define which training epoch model must be tested (only even numbers)
    iteration = 200

    # Set sample interval
    i_start = 0
    i_end =50

    # Save directory
    main_modeldir = f'/home/justinbrusche/modeldirs_FVM/solverResults'
    save_name = "e3_200"

    tester = testNIterationsMultipleMeshes(basePathDatasets,basePathLinearTest,datasetName,openfoam_source_path,config_file,device,iteration=iteration)
    tester.run(test_cases,i_start, i_end,main_modeldir,save_name,conf)






