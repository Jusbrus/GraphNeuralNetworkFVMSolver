import os
import subprocess
import shutil
import yaml
import pickle
import torch
import sys
import re
import time

from Utilities.embed_data import *
from Utilities.LossFunctionScript import *
# from Utilities.plotScripts_old import *
from Utilities._loadModelTestSpeed import *
from Utilities.plotScripts import *


def get_edgeAttrPrep(model,GNNdataDict):
    weights_dict = {}

    level_dict = {"convN_1":0,
                  "convN_2":1,
                  "convN_3":2,
                  "convN_4":3,
                  "convN_7":2,
                  "convN_8":1,
                  "convN_9":0}

    print(GNNdataDict["graphData"][0].keys())
    print(GNNdataDict["graphData"][0].values())
    print(GNNdataDict["GNNDataForward"][0]['facePoint'])

    aggrTypeLink = {"layerCenterFace": "centerFace",
                    "layerFacePoint": "facePoint"}

    # Populate the dictionary with model parameters (only weights)
    for name, param in model.named_parameters():
        # print(name)
        if 'weight_matrix' in name:
            keys = name.split('.')
            d = weights_dict
            for key in keys[:-3]:
                if key not in d:
                    d[key] = {}
                d = d[key]

            # print(keys)
            level = level_dict.get(keys[0], None)
            # print(level)

            if keys[2] == "pointPoint":
                aggrType = "pointPoint"
            elif keys[2] == "pointCenter":
                if keys[0] == 'convN_9':
                    aggrType = "pointCenter"
                else:
                    aggrType = aggrTypeLink[keys[1]]

            tensor = GNNdataDict["graphData"][level][aggrType].edge_attr

            d[keys[-3]] = torch.einsum('ij,jkl->ikl', tensor, param.data).contiguous()

    return weights_dict


class testNIterationsMultipleMeshesCFD:
    def __init__(self, basePathDatasets,basePathLinearTest,datasetName,openfoam_source_path,config_file_field,config_file_mag,device,iteration=None):
        self.basePathDatasets = basePathDatasets
        self.basePathLinearTest = basePathLinearTest
        self.datasetName = datasetName
        self.openfoam_source_path = openfoam_source_path
        self.config_file_field = config_file_field
        self.config_file_mag = config_file_mag
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

    def copy_and_rename(self, source_path, destination_directory, new_name):
        destination_path = os.path.join(destination_directory, new_name)

        # Copy the source file to the destination path
        shutil.copy2(source_path, destination_path)


    def load_data(self, t,name):
        # files = os.listdir(os.path.join(self.path_data, str(t)))
        # for file in files:
        #     print(file)

        p_path = os.path.join(self.path_data, str(t), name)
        with open(p_path, 'r') as file:
            p_data = file.readlines()

        return p_data

    def getModelFile(self,p_data,p_guess):
        guess_list = [f'{val}\n' for val in p_guess.numpy().flatten()]
        index_start_data = p_data.index("(\n")+1
        p_data[index_start_data:index_start_data+len(guess_list)] = guess_list

        return p_data

    def setup_test_conditions(self, t,p_model):
        p_data = self.load_data(t,"pAfter")
        source_dir = os.path.join(self.path_data, str(t))
        target_dir = os.path.join(self.path_model, "0")


        self.copy_and_rename(os.path.join(source_dir, "tau"), target_dir, "tau")
        self.copy_and_rename(os.path.join(source_dir, "source_fvc"), target_dir, "source")
        self.copy_and_rename(os.path.join(source_dir, "pEqnSource"), target_dir, "pEqnSource")
        self.copy_and_rename(os.path.join(source_dir, "pEqnFace"), target_dir, "pEqnFace")
        self.copy_and_rename(os.path.join(source_dir, "pEqnD"), target_dir, "pEqnD")
        self.copy_and_rename(os.path.join(source_dir, "pInitial"), target_dir, "p_zero_GAMG")
        self.copy_and_rename(os.path.join(source_dir, "pInitial"), target_dir, "p_zero_jacobi")
        self.copy_and_rename(os.path.join(source_dir, "pAfter"), target_dir, "p_model_GAMG")
        self.copy_and_rename(os.path.join(source_dir, "pAfter"), target_dir, "p_model_jacobi")

        p_model= self.getModelFile(p_data,p_model)

        final_file_path = os.path.join(target_dir,"p_model_jacobi")
        with open(final_file_path, 'w') as file:
            file.writelines(p_model)

        final_file_path = os.path.join(target_dir,"p_model_GAMG")
        with open(final_file_path, 'w') as file:
            file.writelines(p_model)

    def load_model(self,i_first_case):
        with open(self.config_file_field, 'r') as f:
            self.conf_field = yaml.load(f, Loader=yaml.FullLoader)

        sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

        # self.test_case_path = self.conf['test_case_path']

        file_path = f"{self.datasetPath}/case_{i_first_case}/embedding_{self.conf_field['edgeAttrName']}"
        with open(file_path, 'rb') as file:
            GNNdataDict_first = pickle.load(file)

        print(GNNdataDict_first["GNNData"])

        self.model_field = loadModel(self.conf_field, GNNdataDict_first["GNNData"], self.device)
        if self.iteration == None:
            print("iteration == None")
            model_path = os.path.join(self.conf_field["modelDir"], f'{self.conf_field["modelFilename"]}.pth')
        else:
            print("iteration = ", self.iteration)
            model_path = os.path.join(self.conf_field["modelDir"], f'{self.conf_field["modelFilename"]}_{str(self.iteration)}.pth')
        if os.path.exists(model_path):

            print("Pre-trained model found. Loading model.")
            checkpoint = torch.load(model_path)
            self.model_field.load_state_dict(checkpoint['model_state_dict'])

        num_params = sum(p.numel() for p in self.model_field.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in the model: {num_params}")

        if self.config_file_mag != "gt":
            with open(self.config_file_mag, 'r') as f:
                self.conf_mag = yaml.load(f, Loader=yaml.FullLoader)

            if self.conf_mag['edgeAttrName'] != self.conf_field['edgeAttrName']:
                print("WRONGGGGGGGGGGGGGGGGGGG edgeattr")

            file_path = f"{self.datasetPath}/case_{i_first_case}/embedding_{self.conf_mag['edgeAttrName']}"
            with open(file_path, 'rb') as file:
                GNNdataDict_first = pickle.load(file)

            # print(GNNdataDict_first["GNNData"])

            self.model_mag = loadModel(self.conf_mag, GNNdataDict_first["GNNData"], self.device)
            if self.iteration == None:
                print("iteration == None")
                model_path = os.path.join(self.conf_mag["modelDir"], f'{self.conf_mag["modelFilename"]}.pth')
            else:
                print("iteration = ", self.iteration)
                model_path = os.path.join(self.conf_mag["modelDir"],
                                          f'{self.conf_mag["modelFilename"]}_{str(self.iteration)}.pth')
            if os.path.exists(model_path):
                print("Pre-trained model found. Loading model.")
                checkpoint = torch.load(model_path)
                self.model_mag.load_state_dict(checkpoint['model_state_dict'])


    def extract_iterations_and_times(self, poissonFoam_output):

        iteration_patterns = {
            'p_zero_jacobi': r'DICPCG:\s+Solving for p_zero_jacobi,.*No Iterations (\d+)',
            'p_model_jacobi': r'DICPCG:\s+Solving for p_model_jacobi,.*No Iterations (\d+)',
            'p_zero_GAMG': r'GAMG:\s+Solving for p_zero_GAMG,.*No Iterations (\d+)',
            'p_model_GAMG': r'GAMG:\s+Solving for p_model_GAMG,.*No Iterations (\d+)'
        }

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

        # print(times)

        return iterations, times

    def run_openfoam(self):
        # Run the poissonFoam solver directly using subprocess
        # test_case_path = os.path.join(self.path_model, "0")
        poissonFoam_command = f"poissonFoamLinearSolverTestCFD -case {self.path_model}"
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

    # def load_model_input_data(self,i,pEqnD_isInput=False):
    #     localPath = os.path.join(self.path_data, f"{i}_saved")
    #     pEqnD = torch.FloatTensor(self.read_file_lines(os.path.join(localPath, "pEqnD"), 21, -4).reshape(1, -1, 1))
    #     pEqnFace = torch.FloatTensor(self.read_file_lines(os.path.join(localPath, "pEqnFace"), 21, -4).reshape(1, -1, 1))
    #     pEqnSource = torch.FloatTensor(self.read_file_lines(os.path.join(localPath, "pEqnSource"), 21, -4).reshape(1, -1, 1))
    #
    #     nCells = pEqnSource.shape[1]
    #     p = torch.FloatTensor(self.read_file_lines(os.path.join(localPath, "p"), 23, 23 + nCells).reshape(1, -1, 1))
    #
    #     if pEqnD_isInput:
    #         inputArray = torch.cat((pEqnSource, pEqnD), dim=2)
    #     else:
    #         inputArray = sourceArray + 0
    #
    #     return inputArray, p

    def runOneCase(self,i_case, ttab, results_dict,normalize_data=True):
        # print("dsfs",ttab)
        self.path_model = os.path.join(self.linearTestPath, f"case_{i_case}")
        self.path_data = os.path.join(self.datasetPath, f"case_{i_case}/level0")

        file_path = f"{self.datasetPath}/case_{i_case}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshes = pickle.load(file)

        cellPoints = dataDictMeshes[0]["cellPoints"]
        x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
        y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]

        file_path = f"{self.datasetPath}/case_{i_case}/embedding_{self.conf_field['edgeAttrName']}"
        with open(file_path, 'rb') as file:
            GNNdataDict = pickle.load(file)

        lossFunction = LossClass(os.path.join(self.datasetPath, f"case_{i_case}"), self.device)

        file_path = f"{self.datasetPath}/case_{i_case}/itab"
        with open(file_path, 'rb') as file:
            t_tab_ref = pickle.load(file)

        itab = []
        for t in ttab:
            itab.append(t_tab_ref.index(t))

        # print(itab)

        itab = np.array(itab)

        # print(t_tab_ref)

        testDataLoader = loadDataLoader(f"{self.datasetPath}/case_{i_case}",
                                        itab,
                                        1, dataDictMeshes,
                                        BCInput=self.conf_field["modelParam"]["inputChannels"]["BCInput"],
                                        gradientBCInput=self.conf_field["modelParam"]["inputChannels"]["gradientBCInput"], doShuffling=False, normalize_data=normalize_data, compute_diff_cfd=True)


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
            "rme_mag": [],
            "gt_mag": [],
            "model_mag": [],
            "t_model": [],
            "mean_diff": []
        }

        weights_dict_field = get_edgeAttrPrep(self.model_field, GNNdataDict)
        if self.config_file_mag != "gt":
            weights_dict_mag = get_edgeAttrPrep(self.model_mag, GNNdataDict)


        with torch.no_grad():
            time_list = []
            # for i in range(i_start, i_end + 1):
            i_itab = 0
            for p, pEqnD, pEqnFace, pEqnSource, inputCellArray, faceArray_incl_boundaries, BCArray, *factors in testDataLoader:
                i = itab[i_itab]
                t = ttab[i_itab]

                inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 2))
                inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray
                # print(self.device)
                inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)
                inputCellArray_dev = inputCellArray.to(self.device)
                inputFaceArray_dev = inputFaceArray.to(self.device)
                torch.cuda.synchronize()

                t1 = time.time()
                outP = self.model_field(inputCellArray_dev, inputFaceArray_dev, GNNdataDict["graphData"],
                                 GNNdataDict["graphPoolData"],
                                 GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"],weights_dict_field)

                if self.config_file_mag == "gt":
                    magnitude = factors[3] + 0

                    mag_loss = 0
                else:
                    magnitude = 10 ** (self.model_mag(inputCellArray_dev, inputFaceArray_dev, GNNdataDict["graphData"],
                                 GNNdataDict["graphPoolData"],
                                 GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"],weights_dict_mag).to("cpu"))

                    mag_loss = (((magnitude - factors[3][0]) / factors[3][0]) ** 2).item()

                # print("mag_loss",mag_loss, magnitude/factors[3], factors[3]/magnitude)



                torch.cuda.synchronize()
                t_process = time.time() - t1

                lossPressure = lossFunction.computeLossPressure(outP, p.to(self.device))

                # outP = outP - torch.mean(outP)

                # print("pressure loss: ", lossPressure.item())
                # diff = p.unsqueeze(-1) - outP.to("cpu")
                # outP += torch.mean(diff)


                diff = p.unsqueeze(-1) - outP.to("cpu")
                mean_diff = torch.mean(diff).item()

                outP = outP.to("cpu") * magnitude * factors[2] + factors[4].unsqueeze(-1)

                time_list.append(time.time() - t1)

                self.setup_test_conditions(t, outP.to("cpu"))
                # self.setup_test_conditions(t, p)

                # self.setup_test_conditions(t, p.unsqueeze(-1) * factors[3] * factors[2] + factors[4].unsqueeze(-1),factors[4].unsqueeze(-1))

                iterations, times = self.run_openfoam()
                print(iterations)
                # if iterations['p_zero_jacobi']-iterations['p_model_jacobi']>80 and iterations['p_zero_GAMG']-iterations['p_model_GAMG']>1:
                if 1 == 2:
                    # if torch.max(abs(p[0]))<10:
                    p_plot = p[0] * factors[3] * factors[2] + factors[4]
                    outP_plot = (outP - factors[4].unsqueeze(-1))/ (magnitude * factors[2])

                    lower_bound = np.percentile(p, 1)
                    upper_bound = np.percentile(p, 99)
                    p = np.clip(p, lower_bound, upper_bound)

                    lower_bound = np.percentile(outP_plot, 1)
                    upper_bound = np.percentile(outP_plot, 99)
                    outP_plot = np.clip(outP_plot, lower_bound, upper_bound)


                    zmin = min(np.min(p[0].numpy()), np.min(outP_plot[0, :, 0].cpu().numpy()),
                               np.min(p[0].numpy() - outP_plot[0, :, 0].cpu().numpy()))
                    zmax = max(np.max(p[0].numpy()), np.max(outP_plot[0, :, 0].cpu().numpy()),
                               np.max(p[0].numpy() - outP_plot[0, :, 0].cpu().numpy()))
                    zlim = [zmin, zmax]

                    fig = plt.figure(figsize=(20, 10))

                    print(outP_plot.shape)
                    # pr

                    # ax = fig.add_subplot(2, 2, 1)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "Ground truth",zlim=zlim,apply_percentile=True)
                    # ax = fig.add_subplot(2, 2, 2)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP_plot[0, :, 0].cpu(), "Prediction",zlim=zlim,apply_percentile=True)
                    # ax = fig.add_subplot(2, 2, 3)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP_plot[0, :, 0].cpu() - p[0], "Diff",zlim=zlim,centered_colorbar=True,apply_percentile=True)
                    # ax = fig.add_subplot(2, 2, 4)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0], "Source term",centered_colorbar=True,apply_percentile=True)
                    # plt.tight_layout()
                    # plt.show()

                    ax = fig.add_subplot(2, 2, 1)

                    x1 = 6.5
                    x2 = 15
                    y1 = 5.5
                    y2 = 10.5
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "Ground truth",zlim=zlim,apply_percentile=True)
                    plt.xlim([x1,x2])
                    plt.ylim([y1,y2])
                    # plt.xlim([0,28])
                    # plt.ylim([0,16])
                    ax = fig.add_subplot(2, 2, 2)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP_plot[0, :, 0].cpu(), "Prediction",zlim=zlim,apply_percentile=True)
                    plt.xlim([x1,x2])
                    plt.ylim([y1,y2])
                    # plt.xlim([0, 28])
                    # plt.ylim([0, 16])
                    ax = fig.add_subplot(2, 2, 3)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP_plot[0, :, 0].cpu() - p[0], "Diff",centered_colorbar=True,apply_percentile=True)
                    plt.xlim([x1,x2])
                    plt.ylim([y1,y2])
                    # plt.xlim([0, 28])
                    # plt.ylim([0, 16])
                    ax = fig.add_subplot(2, 2, 4)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0], "Source term",centered_colorbar=True,apply_percentile=True)
                    plt.xlim([x1,x2])
                    plt.ylim([y1,y2])
                    # plt.xlim([0, 28])
                    # plt.ylim([0, 16])
                    plt.tight_layout()
                    plt.show()

                # print(iterations)
                # try:

                results_dict_sub["p_zero_jacobi_iterations"].append(iterations["p_zero_jacobi"])
                results_dict_sub["p_model_jacobi_iterations"].append(iterations["p_model_jacobi"])
                results_dict_sub["p_zero_GAMG_iterations"].append(iterations["p_zero_GAMG"])
                results_dict_sub["p_model_GAMG_iterations"].append(iterations["p_model_GAMG"])

                results_dict_sub["p_zero_jacobi_times"].append(times["p_zero_jacobi"])
                results_dict_sub["p_model_jacobi_times"].append(times["p_model_jacobi"])
                results_dict_sub["p_zero_GAMG_times"].append(times["p_zero_GAMG"])
                results_dict_sub["p_model_GAMG_times"].append(times["p_model_GAMG"])

                results_dict_sub["rme"].append(lossPressure.item())
                results_dict_sub["rme_mag"].append(mag_loss)

                results_dict_sub["gt_mag"].append(factors[3])
                results_dict_sub["model_mag"].append(magnitude)

                results_dict_sub["t_model"].append(t_process)
                results_dict_sub["mean_diff"].append(mean_diff)


                # except:
                #     print("code does not work")

                i_itab += 1

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
            results_dict["rme_mag"].append(results_dict_sub["rme_mag"])

            results_dict["gt_mag"].append(results_dict_sub["gt_mag"])
            results_dict["model_mag"].append(results_dict_sub["model_mag"])

            results_dict["t_model"].append(results_dict_sub["t_model"][1:])
            results_dict["mean_diff"].append(results_dict_sub["mean_diff"])


        except:
            print("code does not work 2")


        return results_dict

    def run(self,test_cases, ttab):
        self.load_model(test_cases[0])
        self.model_field.eval()
        if self.config_file_mag != "gt":
            self.model_mag.eval()
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
            "rme_mag": [],
            "gt_mag": [],
            "model_mag": [],
            "t_model": [],
            "mean_diff": []
        }

        for i_case in test_cases:
            results_dict = self.runOneCase(i_case, ttab, results_dict)

        with open(os.path.join(self.conf_field["modelDir"], 'test_n_iterations_dict_test.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)

        print(f"-------------RESULTS {self.config_file_field}, {self.config_file_mag} iteration: {self.iteration}--------------------")
        print("RME MEAN: ",np.mean(results_dict["rme"]))
        # print("RME MEAN PER CASE: ", np.mean(results_dict["rme"],axis=1))
        iterations_PCG = np.array(results_dict["p_zero_jacobi_iterations"]) - np.array(results_dict["p_model_jacobi_iterations"])
        iterations_GAMG = np.array(results_dict["p_zero_GAMG_iterations"]) - np.array(results_dict["p_model_GAMG_iterations"])
        print("")
        print("PCG ITERATIONS SAVED: ", np.mean(iterations_PCG))
        # print("PCG ITERATIONS SAVED PER CASE: ", np.mean(iterations_PCG,axis=1))
        print("")
        print("GAMG ITERATIONS SAVED: ", np.mean(iterations_GAMG))
        # print("GAMG ITERATIONS SAVED PER CASE: ", np.mean(iterations_GAMG,axis=1))
        print("")
        ratios_PCG = np.array(results_dict["p_model_jacobi_iterations"]) / np.array(results_dict["p_zero_jacobi_iterations"])
        ratios_GAMG = np.array(results_dict["p_model_GAMG_iterations"]) / np.array(results_dict["p_zero_GAMG_iterations"])
        print("PCG RATIOS: ", np.mean(ratios_PCG))
        # print("PCG RATIOS PER CASE: ", np.mean(ratios_PCG,axis=1))
        print("")
        print("GAMG RATIOS: ", np.mean(ratios_GAMG))
        # print("GAMG RATIOS PER CASE: ", np.mean(ratios_GAMG,axis=1))
        print("")
        print("GAMG MODEL avg time: ", np.sum(results_dict["p_model_GAMG_times"])/np.sum(results_dict["p_model_GAMG_iterations"]))
        print("GAMG ZERO avg time: ", np.sum(results_dict["p_zero_GAMG_times"])/np.sum(results_dict["p_zero_GAMG_iterations"]))
        print("jacobi MODEL avg time: ", np.sum(results_dict["p_model_jacobi_times"])/np.sum(results_dict["p_model_jacobi_iterations"]))
        print("jacobi ZERO avg time: ", np.sum(results_dict["p_zero_jacobi_times"])/np.sum(results_dict["p_zero_jacobi_iterations"]))
        print("MODEL avg time: ", np.mean(results_dict["t_model"]))

        tgamgModel = np.sum(results_dict["p_model_GAMG_times"])/np.sum(results_dict["p_model_GAMG_iterations"])
        tgamgZero = np.sum(results_dict["p_zero_GAMG_times"])/np.sum(results_dict["p_zero_GAMG_iterations"])
        tjacobiModel = np.sum(results_dict["p_model_jacobi_times"])/np.sum(results_dict["p_model_jacobi_iterations"])
        tjacobiZero = np.sum(results_dict["p_zero_jacobi_times"])/np.sum(results_dict["p_zero_jacobi_iterations"])
        tmodel = np.mean(results_dict["t_model"])

        print("GAMG MODEL time ratio: ",tmodel/tgamgModel)
        print("GAMG ZERO time ratio: ",tmodel/tgamgZero)
        print("jacobi MODEL time ratio: ",tmodel/tjacobiModel)
        print("jacobi ZERO time ratio: ",tmodel/tjacobiZero)

        print("mean_diff: ",np.mean(abs(np.array(results_dict["mean_diff"]))))


        return results_dict

def autoTest(test_cases,ttab,config_files_field,config_files_mag,basePathDatasets,basePathLinearTest,datasetName,device,main_modeldir,save_name,iteration=None):

    save_path = os.path.join(main_modeldir, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for config_file_field in config_files_field:
        for config_file_mag in config_files_mag:

            tester = testNIterationsMultipleMeshesCFD(basePathDatasets, basePathLinearTest, datasetName,openfoam_source_path, config_file_field, config_file_mag, device, iteration=iteration)
            results_dict = tester.run(test_cases, ttab)

            field_name = config_file_field.split('/')[-1].split('.')[0]
            mag_name = config_file_mag.split('/')[-1].split('.')[0]

            fileName = f"{save_name}/field_{field_name}_mag_{mag_name}.pkl"

            path = os.path.join(main_modeldir, fileName)
            with open(path, 'wb') as f:
                pickle.dump(results_dict, f)



if __name__ == '__main__':
    #define which step must be evaluated
    step = 4

    ### DO NOT TOUCH
    basePathDatasets = "/home/justinbrusche/datasets"
    datasetName = f"step{step}"
    basePathLinearTest = "/home/justinbrusche/linearSolverTests"
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"
    device = 'cuda'

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
    ###

    # Define model
    config_files_field = [f'config_files_step{step}/veryVeryVerySmall.yaml',
                        f'config_files_step{step}/veryVerySmall.yaml',
                        f'config_files_step{step}/verySmall.yaml',
                        f'config_files_step{step}/small.yaml']


    config_files_mag = ['gt',
                        f'config_files_step{step}/magVeryVeryVerySmall.yaml',
                        f'config_files_step{step}/magVeryVerySmall.yaml',
                        f'config_files_step{step}/magVerySmall.yaml',
                        f'config_files_step{step}/magSmall.yaml',]

    # define which cases must be evaluated
    test_cases = [0]
    test_cases = [5,15]

    #Define which training epoch model must be tested (only even numbers)
    iteration = None
    iteration = 200

    # Save directory
    main_modeldir = f'/home/justinbrusche/modeldirs_step{step}/solverResults'
    save_name = "e6"

    if nSets == 1:
        # iTab = np.arange(nPerCase)
        # iTabTest = iTab[::10]

        writeInterval = endTime / nPerCase
        ttab = np.arange(writeInterval, endTime + 0.5 * writeInterval, writeInterval)
        ttab = [str(format(num, '.5f')).rstrip('0').rstrip('.') for num in ttab]
        # ttab = ttab[2:]
        ttab = ttab[::10]
        ttab = ttab[1:]

    else:
        writeInterval = endTime / nPerCase
        ttab = np.arange(writeInterval, endTime + 0.5 * writeInterval, writeInterval)
        ttab = [str(format(num, '.5f')).rstrip('0').rstrip('.') for num in ttab]
        ttab = ttab[1:]

    autoTest(test_cases,ttab,config_files_field,config_files_mag,basePathDatasets,basePathLinearTest,datasetName,device,main_modeldir,save_name,iteration)

    # tester = testNIterationsMultipleMeshesCFD(basePathDatasets,basePathLinearTest,datasetName,openfoam_source_path,config_file,device,iteration=iteration)
    # results_dict = tester.run(test_cases, ttab)
