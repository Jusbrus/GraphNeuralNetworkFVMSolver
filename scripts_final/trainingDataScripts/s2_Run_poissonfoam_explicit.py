import os
import subprocess
import time
import shutil
import pickle
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import numpy as np

def setup_source(openfoam_source_path):
    # Create a shell script with the combined commands to source the environment
    source_commands = f"""
    cd /
    source {openfoam_source_path}/etc/bashrc
    env
    """

    # Execute the commands in a single shell session to source the OpenFOAM environment and capture the environment variables
    result = subprocess.run(source_commands, shell=True, capture_output=True, text=True, executable="/bin/bash")

    # Print the results of sourcing the environment
    if result.returncode == 0:
        print("Environment sourced successfully.")
        env_vars = result.stdout.splitlines()
        env_dict = dict(line.split("=", 1) for line in env_vars if "=" in line)
    else:
        print("Sourcing environment failed.")
        print(result.stderr)
        exit(1)

    # Set the environment variables in the current process
    os.environ.update(env_dict)

def load_data(seeds_dict_path):
    with open(seeds_dict_path, 'rb') as f:
        seeds_dict = pickle.load(f)

    return seeds_dict

def run_openfoam(test_case_path):
    # Run the poissonFoam solver directly using subprocess
    poissonFoam_command = f"poissonFoamExplicit -case {test_case_path}"
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

def array_to_nonuniform_list_value(array):
    values = "\n".join(f"    {value}" for value in array)
    nonuniform_list = f"nonuniform List<scalar>\n{len(array)}(\n{values}\n)"
    return nonuniform_list

def set_initial_conditions(i, test_case_path, seeds_dict, bc_value_dict, bc_gradient_dict, fields_dict, sides_list, p_bcs_set_to_zero=False, only_diriglet=False, tau_1=True, p_bias=False):
    if tau_1:
        tau_factor = 0
    else:
        tau_factor = 1

    if p_bias:
        p_mag_factor = 1
        p_bias_value = np.random.uniform(-1.5,1.5)
    else:
        p_mag_factor = 1
        p_bias_value = 0

    bc_types = ["fixedValue","fixedGradient"]
    bc_types_zero_bcs = ["fixedValue","zeroGradient"]
    sides_list_bounds = ["Left","Upper","Right","Lower"]

    # p input
    p_input_file_path = f"{test_case_path}/0/p_input"
    p_input_file = ParsedParameterFile(p_input_file_path)

    for j in range(len(sides_list)):
        p_input_file["boundaryField"][sides_list[j]] = {}
        if sides_list[j] in sides_list_bounds:
            if only_diriglet:
                p_input_file["boundaryField"][sides_list[j]]["type"] = "fixedValue"
            else:
                p_input_file["boundaryField"][sides_list[j]]["type"] = bc_types[seeds_dict["p_BC_type"][i][j]]
            if seeds_dict["p_BC_type"][i][j] == 0 or only_diriglet == True:
                p_input_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(p_mag_factor * bc_value_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][sides_list[j]][seeds_dict["p_seeds"][i]] + p_bias_value)
            else:
                p_input_file["boundaryField"][sides_list[j]]["gradient"] = array_to_nonuniform_list_value(p_mag_factor * bc_gradient_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][sides_list[j]][seeds_dict["p_seeds"][i]])
        else:
            p_input_file["boundaryField"][sides_list[j]]["type"] = "zeroGradient"

    p_input_file["internalField"] = array_to_nonuniform_list_value(p_mag_factor * fields_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][seeds_dict["p_seeds"][i]] + p_bias_value)

    # p
    p_file_path = f"{test_case_path}/0/p"
    p_file = ParsedParameterFile(p_file_path)

    for j in range(len(sides_list)):
        p_file["boundaryField"][sides_list[j]] = {}
        if sides_list[j] in sides_list_bounds:

            if only_diriglet == True:
                p_file["boundaryField"][sides_list[j]]["type"] = "fixedValue"
            elif p_bcs_set_to_zero:
                p_file["boundaryField"][sides_list[j]]["type"] = bc_types_zero_bcs[seeds_dict["p_BC_type"][i][j]]
            else:
                p_file["boundaryField"][sides_list[j]]["type"] = bc_types[seeds_dict["p_BC_type"][i][j]]
            if seeds_dict["p_BC_type"][i][j] == 0 or only_diriglet == True:
                if p_bcs_set_to_zero:
                    p_file["boundaryField"][sides_list[j]]["value"] = "uniform 0"
                else:
                    p_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(p_mag_factor * bc_value_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][sides_list[j]][seeds_dict["p_seeds"][i]] + p_bias_value)
            else:
                if p_bcs_set_to_zero:
                    pass
                else:
                    p_file["boundaryField"][sides_list[j]]["gradient"] = array_to_nonuniform_list_value(p_mag_factor * bc_gradient_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][sides_list[j]][seeds_dict["p_seeds"][i]])

        else:
            p_file["boundaryField"][sides_list[j]]["type"] = "zeroGradient"


    p_file["internalField"] = array_to_nonuniform_list_value(p_mag_factor * fields_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][seeds_dict["p_seeds"][i]] + p_bias_value)
    # p_file["internalField"] = "uniform 0"
    # Tau
    tau_file_path = f"{test_case_path}/0/tau"
    tau_file = ParsedParameterFile(tau_file_path)

    tau_field = 4+tau_factor*fields_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][seeds_dict["tau_seeds"][i]]
    tau_field = np.where(tau_field<1,2-tau_field,tau_field)

    tau_file["internalField"] = array_to_nonuniform_list_value(tau_field)

    # # # Histogram with automatic bin width
    # plt.figure(figsize=(10, 6))
    # plt.hist(tau_field, bins='auto', edgecolor='black')
    # plt.title('Histogram with Automatic Bin Width cylinder flow')
    # plt.xlim(left=0)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()

    for j in range(len(sides_list)):
        tau_bc = 4+tau_factor*bc_value_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][sides_list[j]][seeds_dict["tau_seeds"][i]]
        tau_bc = np.where(tau_bc < 1, 2-tau_bc, tau_bc)
        # print(tau_file["boundaryField"])
        tau_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(tau_bc)

    # centers_x, centers_y = get_cell_centers(test_case_path)
    # fig1 = plt.figure(figsize=(9, 9))
    # ax1 = fig1.add_subplot(111)
    # plot_2d(fig1, ax1, centers_x, centers_y,fields_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][seeds_dict["tau_seeds"][i]], "0")
    # plt.show()

    p_input_file.writeFile()
    p_file.writeFile()
    tau_file.writeFile()

def set_initial_conditions_direct(i, test_case_path, p_field, p_boundary_dict, tau_field, tau_boundary_dict, p_BC_type, sides_list, p_bcs_set_to_zero=False, only_diriglet=False, tau_1=True, p_bias=False):
    if tau_1:
        tau_factor = 0
    else:
        tau_factor = 1

    if p_bias:
        p_mag_factor = 1
        p_bias_value = np.random.uniform(-1.5,1.5)
    else:
        p_mag_factor = 1
        p_bias_value = 0

    bc_types = ["fixedValue","fixedGradient"]
    bc_types_zero_bcs = ["fixedValue","zeroGradient"]
    sides_list_bounds = ["Left","Upper","Right","Lower"]

    # p input
    p_input_file_path = f"{test_case_path}/0/p_input"
    p_input_file = ParsedParameterFile(p_input_file_path)
    i_boundary = 0
    for j in range(len(sides_list)):
        p_input_file["boundaryField"][sides_list[j]] = {}
        if sides_list[j] in sides_list_bounds:
            if only_diriglet:
                p_input_file["boundaryField"][sides_list[j]]["type"] = "fixedValue"
            else:
                p_input_file["boundaryField"][sides_list[j]]["type"] = bc_types[p_BC_type[i_boundary]]
            if p_BC_type[i_boundary] == 0 or only_diriglet == True:
                p_input_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(p_mag_factor * p_boundary_dict[sides_list_bounds[i_boundary]] + p_bias_value)
            else:
                p_input_file["boundaryField"][sides_list[j]]["gradient"] = array_to_nonuniform_list_value(p_mag_factor * p_boundary_dict[sides_list_bounds[i_boundary]])
            i_boundary+=1
        else:
            p_input_file["boundaryField"][sides_list[j]]["type"] = "zeroGradient"

    p_input_file["internalField"] = array_to_nonuniform_list_value(p_mag_factor * p_field + p_bias_value)

    # p
    p_file_path = f"{test_case_path}/0/p"
    p_file = ParsedParameterFile(p_file_path)
    i_boundary = 0
    for j in range(len(sides_list)):
        p_file["boundaryField"][sides_list[j]] = {}
        if sides_list[j] in sides_list_bounds:

            if only_diriglet == True:
                p_file["boundaryField"][sides_list[j]]["type"] = "fixedValue"
            elif p_bcs_set_to_zero:
                p_file["boundaryField"][sides_list[j]]["type"] = bc_types_zero_bcs[p_BC_type[i_boundary]]
            else:
                p_file["boundaryField"][sides_list[j]]["type"] = bc_types[p_BC_type[i_boundary]]
            if p_BC_type[i_boundary] == 0 or only_diriglet == True:
                if p_bcs_set_to_zero:
                    p_file["boundaryField"][sides_list[j]]["value"] = "uniform 0"
                else:
                    p_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(p_mag_factor * p_boundary_dict[sides_list_bounds[i_boundary]] + p_bias_value)
            else:
                if p_bcs_set_to_zero:
                    pass
                else:
                    p_file["boundaryField"][sides_list[j]]["gradient"] = array_to_nonuniform_list_value(p_mag_factor * p_boundary_dict[sides_list_bounds[i_boundary]])
            i_boundary+=1

        else:
            p_file["boundaryField"][sides_list[j]]["type"] = "zeroGradient"


    p_file["internalField"] = array_to_nonuniform_list_value(p_mag_factor * p_field + p_bias_value)
    # p_file["internalField"] = "uniform 0"
    # Tau
    tau_file_path = f"{test_case_path}/0/tau"
    tau_file = ParsedParameterFile(tau_file_path)

    tau_field = 4+tau_factor*tau_field
    tau_field = np.where(tau_field<1,2-tau_field,tau_field)

    tau_file["internalField"] = array_to_nonuniform_list_value(tau_field)

    # # # Histogram with automatic bin width
    # plt.figure(figsize=(10, 6))
    # plt.hist(tau_field, bins='auto', edgecolor='black')
    # plt.title('Histogram with Automatic Bin Width cylinder flow')
    # plt.xlim(left=0)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()
    # print(p_boundary_dict[sides_list[0]])
    for j in range(len(sides_list)):
        tau_bc = 4+tau_factor*tau_boundary_dict[sides_list[j]]
        tau_bc = np.where(tau_bc < 1, 2-tau_bc, tau_bc)
        # print(tau_file["boundaryField"])
        tau_file["boundaryField"][sides_list[j]]["value"] = array_to_nonuniform_list_value(tau_bc)

    # centers_x, centers_y = get_cell_centers(test_case_path)
    # fig1 = plt.figure(figsize=(9, 9))
    # ax1 = fig1.add_subplot(111)
    # plot_2d(fig1, ax1, centers_x, centers_y,fields_dict[seeds_dict["orientation"][i]][seeds_dict["smoothness"][i]][seeds_dict["tau_seeds"][i]], "0")
    # plt.show()

    p_input_file.writeFile()
    p_file.writeFile()
    tau_file.writeFile()


def modify_control_dict(test_case_path, i):
    control_dict_path = f"{test_case_path}/system/controlDict"
    control_dict = ParsedParameterFile(control_dict_path)

    control_dict["endTime"] = i + 1
    control_dict["deltaT"] = i + 1
    control_dict["writeInterval"] = i + 1

    control_dict.writeFile()

if __name__ == '__main__':
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0/level0"
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1/level0"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2/level0"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p4"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"


    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    sides_list = dataDictMeshes[0]["boundaryNames"]

    test_case_path = test_case_path + "/level0"

    seeds_dict_path = "/home/justinbrusche/scripts_FVM_2/trainingDataScripts/data_input_openfoam/seeds_dict.pkl"


    seeds_dict = load_data(seeds_dict_path)
    print(seeds_dict.keys())
    print(len(seeds_dict["smoothness"]))

    setup_source(openfoam_source_path)

    # i = 1
    # set_initial_conditions(i, test_case_path, seeds_dict, bc_dict, fields_dict)
    # modify_control_dict(test_case_path, i)
    # run_openfoam(test_case_path)
    # i = 0
    # set_initial_conditions(i, test_case_path, seeds_dict, bc_value_dict, bc_gradient_dict, fields_dict)

    for i in range(11300,19200):
    # for i in [0, 60]:
        # set_initial_conditions_constant_smoothness(i, test_case_path, seeds_dict, bc_dict, fields_dict)
        set_initial_conditions(i, test_case_path, seeds_dict, bc_value_dict, bc_gradient_dict, fields_dict, sides_list, p_bcs_set_to_zero=False, only_diriglet=False, tau_1=False,
                               p_bias=True)
        modify_control_dict(test_case_path, i)
        time_elapsed = run_openfoam(test_case_path)
        print(i, time_elapsed)

        if os.path.exists(os.path.join(test_case_path, str(i + 1) + "_saved")):
            shutil.rmtree(os.path.join(test_case_path, str(i + 1) + "_saved"))
        os.rename(os.path.join(test_case_path, str(i + 1)), os.path.join(test_case_path, str(i + 1) + "_saved"))

