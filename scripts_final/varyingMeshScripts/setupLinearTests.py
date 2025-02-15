import shutil
import os
from createMeshes import getMeshes
import subprocess
import numpy as np
import pickle
import re
import sys

sys.path.append("/home/justinbrusche/scripts_final")
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")

class SetupLinearTests:
    def __init__(self, basePathDatasets,basePathLinearTest, datasetName,openfoam_source_path,nSets):
        self.basePathDatasets = basePathDatasets
        self.basePathLinearTest = basePathLinearTest
        self.openfoam_source_path = openfoam_source_path
        self.datasetName = datasetName
        self.nSets = nSets

        self.datasetPath = os.path.join(self.basePathDatasets, self.datasetName)
        self.LinearSolverTestsPath = os.path.join(self.basePathLinearTest, self.datasetName)

    def modify_boundary_file(self,filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        in_side_block = False
        in_object_block = False

        for line in lines:
            # Check for the start of the "Side" block
            if "Side" in line:
                in_side_block = True
            # Check for the start of the "Object" block
            elif "Object" in line:
                in_object_block = True

            # Modify the type and physicalType in the "Side" block
            if in_side_block:
                if "type" in line:
                    modified_lines.append('        type            empty;\n')
                    continue
                if "physicalType" in line:
                    modified_lines.append('        physicalType    empty;\n')
                    in_side_block = False  # End of modification for "Side"
                    continue

            # Modify the type and physicalType in the "Object" block
            if in_object_block:
                if "type" in line:
                    modified_lines.append('        type            wall;\n')
                    continue
                if "physicalType" in line:
                    modified_lines.append('        physicalType    wall;\n')
                    in_object_block = False  # End of modification for "Object"
                    continue

            # Append the line without modification if no condition is met
            modified_lines.append(line)

        # Write the modified content back to the file
        with open(filename, 'w') as file:
            file.writelines(modified_lines)

    def modify_tolerance(self,file_path, new_tolerance):
        with open(file_path, 'r') as file:
            content = file.read()

        # Use regular expressions to find and replace the tolerance values
        content_modified = re.sub(r'tolerance\s+[0-9.e+-]+;', f'tolerance {new_tolerance};', content)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(content_modified)

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

        # Check if the destination already exists
        if os.path.exists(destination_path):
            print(f"'{destination_path}' already exists. Skipping copy.")
        else:
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Check if the source is a directory or a file
            if os.path.isdir(source_path):
                # Copy the source directory to the destination path
                shutil.copytree(source_path, destination_path)
                print(f"Copied directory from {source_path} to {destination_path}")
            elif os.path.isfile(source_path):
                # Copy the source file to the destination path
                shutil.copy2(source_path, destination_path)
                print(f"Copied file from {source_path} to {destination_path}")
            else:
                print(f"Source path '{source_path}' does not exist or is not a valid file/directory.")

    def runGmshToFoam(self,mesh_name,path):
        mesh_file_name = f"{mesh_name}.msh"
        commands = f"""
        cd /
        cd {path}
        gmshToFoam {mesh_file_name}
        writeMeshData
        """
        subprocess.run(commands, shell=True, capture_output=True, text=True, executable="/bin/bash")

    def createDirectories(self,tolerance,perlin_noise=False):
        # Copy and rename the base directory only if it doesn't exist
        self.setup_source()
        self.copy_and_rename(os.path.join(self.basePathLinearTest, "baseCase"), self.basePathLinearTest, self.datasetName)

        mesh_name = "generatedMesh"
        if perlin_noise:
            containsObject, _, _, _, _ = getMeshes(21)
        else:
            containsObject = np.full(self.nSets, True)
        for i in range(self.nSets):
            if containsObject[i]:
                self.copy_and_rename(os.path.join(self.LinearSolverTestsPath, "baseCaseObject"), self.LinearSolverTestsPath,
                                         f"case_{i}")

            else:
                self.copy_and_rename(os.path.join(self.LinearSolverTestsPath, "baseCaseNoObject"), self.LinearSolverTestsPath,
                                         f"case_{i}")

            self.copy_and_rename(os.path.join(self.datasetPath, f"case_{i}/level0/{mesh_name}.msh"),
                                 self.LinearSolverTestsPath,
                                 f"case_{i}/{mesh_name}.msh")

            self.runGmshToFoam(mesh_name,os.path.join(self.LinearSolverTestsPath,f"case_{i}"))

            self.modify_boundary_file(os.path.join(self.LinearSolverTestsPath,f"case_{i}/constant/polyMesh/boundary"))

            self.modify_tolerance(os.path.join(self.LinearSolverTestsPath,f"case_{i}/system/fvSolution"),tolerance)


if __name__ == '__main__':
    basePathDatasets = "/home/justinbrusche/datasets"
    datasetName = "foundationalParameters_cfd_mesh_airfoil_2"

    # Select dataset
    # datasetName = "step3"
    datasetName = "foundationalParameters"


    basePathLinearTest = "/home/justinbrusche/linearSolverTests"
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"

    # Define tolerance
    tolerance = 1e-3

    # Select if the dataset is a perlin noise dataset or not
    is_perlin_noise = True

    # Define number of simulations in the dataset
    nSets = 22


    builder = SetupLinearTests(basePathDatasets, basePathLinearTest, datasetName,openfoam_source_path, nSets)
    builder.createDirectories(tolerance,perlin_noise=is_perlin_noise)

