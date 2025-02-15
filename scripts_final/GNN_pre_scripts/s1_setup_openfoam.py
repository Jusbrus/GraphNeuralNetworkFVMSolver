import torch
import subprocess
import os
import shutil
import copy
from GNN_pre_scripts.generateMeshPerlinDataset import *
from GNN_pre_scripts.generateMeshCFDDataset import *
from GNN_pre_scripts._GetGraphData import *
import gmsh
import pickle

class SetupOpenfoam:
    def __init__(self,base_path,test_case_path,mesh_name,nLevels,characteristicLength,objects_present = False,faceToCell=False,faceToPoint=False, faceToEdgeCenters = False):
        self.base_path = base_path
        self.test_case_path = test_case_path
        self.mesh_name = mesh_name
        self.nLevels = nLevels
        self.characteristicLength = characteristicLength
        self.faceToCell = faceToCell
        self.faceToPoint = faceToPoint
        self.faceToEdgeCenters = faceToEdgeCenters
        self.objects_present = objects_present

    def setup_source(self,openfoam_source_path):
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
            # print("Environment sourced successfully.")
            env_vars = result.stdout.splitlines()
            env_dict = dict(line.split("=", 1) for line in env_vars if "=" in line)
        else:
            # print("Sourcing environment failed.")
            # print(result.stderr)
            exit(1)

        # Set the environment variables in the current process
        os.environ.update(env_dict)

    # Python script to modify OpenFOAM boundary file

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

    def createMesh(self,level,characteristicLength):
        gmsh.initialize()
        geo_file = f"{self.test_case_path}/{self.mesh_name}.geo"

        with open(geo_file, 'r') as file:
            lines = file.readlines()
        with open(geo_file, 'w') as file:
            for line in lines:
                # print(line)
                if line.startswith("characteristicLength ="):
                    file.write(f"characteristicLength = {characteristicLength};\n")
                else:
                    file.write(line)

        gmsh.open(geo_file)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        output_directory = f"{self.test_case_path}/level{level}"
        mesh_file_name = f"{self.mesh_name}.msh"

        os.makedirs(output_directory, exist_ok=True)
        gmsh.write(os.path.join(output_directory, mesh_file_name))
        gmsh.finalize()

    def createMeshWithObject(self,level,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=None, box_list=None):
        # print(level)

        if moreObjects:
            variable_dict_variant_copy = copy.deepcopy(variable_dict_variant)

            factor = 2**level
            # print("fdsagfdsafdsafdsafdsaz",factor)
            for key in variable_dict_variant.keys():
                variable_dict_variant_copy[key] = int(variable_dict_variant[key]/factor)

            # print(standard_variable_dict,custom_variable_dict,level)

            generator = GenerateMeshMoreObjects(objects_list,box_list, variable_dict_variant_copy,variable_dict_invariant, f"{self.test_case_path}/level{level}/{self.mesh_name}.geo")
            generator.get_mesh()
        else:
            variable_dict_variant_copy = copy.deepcopy(variable_dict_variant)

            factor = 2**level
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",factor)
            for key in variable_dict_variant.keys():
                variable_dict_variant_copy[key] = int(variable_dict_variant[key]/factor)

            # print(standard_variable_dict,custom_variable_dict,level)

            generator = GenerateMeshOneObject(objects_list[0], variable_dict_variant_copy,variable_dict_invariant, f"{self.test_case_path}/level{level}/{self.mesh_name}.geo")
            generator.get_mesh()

        gmsh.initialize()
        geo_file = f"{self.test_case_path}/level{level}/{self.mesh_name}.geo"

        gmsh.open(geo_file)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        output_directory = f"{self.test_case_path}/level{level}"
        mesh_file_name = f"{self.mesh_name}.msh"

        os.makedirs(output_directory, exist_ok=True)
        gmsh.write(os.path.join(output_directory, mesh_file_name))
        gmsh.finalize()

    def createMeshes(self):
        for i in range(self.nLevels):
            self.createMesh(i,int(round(self.characteristicLength/(2**i))))

    def createMeshesWithObjects(self,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=False, box_list=None):
        for i in range(self.nLevels):
            self.createMeshWithObject(i,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=moreObjects, box_list=box_list)

    def duplicate_maps(self):
        map_to_duplicate = f"{self.test_case_path}/level0"
        for i in range(1,self.nLevels):
            new_map_name = f"{self.test_case_path}/level{i}"
            if os.path.exists(new_map_name):
                # Remove the target directory and all its contents
                shutil.rmtree(new_map_name)
                # print(f"Deleted existing directory: {new_map_name}")

            # Copy the source directory to the target location
            shutil.copytree(map_to_duplicate, new_map_name)
            # print(f"Copied {map_to_duplicate} to {new_map_name}")

    def runGmshToFoam(self):
        for i in range(self.nLevels):
            mesh_file_name = f"{self.mesh_name}.msh"
            commands = f"""
            cd /
            cd {self.test_case_path}/level{i}
            gmshToFoam {mesh_file_name}
            """
            subprocess.run(commands, shell=True, capture_output=True, text=True, executable="/bin/bash")

    def runWriteMeshData(self):
        # print("dfs")
        for i in range(self.nLevels):
            commands = f"""
            cd /
            cd {self.test_case_path}/level{i}
            writeMeshData
            """
            subprocess.run(commands, shell=True, capture_output=True, text=True, executable="/bin/bash")

    def prepareFiles(self,openfoam_source_path):
        self.setup_source(openfoam_source_path)
        self.duplicate_maps()
        self.createMeshes()
        self.runGmshToFoam()
        self.runWriteMeshData()
        self.modify_boundary_file(self.test_case_path+"/level0/constant/polyMesh/boundary")

    def prepareFilesWithObject(self,openfoam_source_path,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=False,box_list=None):
        self.setup_source(openfoam_source_path)
        self.duplicate_maps()
        self.createMeshesWithObjects(variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=moreObjects, box_list=box_list)
        self.runGmshToFoam()
        self.runWriteMeshData()
        self.modify_boundary_file(self.test_case_path+"/level0/constant/polyMesh/boundary")

    def getMeshData(self,no_layers=False):
        dataDictMeshes = {}
        for i in range(self.nLevels):
            if no_layers:
                directory = f"{self.test_case_path}"
            else:
                directory = f"{self.test_case_path}/level{i}"

            meshProcessor = GetGraphData(directory)
            dataDictMeshes[i] = {}

            dataDictMeshes[i]["cellCenters"] = meshProcessor.cellCenters
            dataDictMeshes[i]["nNodesDualMesh"] = len(dataDictMeshes[i]["cellCenters"])

            dataDictMeshes[i]["boundaryPoints"] = meshProcessor.get_boundary_points()
            dataDictMeshes[i]["pointTypes"] = meshProcessor.get_point_types()

            dataDictMeshes[i]["faceCoordinates"] = meshProcessor.get_face_centers()

            dataDictMeshes[i]["pointCoordinates"] = meshProcessor.pointCoordinates

            sides_list = meshProcessor.get_boundary_names()
            dataDictMeshes[i]["boundaryNames"] = sides_list

            dataDictMeshes[i]["boundaryFacesDict"] = meshProcessor.get_boundary_faces_ordened(sides_list)

            if i == 0:
                pointTypes = torch.from_numpy(meshProcessor.get_point_types())
                # print("pointTypes", pointTypes)
                file_path = f"{self.test_case_path}/pointTypes.pt"
                # Save the tensor to the file
                torch.save(pointTypes, file_path)

                cellTypes = torch.from_numpy(meshProcessor.get_cell_types())
                file_path = f"{self.test_case_path}/cellTypes.pt"
                # Save the tensor to the file
                torch.save(cellTypes, file_path)

                # print(dataDictMeshes[i]["pointCoordinates"][dataDictMeshes[i]["boundaryPoints"]].shape)
                wall_x = dataDictMeshes[i]["pointCoordinates"][dataDictMeshes[i]["boundaryPoints"]][:,0]
                wall_y = dataDictMeshes[i]["pointCoordinates"][dataDictMeshes[i]["boundaryPoints"]][:,1]
                field_x = dataDictMeshes[i]["cellCenters"][:,0]
                field_y = dataDictMeshes[i]["cellCenters"][:,1]

                cellWeights = torch.from_numpy(meshProcessor.get_cell_weights())

                file_path = f"{self.test_case_path}/cellWeights.pt"
                torch.save(cellWeights, file_path)

            # print("fdasfdsaf",meshProcessor.cellCenters.shape)
            dataDictMeshes[i]["nNodesPrimalMesh"] = len(dataDictMeshes[i]["pointCoordinates"])
            dataDictMeshes[i]["cellPoints"] = meshProcessor.cellPoints
            dataDictMeshes[i]["cellFaces"] = meshProcessor.cellFaces
            dataDictMeshes[i]["faceCells"] = meshProcessor.faceCells

            dataDictMeshes[i]["primalMesh"] = {}
            primalmesh = meshProcessor.facePoints
            dataDictMeshes[i]["primalMesh"]["sources"] = primalmesh[:, 0]
            dataDictMeshes[i]["primalMesh"]["targets"] = primalmesh[:, 1]
            dataDictMeshes[i]["centerPoints"] = {}
            dataDictMeshes[i]["centerPoints"]["sources"], dataDictMeshes[i]["centerPoints"]["targets"] = meshProcessor.get_sources_targets_center_point()
            dataDictMeshes[i]["dualMesh"] = {}
            dataDictMeshes[i]["dualMesh"]["sources"], dataDictMeshes[i]["dualMesh"]["targets"] = meshProcessor.get_sources_targets_dual_mesh()
            dataDictMeshes[i]["pointPoints"] = {}
            dataDictMeshes[i]["pointPoints"]["sources"], dataDictMeshes[i]["pointPoints"]["targets"], dataDictMeshes[i]["pointPoints"]["faces"] = meshProcessor.get_pointPoints()

            dataDictMeshes[i]["orderedPolygonList"] = meshProcessor.get_polygons()

            dataDictMeshes[i]["boundaryCells"] = meshProcessor.get_boundary_cells()
            dataDictMeshes[i]["boundaryCells"] = meshProcessor.get_boundary_cells()

            dataDictMeshes[i]["boundaryFaces"] = meshProcessor.get_boundary_faces()
            dataDictMeshes[i]["NboundaryFaces"] = len(dataDictMeshes[i]["boundaryFaces"])

            # print("a",dataDictMeshes[i]["NboundaryFaces"])

            if self.faceToCell:
                dataDictMeshes[i]["faceCenters"] = {}
                dataDictMeshes[i]["faceCenters"]["sources"], dataDictMeshes[i]["faceCenters"]["targets"]  = meshProcessor.get_sources_targets_face_center()

            if self.faceToPoint:
                dataDictMeshes[i]["facePoints"] = {}
                dataDictMeshes[i]["facePoints"]["sources"], dataDictMeshes[i]["facePoints"]["targets"]  = meshProcessor.get_sources_targets_face_point()
                dataDictMeshes[i]["faceCoordinates_from_points"] = meshProcessor.get_face_centers()
                dataDictMeshes[i]["faceCoordinates_from_cells"] = meshProcessor.get_face_centers_from_cells()
                dataDictMeshes[i]["Nfaces"] = np.max(dataDictMeshes[i]["facePoints"]["sources"])+1

                # print(np.max(dataDictMeshes[i]["facePoints"]["sources"]))
            if self.faceToEdgeCenters:
                dataDictMeshes[i]["faceEdgeCenters"] = {}
                dataDictMeshes[i]["faceEdgeCenters"]["sources"], dataDictMeshes[i]["faceEdgeCenters"]["targets"] = meshProcessor.get_sources_targets_face_edgeCenters()

            if no_layers:
                break

        file_path = f"{self.test_case_path}/dataDictMeshes.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(dataDictMeshes, file)

if __name__ == '__main__':
    import sys
    import gmsh
    import os
    import shutil
    import subprocess
    from _GetGraphData import *
    from generateMeshPerlinDataset import *
    from generateMeshCFDDataset import *
    import pickle

    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_big"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"

    #
    # test_case_path = "/home/justinbrusche/openfoam_tutorial/cyli_005"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"
    #
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p2"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/base_case"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"
    #
    test_case_path = "/home/justinbrusche/test_cases/cylinder"
    # test_case_path = "/home/justinbrusche/test_cases/cylinder_0"

    # test_case_path = "/home/justinbrusche/test_cases/cylinder_fast"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/big_mesh"

    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"

    # base_path = "/home/justinbrusche/dataVaryingMeshes"
    # base_path ="/home/justinbrusche/gnn_openfoam/test_case_step_3"
    # base_path ="/home/justinbrusche/gnn_openfoam/test_case_step_3_p2"
    # base_path ="/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"

    # test_case_path = "/home/justinbrusche/openfoam_tutorial/cylinder_20"

    objects_list = [{"type":"circle", "x":0.3, "y": 0.5, "r": 0.01, "n":"ncyl"},
                    {"type":"circle", "x":0.7, "y": 0.5, "r": 0.01, "n":"ncyl"}]

    box_list = [{"x1": 7, "x2": 26, "y1": 7, "y2": 9, "nx": "nx_box2", "ny": "ny_box2", "objects": [0]},
                    ]

    mesh_name = "mesh_unstructured_square"
    # mesh_name = "custom_geo_file"
    # mesh_name = "custom_geo_file_specific"

    mesh_name = "generatedMesh"


    variable_dict_variant = {
        "ncyl": 200,
        "ninlet": 32,
        "ntop": 32,
        "nbot": 32,
        "noutlet": 32,
        "nxbox": 120,
        "nybox": 16
    }

    variable_dict_invariant = {
                            "y": 16,
                            "x": 28,
                            "cell_distribution_type": ["Progression", "Progression", "Progression", "Progression"],
                            "cell_distribution_value": [1, 1, 1, 1],
    }

    variable_dict_variant = {"nx": 40,
                             "ny": 40,
                            "ncyl": 330,
                            "nx_box": 200,
                            "ny_box": 25,
                             }

    variable_dict_invariant = {
                            "y": 16,
                            "x": 28,
                     }

    objects_list = [{"type":"circle", "x":8, "y": 8, "r": 0.5, "n":"ncyl"}]

    a = SetupOpenfoam(test_case_path,test_case_path,mesh_name,4,4,faceToCell=True,faceToPoint=True,faceToEdgeCenters=True)
    a.setup_source(openfoam_source_path)
    a.runWriteMeshData()
    # a.prepareFiles(openfoam_source_path)
    # a.prepareFilesWithObject(openfoam_source_path,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=True,box_list=None)
    # a.prepareFilesWithObject(openfoam_source_path,variable_dict_variant,variable_dict_invariant,objects_list,moreObjects=False)

    a.getMeshData()

    # a = SetupOpenfoam(test_case_path,mesh_name,5,64,faceToCell=True,faceToPoint=True,faceToEdgeCenters=False)
    # a.setup_source(openfoam_source_path)
    # a.runWriteMeshData()
    # a.getMeshData(no_layers=True)
