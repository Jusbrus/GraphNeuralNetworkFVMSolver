from openfoam_inputs.Generate_input_data_openfoam import *
from mesh_scripts.Get_mesh_attributes import *

test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_big"
test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p4"
test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"

file_path = f"{test_case_path}/dataDictMeshes.pkl"
with open(file_path, 'rb') as file:
    dataDictMeshes = pickle.load(file)

# test_case_path = "/home/justinbrusche/test_custom_solver_cases/explicit_small_poissonfoam"
#
# field_file = 'data_input_openfoam/fields_test.pkl'
# bc_file = 'data_input_openfoam/bc_test.pkl'

# test_case_path = "/home/justinbrusche/test_custom_solver_cases/poissonFoam_case"

# file_name = 'data_input_openfoam/data_bigger'
# file_name = 'data_input_openfoam/data_Cylinder'
# file_name = 'data_input_openfoam/data_Cylinder_p4'
# file_name = 'data_input_openfoam/data_Cylinder_var'
file_name = 'data_input_openfoam/data_Cylinder_big_var'
file_name = 'data_input_openfoam/data_large_Cylinder'


print(dataDictMeshes[0].keys())
print(dataDictMeshes[0]['cellCenters'])

test_case_path = test_case_path + "/level0"

centers_x = dataDictMeshes[0]['cellCenters'][:,0]
centers_y = dataDictMeshes[0]['cellCenters'][:,1]

# centers_x, centers_y = get_cell_centers(test_case_path)

# center_domain = [0.5,0.5]
center_domain = [0, 0]
# sides_list = ["Left", "Upper", "Right", "Lower"]

sides_list = dataDictMeshes[0]["boundaryNames"]
print(sides_list)

x_tab_face_centers, y_tab_face_centers = get_Face_centers(test_case_path,sides_list)

# x_tab_face_centers = dataDictMeshes[0]['faceCenters'][:,0]
# y_tab_face_centers = dataDictMeshes[0]['faceCenters'][:,1]


octaves_list = [4, 8, 12, 14, 14, 18]
persistences = [0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
lacunarities = [2, 2.2, 2.2, 2.2, 2.5, 3.5]

octaves_list = [1, 2, 4, 6, 8,10]
persistences = [0.3, 0.45, 0.6, 0.7, 0.8,0.9]
lacunarities = [2, 2.2, 2.2, 2.2, 2.2, 3.3]

# octaves_list = [6]
# persistences = [0.3]
# lacunarities = [2]
max_seed = 800
scale = 5
scale_range = [0.1,28]
# scale_range = [-1,0.6989700043]

# get_fields(center_domain,centers_x, centers_y, x_tab_face_centers, y_tab_face_centers,sides_list, octaves_list, persistences, lacunarities,
#            max_seed ,file_name, scale=scale)

get_fields(center_domain,centers_x, centers_y, x_tab_face_centers, y_tab_face_centers,sides_list, octaves_list, persistences, lacunarities,
           max_seed ,file_name, scale_range=scale_range,variable_scale=True,log_scale=False,include_randomness=True)


