import numpy as np
import copy
def getMeshes(n):
    objectListList = []
    boxListList = []
    variable_dict_variant_list = []
    variable_dict_invariant_list =[]
    containsObject = np.full(n, True)
    n = n-3

    variable_dict_variant_base = {"nx": 8,
                                  "ny": 8,
                                  "ncyl": 7,
                                  "nx_box": 4,
                                  "ny_box": 4,
                                  }

    variable_dict_invariant_base = {
        "y": 16,
        "x": 16,
        "cell_distribution_type": ["Progression", "Progression", "Progression", "Progression"],
        "cell_distribution_value": [1, 1, 1, 1],
    }

    # without_object
    containsObject[:3] = False
    xTabNoObject = np.array([4, 4, 8])
    yTabNoObject = np.array([4, 4, 4])
    nxTabNoObject = np.array([64, 96, 128])
    nyTabNoObject = np.array([64, 96, 64])


    for i in range(3):
        objectListList.append([])
        boxListList.append([])

        variable_dict_invariant_base["x"] = float(xTabNoObject[i])
        variable_dict_invariant_base["y"] = float(yTabNoObject[i])

        variable_dict_variant_base["nx"] = int(nxTabNoObject[i])
        variable_dict_variant_base["ny"] = int(nyTabNoObject[i])

        variable_dict_variant_list.append(copy.deepcopy(variable_dict_variant_base))
        variable_dict_invariant_list.append(copy.deepcopy(variable_dict_invariant_base))

    l_box_options = np.array([2, 3, 4, 5, 3, 2])

    # with object
    x_pos_options = np.linspace(0.3,0.7,6)
    y_pos_options =np.linspace(0.3,0.7,int(round(n/6)))

    l_box_options = np.array([2, 2, 2, 2, 2, 2])
    # l_box_options = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

    x_options = np.linspace(4, 8, n)
    y_options = np.array([4])

    alpha_options = np.linspace(-90, 80, n)
    ellips_t_options = np.linspace(0.05, 0.9, n)

    ny_options = np.linspace(32, 49, n)
    # ny_options = np.linspace(24,41,18)

    ncyl_options = np.linspace(120, 290, n)
    print(ncyl_options)
    box_n_ratio = 1 / 8

    i_list = np.arange(n)
    i_t_list = np.random.permutation(i_list)
    i_alpha_list = np.random.permutation(i_list)
    i_mesh_density_list = np.random.permutation(i_list)
    i_x = np.random.permutation(i_list)
    # print(i_x)

    for i in range(n):
        variable_dict_invariant_base["x"] = float(x_options[i_x[i]])
        variable_dict_invariant_base["y"] = float(y_options[0])

        variable_dict_variant_base["nx"] = int(
            np.round(x_options[i_x[i]] / y_options[0] * ny_options[i_mesh_density_list[i]]))
        variable_dict_variant_base["ny"] = int(ny_options[i_mesh_density_list[i]])
        variable_dict_variant_base["ncyl"] = int(ncyl_options[i_mesh_density_list[i]])

        i_pos_x = i % 6
        i_pos_y = int(np.floor(i / 6))

        x_object = x_pos_options[i_pos_x] * variable_dict_invariant_base["x"]
        y_object = y_pos_options[i_pos_y] * variable_dict_invariant_base["y"]

        objects_list = [{"type": "ellipse", "x": float(x_object), "y": float(y_object), "rx": 0.5,
                         "ry": float(ellips_t_options[i_t_list[i]] / 2), "alpha": float(alpha_options[i_alpha_list[i]]),
                         "n": "ncyl", "box": True}]
        box_list = [
            {"x1": float(x_object - l_box_options[i_pos_x] / 2), "x2": float(x_object + l_box_options[i_pos_x] / 2),
             "y1": float(y_object - l_box_options[i_pos_x] / 2), "y2": float(y_object + l_box_options[i_pos_x] / 2),
             "nx": "nx_box", "ny": "ny_box", "objects": [0]}]

        variable_dict_variant_base["nx_box"] = int(
            ncyl_options[i_mesh_density_list[i]] * box_n_ratio * l_box_options[i_pos_x] / 2)
        variable_dict_variant_base["ny_box"] = int(ncyl_options[i_mesh_density_list[i]] * box_n_ratio)

        objectListList.append(copy.deepcopy(objects_list))
        boxListList.append(copy.deepcopy(box_list))
        variable_dict_variant_list.append(copy.deepcopy(variable_dict_variant_base))
        variable_dict_invariant_list.append(copy.deepcopy(variable_dict_invariant_base))
        # factor = 8
        # for key in variable_dict_variant_base.keys():
        #     variable_dict_variant_base[key] = int(variable_dict_variant_base[key]/factor)

        # save_path = "/home/justinbrusche/dataVaryingMeshes/geo_file_random.geo"
        #
        # a = GenerateMeshMoreObjects(objects_list, box_list, variable_dict_variant_base, variable_dict_invariant_base,
        #                             save_path)
        # a.get_mesh()
    print(containsObject)
    print(variable_dict_variant_list)
    print(variable_dict_invariant_list)

    return containsObject, variable_dict_invariant_list, variable_dict_variant_list, boxListList, objectListList


if __name__ == '__main__':
    getMeshes(39)