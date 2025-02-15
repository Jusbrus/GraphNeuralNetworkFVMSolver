
def arange_input_data(smooth_options,n_field_seeds,filename):
    ### PRESSURE
    # PRESSURE BC type
    n_repeat_BC_type = int((field_seeds*smooth_options*4) / 15) + 1
    BC_type_one_loop = np.array([[0,0,0,0],
                                 [0,0,0,1],
                                 [0,0,1,0],
                                 [0,0,1,1],
                                 [0,1,0,0],
                                 [0,1,0,1],
                                 [0,1,1,0],
                                 [0,1,1,1],
                                 [1,0,0,0],
                                 [1,0,0,1],
                                 [1,0,1,0],
                                 [1,0,1,1],
                                 [1,1,0,0],
                                 [1,1,0,1],
                                 [1,1,1,0]])

    p_BC_type_array = np.tile(BC_type_one_loop, (n_repeat_BC_type, 1))[:field_seeds*smooth_options*4]

    # p smoothness array
    p_smoothness_array = np.array([]).astype(int)
    p_smoothness_array_1 = np.tile(np.arange(smooth_options),int(field_seeds/smooth_options)+1)[:field_seeds].astype(int)
    for i in range(smooth_options*4):
        p_smoothness_array = np.append(p_smoothness_array, (p_smoothness_array_1+i)%6)

    seeds_array = np.tile(np.arange(n_field_seeds),smooth_options*4)

    orientation_array = np.repeat(np.arange(4),n_field_seeds*smooth_options)

    index_list = np.arange(n_field_seeds*smooth_options*4)
    index_list = np.random.permutation(index_list)

    data_dict = {}
    print(len(index_list))
    data_dict['index_list'] = index_list.astype(int)
    data_dict['smoothness'] = p_smoothness_array[index_list].astype(int)
    data_dict["orientation"] = orientation_array[index_list].astype(int)
    data_dict['p_BC_type'] = p_BC_type_array[index_list].astype(int)
    data_dict["p_seeds"] = seeds_array[index_list].astype(int)
    data_dict["tau_seeds"] = np.flip(seeds_array)[index_list].astype(int)


    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the 1D data dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(data_dict, file)


if __name__ == "__main__":
    import meshio
    import numpy as np
    import pickle
    import os
    import matplotlib.pyplot as plt


    field_seeds = 900
    smooth_options = 6


    arange_input_data(smooth_options, field_seeds, 'data_input_openfoam/seeds_dict.pkl')

    # arange_input_data(BC_seeds, BC_smooth_options, field_seeds, field_smooth_options, n_train, n_test,'data_input_openfoam/seeds_dict_test.pkl')



