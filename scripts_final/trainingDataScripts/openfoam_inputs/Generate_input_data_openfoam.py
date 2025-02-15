import matplotlib.pyplot as plt

from trainingDataScripts.openfoam_inputs.Generate_perlin_noise_field import generate_perlin_noise_field
import os
import pickle
import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
#
# def plot_2d(fig, ax, x_coords, y_coords, noise_values, title):
#     x_coords = np.array(x_coords)
#     y_coords = np.array(y_coords)
#     noise_values = np.array(noise_values)
#
#     # Create a grid for the 2D plot
#     xi = np.linspace(x_coords.min(), x_coords.max(), 92)
#     yi = np.linspace(y_coords.min(), y_coords.max(), 92)
#     xi, yi = np.meshgrid(xi, yi)
#
#     # Interpolate noise values to fit the grid
#     zi = griddata((x_coords, y_coords), noise_values, (xi, yi), method='cubic')
#
#     # Create a contourf plot
#     contour = ax.contourf(xi, yi, zi, levels=100, cmap='viridis')
#
#     # Add a color bar
#     fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5, label='Noise Value')
#
#     # Set labels
#     ax.set_xlabel('X Coordinates')
#     ax.set_ylabel('Y Coordinates')
#     ax.set_title(title)

def rotate_90_degrees_clockwise(x, y, center_x, center_y):
    # Translate points to the origin (center of domain)
    x_translated = x - center_x
    y_translated = y - center_y

    # Apply 90 degrees clockwise rotation
    x_rotated = y_translated
    y_rotated = -x_translated

    # Translate points back to the original domain
    x_final = x_rotated + center_x
    y_final = y_rotated + center_y

    return x_final, y_final


# sides_list = ["Left", "Upper", "Right", "Lower"]

def get_fields(center_domain,x_coords, y_coords, x_tab_face_centers, y_tab_face_centers,sides_list, octaves_list, persistences, lacunarities, max_seed, filename, scale=None, scale_range=None, variable_scale=False,log_scale=False,include_randomness=False):
    # print(x_coords, y_coords)
    # print("1234",x_tab_face_centers, y_tab_face_centers)
    x_copy = x_coords + 0
    y_copy = y_coords + 0

    dx = 0.001

    data_dict_field = {}
    data_dict_bc_value = {}
    data_dict_bc_gradient = {}
    for iRotation in range(4):

        data_dict_field[iRotation] = {}
        data_dict_bc_value[iRotation] = {}
        data_dict_bc_gradient[iRotation] = {}
        print("iRotation", iRotation)
        for i in range(len(octaves_list)):
            print("i", i)

            data_dict_field[iRotation][i] = []
            data_dict_bc_value[iRotation][i] = {}
            data_dict_bc_gradient[iRotation][i] = {}

            for j in range(len(sides_list)):
                data_dict_bc_value[iRotation][i][sides_list[j]] = []
                data_dict_bc_gradient[iRotation][i][sides_list[j]] = []

            for j in range(max_seed):
                if variable_scale:
                    scale = np.random.uniform(scale_range[0], scale_range[1])
                    if log_scale:
                        scale = 10**(scale)
                    # print("scale", scale)

                if include_randomness:
                    persistence = persistences[i] * np.random.uniform(0.3,2)
                    lacunarity = lacunarities[i] +  np.random.uniform(-1,1)
                else:
                    persistence = persistences[i] + 0
                    lacunarity = lacunarities[i] + 0

                field = generate_perlin_noise_field(x_coords, y_coords, octaves_list[i],
                                                               persistence, lacunarity, j, scale)

                field_avg = np.mean(field)
                field_std = np.std(field)
                data_dict_field[iRotation][i].append((field-field_avg)/field_std)

                for k in range(len(sides_list)):
                    value = generate_perlin_noise_field(x_tab_face_centers[k], y_tab_face_centers[k], octaves_list[i],persistence, lacunarity, j, scale)
                    if sides_list[(k+iRotation)%4] == "Left" or sides_list[(k+iRotation)%4] == "Right":

                        value_offset = generate_perlin_noise_field(x_tab_face_centers[k]+dx, y_tab_face_centers[k], octaves_list[i],persistence, lacunarity, j, scale)
                    else:
                        value_offset = generate_perlin_noise_field(x_tab_face_centers[k], y_tab_face_centers[k]+dx, octaves_list[i],persistence, lacunarity, j, scale)

                    data_dict_bc_value[iRotation][i][sides_list[k]].append((value - field_avg) / field_std)
                    data_dict_bc_gradient[iRotation][i][sides_list[k]].append(((value - value_offset)/dx) / field_std)

        x_coords, y_coords = rotate_90_degrees_clockwise(x_coords, y_coords, center_domain[0], center_domain[1])
        for i in range(4):
            x_tab_face_centers[i], y_tab_face_centers[i] = rotate_90_degrees_clockwise(x_tab_face_centers[i], y_tab_face_centers[i], center_domain[0], center_domain[1])

        # print("5678", x_tab_face_centers, y_tab_face_centers)
        # break

    filename_field = filename+ "_field.pkl"
    filename_bc_value = filename + "_bc_value.pkl"
    filename_bc_gradient = filename + "_bc_gradient.pkl"

    directory = os.path.dirname(filename_field)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename_field, 'wb') as file:
        pickle.dump(data_dict_field, file)

    print(f"1D data dictionary saved to {filename_field}")

    directory = os.path.dirname(filename_bc_value)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename_bc_value, 'wb') as file:
        pickle.dump(data_dict_bc_value, file)

    print(f"1D data dictionary saved to {filename_bc_value}")

    directory = os.path.dirname(filename_bc_gradient)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename_bc_gradient, 'wb') as file:
        pickle.dump(data_dict_bc_gradient, file)

    print(f"1D data dictionary saved to {filename_bc_gradient}")

def getFieldDirect(center_domain,x_coords, y_coords, x_tab_face_centers_boundaries, y_tab_face_centers_boundaries, x_tab_face_centers, y_tab_face_centers, scale_range, octaves_list, persistences, lacunarities,smoothness,orientation,p_seed,tau_seed,p_BC_type,sides_list,sides_list_boundaries):
    dx = 0.001
    # print(orientation)

    # plt.figure()
    # plt.scatter(x_coords[:50], y_coords[:50])
    # plt.scatter(x_tab_face_centers_boundaries[0][:15], y_tab_face_centers_boundaries[0][:15])
    # plt.scatter(x_tab_face_centers_boundaries[1][:15], y_tab_face_centers_boundaries[1][:15])
    # plt.scatter(x_tab_face_centers_boundaries[2][:15], y_tab_face_centers_boundaries[2][:15])
    # plt.scatter(x_tab_face_centers_boundaries[3][:15], y_tab_face_centers_boundaries[3][:15])

    for i in range(orientation):
        # print(i, len(sides_list_boundaries),len(sides_list))
        x_coords, y_coords = rotate_90_degrees_clockwise(x_coords, y_coords, center_domain[0], center_domain[1])
        for j in range(len(sides_list_boundaries)):
            x_tab_face_centers_boundaries[j], y_tab_face_centers_boundaries[j] = rotate_90_degrees_clockwise(x_tab_face_centers_boundaries[j], y_tab_face_centers_boundaries[j], center_domain[0], center_domain[1])

        for j in range(len(sides_list)):
            x_tab_face_centers[j], y_tab_face_centers[j] = rotate_90_degrees_clockwise(x_tab_face_centers[j], y_tab_face_centers[j], center_domain[0], center_domain[1])


    # plt.figure()
    # plt.scatter(x_coords[:50], y_coords[:50])
    # plt.scatter(x_tab_face_centers_boundaries[0][:15], y_tab_face_centers_boundaries[0][:15])
    # plt.scatter(x_tab_face_centers_boundaries[1][:15], y_tab_face_centers_boundaries[1][:15])
    # plt.scatter(x_tab_face_centers_boundaries[2][:15], y_tab_face_centers_boundaries[2][:15])
    # plt.scatter(x_tab_face_centers_boundaries[3][:15], y_tab_face_centers_boundaries[3][:15])
    # plt.show()

    persistence = persistences[smoothness] * np.random.uniform(0.3, 2)
    lacunarity = lacunarities[smoothness] + np.random.uniform(-1, 1)
    scale = np.random.uniform(scale_range[0], scale_range[1])

    p_field = generate_perlin_noise_field(x_coords, y_coords, octaves_list[smoothness],
                                        persistence, lacunarity, p_seed, scale)

    p_field_avg = np.mean(p_field)
    p_field_std = np.std(p_field)

    p_field = (p_field - p_field_avg) / p_field_std

    p_boundary_dict = {}
    for i in range(4):
        value = generate_perlin_noise_field(x_tab_face_centers_boundaries[i], y_tab_face_centers_boundaries[i], octaves_list[smoothness], persistence,
                                            lacunarity, p_seed, scale)
        # print(octaves_list[smoothness], persistence,
        #                                     lacunarity, p_seed, scale)
        # if i == 0:
        #     print("value",value)
        # print("x_tab_face_centers_boundaries",x_tab_face_centers_boundaries[i])
        # print("y_tab_face_centers_boundaries",y_tab_face_centers_boundaries[i])
        # print(sides_list_boundaries)
        # print(sides_list_boundaries)
        if p_BC_type[i] == 1:
            if sides_list_boundaries[(i + orientation) % 4] == "Left" or sides_list_boundaries[(i + orientation) % 4] == "Right":

                value_offset = generate_perlin_noise_field(x_tab_face_centers_boundaries[i] + dx, y_tab_face_centers_boundaries[i],
                                                           octaves_list[smoothness], persistence, lacunarity, p_seed, scale)
            else:
                value_offset = generate_perlin_noise_field(x_tab_face_centers_boundaries[i], y_tab_face_centers_boundaries[i] + dx,
                                                           octaves_list[smoothness], persistence, lacunarity, p_seed, scale)
            p_boundary_dict[sides_list_boundaries[i]] = ((value - value_offset)/dx) / p_field_std
        else:
            p_boundary_dict[sides_list_boundaries[i]] = (value - p_field_avg) / p_field_std

    tau_field = generate_perlin_noise_field(x_coords, y_coords, octaves_list[smoothness],
                                        persistence, lacunarity, tau_seed, scale)

    tau_field_avg = np.mean(tau_field)
    tau_field_std = np.std(tau_field)

    tau_field = (tau_field - tau_field_avg) / tau_field_std

    tau_boundary_dict = {}
    for i in range(len(sides_list)):
        value = generate_perlin_noise_field(x_tab_face_centers[i], y_tab_face_centers[i], octaves_list[smoothness], persistence,
                                            lacunarity, tau_seed, scale)

        tau_boundary_dict[sides_list[i]] = (value - tau_field_avg) / tau_field_std

    return p_field, p_boundary_dict, tau_field, tau_boundary_dict


if __name__ == "__main__":
    pass