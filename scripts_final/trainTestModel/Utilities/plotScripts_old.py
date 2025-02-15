import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import yaml
import os

def plot_3d(fig, ax, x_coords, y_coords, noise_values, title, zlim=None):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    noise_values = np.array(noise_values)

    # Create a grid for the 3D plot
    xi = np.linspace(x_coords.min(), x_coords.max(), 64)
    yi = np.linspace(y_coords.min(), y_coords.max(), 64)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate noise values to fit the grid
    zi = griddata((x_coords, y_coords), noise_values, (xi, yi), method='cubic')

    # Ensure zi is 2D
    if zi.ndim == 3:
        zi = zi[:, :, 0]

    # Create a surface plot
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Noise Value')

    # Set labels
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Noise Value')
    ax.set_title(title)

    # Set z-axis limits if provided
    if zlim:
        ax.set_zlim(zlim)

def plot_2d(fig, ax, x_coords, y_coords, noise_values, title, zlim=None,show_bar=True):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    noise_values = np.array(noise_values)

    # Clip the noise values to the 1st and 99th percentiles
    lower_bound = np.percentile(noise_values, 1)
    upper_bound = np.percentile(noise_values, 99)
    noise_values_clipped = np.clip(noise_values, lower_bound, upper_bound)

    # Create a grid for the 2D plot
    xi = np.linspace(x_coords.min(), x_coords.max(), 92)
    yi = np.linspace(y_coords.min(), y_coords.max(), 92)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate noise values to fit the grid
    zi = griddata((x_coords, y_coords), noise_values_clipped, (xi, yi), method='cubic')

    # Ensure zi is 2D
    if zi.ndim == 3:
        zi = zi[:, :, 0]

    # Create a contour plot
    contour = ax.contourf(xi, yi, zi, levels=100, cmap='viridis')
    if show_bar:
        fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5, label='Noise Value')

    # Set labels
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title(title)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

def plot_mesh_triangles_3d(fig, ax, cellpoints, x, y, noise_values, title, zlim=None, show_bar=True):
    # Flatten the noise values
    noise_values = noise_values.flatten()

    # Clip the noise values to the 1st and 99th percentiles for better visualization
    lower_bound = np.percentile(noise_values, 1)
    upper_bound = np.percentile(noise_values, 99)
    noise_values_clipped = np.clip(noise_values, lower_bound, upper_bound)
    noise_values_clipped = noise_values

    # Create an array of 3D triangle vertices using the indices from cellpoints
    triangles = np.array([[(x[i], y[i], noise_values_clipped[i]) for i in cell] for cell in cellpoints])

    # Create a Poly3DCollection to represent the 3D triangles
    collection = Poly3DCollection(triangles, facecolors=plt.cm.viridis(noise_values_clipped), linewidths=0.5, edgecolors='k')

    # Add the collection to the axes
    ax.add_collection3d(collection)

    # Auto scale the plot limits based on the data
    ax.auto_scale_xyz([x.min(), x.max()], [y.min(), y.max()], [noise_values_clipped.min(), noise_values_clipped.max()])

    # Set labels and title
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Noise Value (Z Axis)')
    ax.set_title(title)

    # Set z-axis limits if provided (for 3D consistency)
    if zlim:
        ax.set_zlim(zlim)

    # Add a color bar for noise values
    if show_bar:
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(noise_values_clipped)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Noise Value')



def plot_mesh_triangles(fig, ax, cellpoints, x,y, noise_values, title, zlim=None,show_bar=True):
    # Unpack meshpoints into x and y coordinates
    # print("ds",noise_values.shape)

    noise_values = noise_values.flatten()
    # Clip the noise values to the 1st and 99th percentiles
    lower_bound = np.percentile(noise_values, 1)
    upper_bound = np.percentile(noise_values, 99)
    noise_values_clipped = np.clip(noise_values, lower_bound, upper_bound)
    noise_values_clipped = noise_values

    # Create an array of triangle vertices using the indices from cellpoints
    triangles = np.array([[(x[i], y[i]) for i in cell] for cell in cellpoints])

    # Create a collection of polygons to represent the triangles
    collection = PolyCollection(triangles, array=noise_values_clipped, cmap='viridis', edgecolors='none')

    # Add the collection to the axes
    ax.add_collection(collection)

    # Auto scale the plot limits based on the data
    ax.autoscale()
    ax.set_aspect('equal')

    # Add a color bar
    if show_bar:
        fig.colorbar(collection, ax=ax, shrink=0.5, aspect=5, label='Noise Value')

    # Set labels and title
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title(title)

    # Set z-axis limits if provided (for 3D consistency)
    if zlim:
        ax.set_clim(zlim)


from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def plot_mesh_triangles_pool(fig, ax, cellpoints, x, y, noise_values, title, zlim=None, show_bar=True):
    # Define a custom list of colors to cycle through
    color_cycle = [
        'b', 'r', 'g', 'c', 'm', 'y', 'k',  # Blue, Red, Green, Cyan, Magenta, Yellow, Black
        'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy',  # Additional distinct colors
        'lime', 'teal', 'aqua', 'maroon', 'silver', 'gold', 'indigo'  # Even more colors
    ]

    # Flatten noise values (assuming you need it for something else)
    noise_values = noise_values.flatten().astype(int)

    # Rank the noise values and assign them to bins corresponding to color_cycle
    num_colors = len(color_cycle)
    # noise_ranks = np.argsort(np.argsort(noise_values))  # Ranking the noise values
    # color_indices = noise_ranks % num_colors  # Map ranks to color indices

    # Assign colors based on the rank of each noise value
    colors = [color_cycle[i%num_colors] for i in noise_values]
    print(colors)

    # Create an array of triangle vertices using the indices from cellpoints
    triangles = np.array([[(x[i], y[i]) for i in cell] for cell in cellpoints])

    # Create a collection of polygons to represent the triangles
    # collection = PolyCollection(triangles, facecolors=colors, edgecolors='k')
    collection = PolyCollection(triangles, facecolors=colors)

    # Add the collection to the axes
    ax.add_collection(collection)

    # Auto scale the plot limits based on the data
    ax.autoscale()
    ax.set_aspect('equal')

    # Optional: Add a color legend instead of a color bar
    if show_bar:
        # Display a color legend
        handles = [plt.Line2D([0], [0], color=color, lw=4) for color in color_cycle]
        ax.legend(handles, [f'Color {i + 1}' for i in range(len(color_cycle))], title="Color Legend")

    # Set labels and title
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title(title)

    # Set z-axis limits if provided (for 3D consistency)
    if zlim:
        ax.set_clim(zlim)





def plot_predicted_vs_groundtruth(x_coords, y_coords, predicted, groundtruth, title_pred='Predicted Pressure', title_gt='Ground Truth Pressure', plot_type='2d',cellPoints = None,x_mesh = None,y_mesh = None):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    predicted = np.array(predicted)
    groundtruth = np.array(groundtruth)

    fig = plt.figure(figsize=(12, 6))

    zlims = []
    zmin = min(np.min(groundtruth), np.min(predicted), np.min(groundtruth-predicted))
    zmax = max(np.max(groundtruth), np.max(predicted), np.max(groundtruth-predicted))
    zlim = [zmin, zmax]


    if plot_type == '3d':
        ax_pred = fig.add_subplot(1, 3, 1, projection='3d')
        plot_3d(fig, ax_pred, x_coords, y_coords, predicted, title=title_pred, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 2, projection='3d')
        plot_3d(fig, ax_gt, x_coords, y_coords, groundtruth, title=title_gt, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 3, projection='3d')
        plot_3d(fig, ax_gt, x_coords, y_coords, groundtruth-predicted, title=title_gt, zlim=zlim)
    if plot_type == '2d':
        ax_pred = fig.add_subplot(1, 3, 1)
        plot_2d(fig, ax_pred, x_coords, y_coords, predicted, title=title_pred, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 2)
        plot_2d(fig, ax_gt, x_coords, y_coords, groundtruth, title=title_gt, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 3)
        plot_2d(fig, ax_gt, x_coords, y_coords, groundtruth - predicted, title="diff", zlim=zlim)
    else:
        ax_pred = fig.add_subplot(1, 3, 1)
        plot_mesh_triangles(fig, ax_pred, cellPoints,x_mesh,y_mesh, predicted.flatten(), title=title_pred, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 2)
        plot_mesh_triangles(fig, ax_gt, cellPoints,x_mesh,y_mesh, groundtruth.flatten(), title=title_gt, zlim=zlim)

        ax_gt = fig.add_subplot(1, 3, 3)
        plot_mesh_triangles(fig, ax_gt, cellPoints,x_mesh,y_mesh, (groundtruth - predicted).flatten(), title="diff", zlim=zlim)

    plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"




    config_file = 'config_files/trainConfig_FVMSimpleMedium.yaml'

    modeldir_dir = "/home/justinbrusche/modeldirs_step_2/MediumPointType"
    modeldir_dir = "/home/justinbrusche/modeldirs_step_2/MediumPointTypeBC0"

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/SimpleMedium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_Medium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_FullScale_step_2"

    #
    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_Medium_step_2"


    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_modi_FullScale_step_2_version2"

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_modi_FullScale_step_2_version2_conv"

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/BC1Gradient0_modi_FullScale_step_2_lessLayers"
    # modeldir_dir = '/home/justinbrusche/modeldirs_wrong/modeldir_ConvMeanSmall'
    # modeldir_dir = '/home/justinbrusche/modeldir_MeshGraphNetCNNAddSmallNormPointTypeBC0'

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_conv_medium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p4_conv_medium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p4_conv_small"

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_var_conv_medium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p_mag"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p_mag_scale_model"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p_mag_scale_model_hyperparam"

    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p_mag_verySmall_scale_model_hyperparam"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_p_mag_veryVerySmall_scale_model_hyperparam"

    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/step_3_Noconv_medium"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/verySmall_cylinder"
    modeldir_dir = "/home/justinbrusche/modeldirs_FVM/veryVerySmall_cylinder"

    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_cylinder"
    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_cylinder_fast"

    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_big_var"
    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_size"

    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_big_var_scale"


    # modeldir_dir = "/home/justinbrusche/modeldirs_FVM/medium_cylinder_split"


    epoch = 24
    # epoch = 38
    epoch = 104
    epoch = 168
    epoch = 76
    epoch = 6
    epoch = 36
    epoch = 68

    i_start = 20

    n_columns = 3
    n_rows = 3

    # epoch = 86
    #
    # epoch = 80

    plot_type = '3d_triangle'
    plot_type = '3dm'
    # plot_type = '2d'

    # plot_type = '3d'

    test_plots = True
    test_plots = False

    # with open(config_file, 'r') as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)

    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    x_coords = dataDictMeshes[0]["cellCenters"][:, 0]
    y_coords = dataDictMeshes[0]["cellCenters"][:, 1]

    cellPoints = dataDictMeshes[0]["cellPoints"]
    x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
    y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]

    if test_plots:
        outP_save_path = os.path.join(modeldir_dir, f'test_outP_epoch_{epoch}.npy')
        p_save_path = os.path.join(modeldir_dir, f'test_p_epoch_{epoch}.npy')
        pEqnSource_save_path = os.path.join(modeldir_dir, f'test_pEqnSource_epoch_{epoch}.npy')
    else:
        outP_save_path = os.path.join(modeldir_dir, f'outP_epoch_{epoch}.npy')
        p_save_path = os.path.join(modeldir_dir, f'p_epoch_{epoch}.npy')
        pEqnSource_save_path = os.path.join(modeldir_dir, f'pEqnSource_epoch_{epoch}.npy')

    # Load the pressure data
    outP = np.load(outP_save_path)
    # print(outP.shape)
    # print(np.mean(outP, axis=(1,2)))
    p = np.load(p_save_path)
    print(np.std(p, axis=(1)))

    # print(np.mean(p, axis=(1))-np.mean(outP, axis=(1,2)))

    pEqnSource = np.load(pEqnSource_save_path)
    print(np.shape(p),np.shape(pEqnSource))
    print(np.max(abs(pEqnSource), axis=(1)))
    print(np.max(abs(p), axis=(1)))

    # pEqnSource = np.where(abs(pEqnSource)>0.1,0,pEqnSource)


    # Ensure the pressures are in the first dimension
    pressures_pred = outP[i_start:]
    pressures_real = p[i_start:]

    # Create a 3x3 subplot
    fig = plt.figure(figsize=(24, 13))
    # Plot the predicted and real pressures

    # for i in range(n_columns):
    #     pressures_real[i] = pressures_real[i] - np.mean(pressures_real[i])
    #     pressures_pred[i] = pressures_pred[i] - np.mean(pressures_pred[i])

    # Determine z-axis limits for each column
    zlims = []
    for i in range(n_columns):
        zmin = min(np.min(pressures_real[i]), np.min(pressures_pred[i]), np.min(pressures_real[i]-pressures_pred[i][:,0]))
        zmax = max(np.max(pressures_real[i]), np.max(pressures_pred[i]), np.max(pressures_real[i]-pressures_pred[i][:,0]))
        zlims.append((zmin, zmax))

    for i in range(n_columns):
        print(np.mean(abs(pressures_real[i])),np.mean(abs(pressures_real[i]-pressures_pred[i][:, 0])))
        if plot_type == '3d':
            ax = fig.add_subplot(n_rows, n_columns, i + 1, projection='3d')
            plot_3d(fig, ax, x_coords, y_coords, pressures_real[i], title=f'Real Pressure {i+1} at Epoch {epoch}', zlim=zlims[i])
            ax = fig.add_subplot(n_rows, n_columns, i + n_columns + 1, projection='3d')
            plot_3d(fig, ax, x_coords, y_coords, pressures_pred[i], title=f'Predicted Pressure {i+1} at Epoch {epoch}', zlim=zlims[i])
            if n_rows > 2:
                ax = fig.add_subplot(n_rows, n_columns, i + 2 * n_columns + 1, projection='3d')
                plot_3d(fig, ax, x_coords, y_coords, pressures_real[i] - pressures_pred[i][:, 0], title=f'diff {i + 1} at Epoch {epoch}', zlim=zlims[i])
            if n_rows == 4:
                ax = fig.add_subplot(n_rows, n_columns, i + 3 * n_columns + 1, projection='3d')
                plot_3d(fig, ax, x_coords, y_coords, pEqnSource[i], title=f'pEqnSource {i + 1} at Epoch {epoch}', zlim=zlims[i])
        elif plot_type == '2d':
            ax = fig.add_subplot(n_rows, n_columns, i + 1)
            plot_2d(fig, ax, x_coords, y_coords, pressures_real[i], title=f'Real Pressure {i + 1} at Epoch {epoch}')
            ax = fig.add_subplot(n_rows, n_columns, i + n_columns + 1)
            plot_2d(fig, ax, x_coords, y_coords, pressures_pred[i], title=f'Predicted Pressure {i + 1} at Epoch {epoch}')
            if n_rows > 2:
                ax = fig.add_subplot(n_rows, n_columns, i + 2 * n_columns + 1)
                plot_2d(fig, ax, x_coords, y_coords, pressures_real[i] - pressures_pred[i][:, 0], title=f'diff {i + 1} at Epoch {epoch}')
            if n_rows == 4:
                ax = fig.add_subplot(n_rows, n_columns, i + 3 * n_columns + 1)
                plot_2d(fig, ax, x_coords, y_coords, pEqnSource[i], title=f'pEqnSource {i + 1} at Epoch {epoch}')

        elif plot_type == '3d_triangle':
            ax = fig.add_subplot(n_rows, n_columns, i + 1, projection='3d')
            plot_mesh_triangles_3d(fig, ax, cellPoints,x_mesh,y_mesh, pressures_real[i], title=f'Real Pressure {i + 1} at Epoch {epoch}')
            ax = fig.add_subplot(n_rows, n_columns, i + n_columns + 1, projection='3d')
            plot_mesh_triangles_3d(fig, ax, cellPoints,x_mesh,y_mesh, pressures_pred[i], title=f'Predicted Pressure {i + 1} at Epoch {epoch}')
            if n_rows > 2:
                ax = fig.add_subplot(n_rows, n_columns, i + 2 * n_columns + 1, projection='3d')
                plot_mesh_triangles_3d(fig, ax, cellPoints,x_mesh,y_mesh, pressures_real[i] - pressures_pred[i][:, 0], title=f'diff {i + 1} at Epoch {epoch}')
            if n_rows == 4:
                ax = fig.add_subplot(n_rows, n_columns, i + 3 * n_columns + 1, projection='3d')
                plot_mesh_triangles_3d(fig, ax, cellPoints,x_mesh,y_mesh, pEqnSource[i], title=f'pEqnSource {i + 1} at Epoch {epoch}')

        else:
            ax = fig.add_subplot(n_rows, n_columns, i + 1)
            plot_mesh_triangles(fig, ax, cellPoints,x_mesh,y_mesh, pressures_real[i], title=f'Real Pressure {i + 1} at Epoch {epoch}')
            ax = fig.add_subplot(n_rows, n_columns, i + n_columns + 1)
            plot_mesh_triangles(fig, ax, cellPoints,x_mesh,y_mesh, pressures_pred[i], title=f'Predicted Pressure {i + 1} at Epoch {epoch}')
            if n_rows > 2:
                ax = fig.add_subplot(n_rows, n_columns, i + 2 * n_columns + 1)
                plot_mesh_triangles(fig, ax, cellPoints,x_mesh,y_mesh, pressures_real[i] - pressures_pred[i][:, 0], title=f'diff {i + 1} at Epoch {epoch}')
            if n_rows == 4:
                ax = fig.add_subplot(n_rows, n_columns, i + 3 * n_columns + 1)
                plot_mesh_triangles(fig, ax, cellPoints,x_mesh,y_mesh, pEqnSource[i], title=f'pEqnSource {i + 1} at Epoch {epoch}')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.4)

    plt.tight_layout()
    plt.show()
