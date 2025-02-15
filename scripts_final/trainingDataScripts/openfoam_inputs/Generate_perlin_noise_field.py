import noise
from scipy.interpolate import griddata
import numpy as np
import concurrent.futures
from matplotlib.collections import PolyCollection

def generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed, scale):
    noise_values = []
    for x, y in zip(x_coords, y_coords):
        noise_value = noise.pnoise2(x/scale,
                                    y/scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    base=seed)
        noise_values.append(noise_value)
    return np.array(noise_values)

# from opensimplex import OpenSimplex
# import numpy as np
#
# from perlin_noise import PerlinNoise
# import numpy as np


# def generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed, scale):
#     # Create a PerlinNoise object with the given seed and parameters
#     noise_generator = PerlinNoise(octaves=octaves, seed=seed)
#
#     noise_values = []
#     for x, y in zip(x_coords, y_coords):
#         # Generate noise for each coordinate, scaled accordingly
#         noise_value = noise_generator([x / scale, y / scale])
#         noise_values.append(noise_value)
#
#     return np.array(noise_values)


# def generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed, scale):
#     simplex = OpenSimplex(seed)  # Create an OpenSimplex object with the given seed
#
#     noise_values = []
#     for x, y in zip(x_coords, y_coords):
#         freq = 1.0
#         amplitude = 1.0
#         total_noise = 0.0
#         max_value = 0.0
#         for _ in range(octaves):
#             noise_value = simplex.noise2(x * freq / scale, y * freq / scale)
#             total_noise += noise_value * amplitude
#             max_value += amplitude
#             amplitude *= persistence
#             freq *= lacunarity
#
#         # Normalize the result
#         noise_values.append(total_noise / max_value)
#
#     return np.array(noise_values)


# def generate_perlin_noise_value(args):
#     x, y, octaves, persistence, lacunarity, seed = args
#     return noise.pnoise2(x,
#                          y,
#                          octaves=octaves,
#                          persistence=persistence,
#                          lacunarity=lacunarity,
#                          base=seed)
#
# def generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed, num_workers=None):
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#         args = [(x, y, octaves, persistence, lacunarity, seed) for x, y in zip(x_coords, y_coords)]
#         noise_values = list(executor.map(generate_perlin_noise_value, args))
#     return np.array(noise_values)

def plot_noise_field(x_coords, y_coords, noise_values):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    noise_values = np.array(noise_values)

    # Create a grid for the color plot
    xi = np.linspace(x_coords.min(), x_coords.max(), 256)
    yi = np.linspace(y_coords.min(), y_coords.max(), 256)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate noise values to fit the grid
    zi = griddata((x_coords, y_coords), noise_values, (xi, yi), method='cubic')

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xi, yi, zi, shading='auto', cmap='viridis')
    plt.colorbar(label='Perlin Noise Value')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Perlin Noise Field')
    plt.show()

def plot_mesh_triangles(fig, ax, cellpoints, x,y, noise_values, title, zlim=None,show_bar=True):
    # Unpack meshpoints into x and y coordinates
    # print("ds",noise_values.shape)
    label_size = 20
    title_size = 15
    legend_size = 20
    title_size = 25
    axis_size = 15

    noise_values = noise_values.flatten()
    # Clip the noise values to the 1st and 99th percentiles
    lower_bound = np.percentile(noise_values, 1)
    upper_bound = np.percentile(noise_values, 99)
    noise_values_clipped = np.clip(noise_values, lower_bound, upper_bound)

    # Create an array of triangle vertices using the indices from cellpoints
    triangles = np.array([[(x[i], y[i]) for i in cell] for cell in cellpoints])

    # Create a collection of polygons to represent the triangles
    collection = PolyCollection(triangles, array=noise_values_clipped, cmap='viridis',
                                )

    # Add the collection to the axes
    ax.add_collection(collection)

    # Auto scale the plot limits based on the data
    ax.autoscale()
    ax.set_aspect('equal')

    # Add a color bar
    if show_bar:
        fig.colorbar(collection, ax=ax, shrink=0.5, aspect=5, label='Noise Value [-]')

    # Set labels and title
    ax.set_xlabel('X [m]',fontsize=label_size)
    ax.set_ylabel('Y [m]',fontsize=label_size)
    ax.tick_params(axis='both', labelsize=axis_size)

    # ax.set_title(title)
    ax.set_xlim([0,5.5])
    ax.set_ylim([0,4])

    # Set z-axis limits if provided (for 3D consistency)
    if zlim:
        ax.set_clim(zlim)


def plot_noise_fields_triangle(x_coords, y_coords, octaves_list, persistences, lacunarities, seed,scale,cellPoints,x_mesh,y_mesh):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    num_fields = len(octaves_list)
    rows = 2
    cols = 3

    result_tab = []
    for i in range(min(num_fields, rows * cols)):
    # for i in range(1):

        octaves = octaves_list[i]
        persistence = persistences[i]
        lacunarity = lacunarities[i]
        noise_values = generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed,scale)
        noise_values = np.array(noise_values)

        fig = plt.figure(figsize=(15, 9))

        ax = fig.add_subplot(1, 1, 1)
        plot_mesh_triangles(fig, ax, cellPoints, x_mesh,y_mesh, noise_values, "data")
        plt.show()

def plot_noise_fields_triangles(x_coords, y_coords, octaves_list, persistences, lacunarities, seed,scale,cellPoints,x_mesh,y_mesh):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    num_fields = len(octaves_list)
    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    result_tab = []
    for i in range(min(num_fields, rows * cols)):
    # for i in range(1):

        octaves = octaves_list[i]
        persistence = persistences[i]
        lacunarity = lacunarities[i]
        noise_values = generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed,scale)
        noise_values = np.array(noise_values)

        # fig = plt.figure(figsize=(15, 9))

        # ax = fig.add_subplot(1, 1, 1)
        ax = axes[i]

        plot_mesh_triangles(fig, ax, cellPoints, x_mesh,y_mesh, noise_values/np.std(noise_values), f"Octaves: {octaves}, Persistance: {persistence}, Lacunarity: {lacunarity}",show_bar=False)

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.show()


def compare_fields(x_coords, y_coords, octaves_list, persistences, lacunarities,scale):
    result_tab = []

    for i in range(10000):
        octaves = octaves_list[0]
        persistence = persistences[0]
        lacunarity = lacunarities[0]
        noise_values = generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, i,scale)
        noise_values = np.array(noise_values)
        result_tab.append(noise_values)

    result_tab = np.array(result_tab)
    diff = abs(result_tab[1:] - result_tab[:-1])
    print(np.shape(diff))
    print(np.sum(diff, axis=1))
    plt.plot(np.sum(diff, axis=1),"o")
    plt.show()


def plot_noise_fields(x_coords, y_coords, octaves_list, persistences, lacunarities, seed,scale):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    num_fields = len(octaves_list)
    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(num_fields, rows * cols)):
        octaves = octaves_list[i]
        persistence = persistences[i]
        lacunarity = lacunarities[i]
        noise_values = generate_perlin_noise_field(x_coords, y_coords, octaves, persistence, lacunarity, seed,scale)
        noise_values = np.array(noise_values)

        # Create a grid for the color plot
        xi = np.linspace(x_coords.min(), x_coords.max(), 256)
        yi = np.linspace(y_coords.min(), y_coords.max(), 256)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate noise values to fit the grid
        zi = griddata((x_coords, y_coords), noise_values, (xi, yi), method='cubic')

        ax = axes[i]
        c = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='viridis')
        ax.set_title(f'Octaves: {octaves}, Persistence: {persistence}, Lacunarity: {lacunarity}')

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from trainingDataScripts.mesh_scripts.Get_mesh_attributes import *
    import pickle

    # script_dir = os.path.dirname(__file__)  # Current script directory
    # parent_dir = os.path.join(script_dir, os.pardir)  # Parent directory
    # module_dir = os.path.abspath(os.path.join(parent_dir, 'mesh_scripts'))

    case_path = "/home/justinbrusche/test_custom_solver_cases/poissonFoam_case"
    case_path = "/home/justinbrusche/test_custom_solver_cases/pooling_test"
    case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1/level0"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"
    test_case_path = "/home/justinbrusche/datasets/foundationalParameters/case_11"

    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    centers_x = dataDictMeshes[0]['cellCenters'][:, 0]
    centers_y = dataDictMeshes[0]['cellCenters'][:, 1]

    cellPoints = dataDictMeshes[0]["cellPoints"]
    x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
    y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]

    case_path = test_case_path + "/level0"

    centers_x, centers_y = get_cell_centers(case_path)

    # print(np.shape(centers_x))
    # print(np.shape(centers_x_2))
    # plt.plot(centers_y_2)
    # plt.plot(centers_y)
    # print(centers_x_2)

    scale = 4
    octaves = 10
    persistence = 0.4
    lacunarity = 3.0
    seed = 10

    # noise_result = generate_perlin_noise_field(centers_x, centers_y, octaves, persistence, lacunarity, seed)
    # print(noise_result)

    # plot_noise_field(centers_x, centers_y, noise_result)

    # octaves_list = [6, 10, 12, 14, 14]
    # persistences = [0.3, 0.45, 0.6, 0.7, 0.9]
    # lacunarities = [2, 2.2, 2.2, 2.2, 3.3]

    octaves_list = [1, 2, 4, 6, 8, 10]
    persistences = [0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
    lacunarities = [2, 2.2, 2.2, 2.2, 2.2, 3.3]

    # octaves_list = [1, 2, 4, 6, 8, 10]
    # persistences = [0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
    # lacunarities = [2, 2.2, 2.2, 2.2, 2.2, 3.3]
    # max_seed = 3

    # compare_fields(centers_x, centers_y, octaves_list, persistences, lacunarities, scale)

    # plot_noise_fields_triangle(centers_x, centers_y, octaves_list, persistences, lacunarities, seed,scale,cellPoints,x_mesh,y_mesh)
    plot_noise_fields_triangles(centers_x, centers_y, octaves_list, persistences, lacunarities, seed,scale,cellPoints,x_mesh,y_mesh)

    # plot_noise_fields(centers_x, centers_y, octaves_list, persistences, lacunarities, seed,scale)
