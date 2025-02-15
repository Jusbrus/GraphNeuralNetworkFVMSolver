import meshio
import numpy as np
import matplotlib.pyplot as plt

def get_cell_centers(path):
    # Load the text file into a numpy array
    file_path = path+ '/constant/polyMesh/cellCenters.txt'
    cell_centers = np.loadtxt(file_path)

    return cell_centers[:, 0], cell_centers[:, 1]

def get_points_coordinates(path):
    # Load the text file into a numpy array
    file_path = path+ '/constant/polyMesh/pointCoordinates.txt'
    pointCoordinates = np.loadtxt(file_path)

    return pointCoordinates[:, 0], pointCoordinates[:, 1]

def get_Face_centers(path,sides_list):
    x_tab = []
    y_tab = []
    # Load the text file into a numpy array
    for i in range(len(sides_list)):
        file_path = path+ '/constant/polyMesh/faceCenters_'+ sides_list[i] +'.txt'
        face_centers = np.loadtxt(file_path)
        x_tab.append(face_centers[:, 0])
        y_tab.append(face_centers[:, 1])

    return x_tab, y_tab

def plot_mesh_with_centers(mesh, centers_x, centers_y):
    plt.figure(figsize=(10, 10))

    # Plot the mesh triangles
    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            for cell in cell_block.data:
                polygon_points = mesh.points[cell][:, :2]  # Use only x and y coordinates
                polygon = plt.Polygon(polygon_points, edgecolor='k', facecolor='none')
                plt.gca().add_patch(polygon)
            break

    # Plot the cell centers
    plt.scatter(centers_x, centers_y, color='red', marker='x', label='Cell Centers')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mesh with Cell Centers')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def get_mesh_attributes(pathj):
    pass

if __name__ == "__main__":
    # Load the .msh file

    case_path = "/home/justinbrusche/test_custom_solver_cases/poissonFoam_case"

    centers_x, centers_y = get_cell_centers(case_path)
    filename = "/home/justinbrusche/scripts/mesh_scripts/mesh_check_case/research_mesh.msh"


    mesh = meshio.read(case_path+"/mesh_unstructured_square_rough.msh")


    # Plot the mesh and cell centers
    plot_mesh_with_centers(mesh, centers_x, centers_y)
