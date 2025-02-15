import numpy as np
from GNN_pre_scripts.plots.create_plots import *
import os

class GetGraphData:
    def __init__(self, test_case_path):
        self.test_case_path = test_case_path
        self.facePoints_uncorrected = self.load_data('/constant/polyMesh/facePoints.txt', dtype=int)
        self.faceCells = self.load_data('/constant/polyMesh/faceCells.txt', dtype=int)
        self.pointIndices = self.load_data('/constant/polyMesh/pointIndices.txt', dtype=int)
        self.faceIndices = self.load_data('/constant/polyMesh/faceIndices.txt', dtype=int)
        self.cellCenters, self.pointCoordinates = self.get_position_data()
        self.index_array = self.create_index_array()
        self.index_array_faces = self.create_index_array_faces()
        self.getFacePoints()
        self.get_cellPoints()
        self.get_boundary_points()
        self.get_boundary_faces()
        self.get_boundary_cells()

        cellFaces_uncorrected = self.load_data('/constant/polyMesh/cellFaces.txt', dtype=int)
        self.cellFaces = self.index_array_faces[cellFaces_uncorrected].astype(int)
        self.get_face_centers_from_cells()


    def get_boundary_names(self):
        # Define the file path
        file_path = "constant/polyMesh/boundaryNames.txt"
        total_path = self.test_case_path + '/constant/polyMesh/boundaryNames.txt'
        # Initialize an empty list to store the boundary names
        boundary_names = []

        # Open the file and read its contents
        with open(total_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Strip any leading/trailing whitespace (like newlines) and add to the list
                boundary_names.append(line.strip())

        return boundary_names

    def load_data(self, file_path, dtype=float):
        return np.loadtxt(self.test_case_path + file_path).astype(dtype)

    def get_boundary_faces_ordened(self,sides_list):
        index_dict = {}
        for i in range(len(sides_list)):
            faceIndicesUncorrected = self.load_data('/constant/polyMesh/boundaryFaceIndices_' + sides_list[i] + '.txt', dtype=int)
            faceIndices = self.index_array_faces[faceIndicesUncorrected].astype(int)
            index_dict[sides_list[i]] = faceIndices
            # print(faceIndicesUncorrected)
            # print(faceIndices)

        return index_dict

    def get_boundary_faces(self):
        boundaryFaces = self.load_data('/constant/polyMesh/boundaryFaces.txt', dtype=int)
        boundaryFaces = self.index_array_faces[boundaryFaces].astype(int)
        # print("boundaryfaces",boundaryFaces)

        return boundaryFaces

    def get_boundary_cells(self):
        boundaryCells = self.load_data('/constant/polyMesh/boundaryCells.txt', dtype=int)
        # print("boundaryCells",boundaryCells)

        return boundaryCells

    def create_index_array(self):
        index_array = np.zeros(np.max(self.facePoints_uncorrected) + 1)
        # print(self.facePoints_uncorrected)
        # print(self.pointIndices)
        index_array[self.pointIndices] = np.arange(len(self.pointIndices)).astype(int)
        return index_array

    def get_boundary_points(self):
        BoundarypointIndices = self.load_data('/constant/polyMesh/boundaryPointIndices.txt', dtype=int)
        boundaryPoints = np.where(np.isin(self.pointIndices, BoundarypointIndices))[0]
        return boundaryPoints

    def get_point_types(self):
        point_types = np.zeros(len(self.pointIndices))
        boundaryPoints = self.get_boundary_points()
        point_types[boundaryPoints] = 1
        return point_types

    def get_cell_types(self):
        point_types = self.get_point_types()

        sources, targets = self.get_sources_targets_center_point()
        edge_types = point_types[targets].astype(int)
        cell_types = np.zeros(np.max(sources)+1)
        for i in range(len(edge_types)):
            cell_types[sources[i]] += edge_types[i]

        # print(cell_types.astype(int))
        cell_types = np.where(cell_types == 2,1,0).astype(int)
        return cell_types

    def get_cell_weights(self):
        cell_types = self.get_cell_types()
        factor = 1/2
        n_boundary_cells = np.sum(cell_types)
        n_field_cells = len(cell_types) - n_boundary_cells

        weight_boundaries = (n_field_cells / (1-factor)) * factor / n_boundary_cells

        cell_weights = cell_types*weight_boundaries + 1 * (1-cell_types)
        # print(cell_weights)
        return cell_weights

    def create_index_array_faces(self):
        index_array_faces = np.zeros(np.max(self.faceIndices) + 1)
        # print(self.faceIndices)
        index_array_faces[self.faceIndices] = np.arange(len(self.faceIndices)).astype(int)
        return index_array_faces

    def getFacePoints(self):
        self.facePoints = self.index_array[self.facePoints_uncorrected].astype(int)
        return self.facePoints

    def get_sources_targets_center_point(self):
        shape_cellPoints = np.shape(self.cellPoints)
        sources = np.tile(np.arange(shape_cellPoints[0]), (shape_cellPoints[1], 1)).transpose().flatten()
        targets = self.cellPoints.flatten()
        return sources, targets

    def get_sources_targets_face_edgeCenters(self):
        self.faceIndices = self.load_data('/constant/polyMesh/faceIndices.txt', dtype=int)
        cellFaces = self.load_data('/constant/polyMesh/cellFaces.txt', dtype=int)
        cellFaces = self.index_array_faces[cellFaces].astype(int)

        centerPoint_Sources, centerPoint_Targets = self.get_sources_targets_center_point()


        potential_faces = cellFaces[centerPoint_Sources]

        targets = np.tile(np.arange(len(centerPoint_Sources)), (3, 1)).transpose()

        pointsPerFace = self.facePoints[potential_faces]
        mask = np.any(pointsPerFace == centerPoint_Targets[:, np.newaxis, np.newaxis], axis=2)

        return potential_faces[mask].flatten(), targets[mask].flatten()

    def get_sources_targets_face_point(self):
        shape_facePoints = np.shape(self.facePoints_uncorrected)
        targets = self.index_array[self.facePoints_uncorrected].astype(int).flatten()
        sources = np.tile(np.arange(shape_facePoints[0]), (shape_facePoints[1], 1)).transpose().flatten()
        return sources, targets

    def get_sources_targets_face_center(self):
        targets = self.faceCells.flatten()
        shape_targets = np.shape(self.faceCells)
        sources = np.tile(np.arange(shape_targets[0]), (shape_targets[1], 1)).transpose().flatten()
        i_edge_not_delete = np.where(targets != -1)[0]
        return sources[i_edge_not_delete], targets[i_edge_not_delete]

    def get_edgeCenters_center_point(self):
        sources_center_point, targets_center_point = self.get_sources_targets_center_point()
        source_points = self.cellCenters[sources_center_point]
        target_points = self.pointCoordinates[targets_center_point]
        return 0.5 * (source_points + target_points)

    def get_sources_primal_mesh(self):
        sources = self.index_array[self.facePoints[:, 0]].astype(int).flatten()
        targets = self.index_array[self.facePoints[:, 1]].astype(int).flatten()
        return sources, targets

    def get_sources_targets_dual_mesh(self):
        rows_without_negative_one = ~np.any(self.faceCells == -1, axis=1)
        faceCells_filtered = self.faceCells[rows_without_negative_one]
        return faceCells_filtered[:, 0].flatten(), faceCells_filtered[:, 1].flatten()

    def get_position_data(self):
        pointCoordinates = self.load_data('/constant/polyMesh/pointCoordinates.txt')[:, :-1]
        cellCenters = self.load_data('/constant/polyMesh/cellCenters.txt')[:, :-1]

        return cellCenters, pointCoordinates

    def get_face_centers(self):
        points = self.index_array[self.facePoints_uncorrected].astype(int)
        coordinates = self.pointCoordinates[points]
        face_centers = np.average(coordinates, axis=1)
        return face_centers

    def get_face_centers_from_cells(self):
        coordinates = self.cellCenters[self.faceCells]
        # print(self.faceCells)
        face_centers_from_cells = np.average(coordinates, axis=1)
        face_centers_from_cells[self.get_boundary_faces()] = self.get_face_centers()[self.get_boundary_faces()]
        return face_centers_from_cells

    def get_cellPoints(self):
        cellPoints_uncorrected = self.load_data('/constant/polyMesh/cellPoints.txt', dtype=int)
        self.cellPoints = self.index_array[cellPoints_uncorrected].astype(int)
        return self.cellPoints

    def get_pointPoints(self):
        # print(self.facePoints[:,0])
        sources_face_point, targets_face_point = self.get_sources_targets_face_point()


        pointPointsHalf = targets_face_point.reshape(-1,2)
        # print(sources_face_point.reshape(-1,2))
        # print(sources_face_point[::2])
        pointPoints = np.concatenate((pointPointsHalf, np.flip(pointPointsHalf,axis=1)), axis=0)

        self_loops_part = np.arange(np.max(pointPoints)+1)
        self_loops = np.column_stack((self_loops_part, self_loops_part))
        pointPoints = np.concatenate((pointPoints, self_loops), axis=0)
        # print(pointPoints)
        # print(pointPoints)
        # print(np.shape(pointPoints))
        pointPointFaces = np.concatenate((sources_face_point[::2], sources_face_point[::2]))
        print(pointPointFaces)
        return pointPoints[:,0].astype(int).flatten(), pointPoints[:,1].astype(int).flatten(), pointPointFaces.astype(int)

    def order_polygon(self, polygon):
        if len(polygon) == 0:
            return polygon

        # Convert the list of points to a numpy array
        polygon = np.array(polygon)

        # Calculate the centroid of the polygon
        centroid = np.mean(polygon, axis=0)

        # Calculate the angles between the centroid and each point
        angles = np.arctan2(polygon[:,1] - centroid[1], polygon[:,0] - centroid[0])

        # Sort the points by angle in clockwise order
        ordered_points = polygon[np.argsort(angles)]
        return ordered_points

    def plot_polygons(self, polygons,ax):
        for polygon in polygons:
            polygon = np.array(polygon)
            ax.plot(polygon[:, 0], polygon[:, 1], '-', label='Polygon')
            # Ensure the polygon is closed for plotting
            closed_polygon = np.vstack([polygon, polygon[0]])
            ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], '-',color="blue")

        ax.set_aspect('equal')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Polygons')
        # plt.show()

    def get_polygons(self):
        sources, targets = self.get_sources_targets_center_point()
        nPoints = np.max(targets)+1

        cellCenters, pointCoordinates = self.get_position_data()
        pointTypes = self.get_point_types()

        sources_pointPoints, targets_pointPoints,_ = self.get_pointPoints()
        sources_pointPoints = sources_pointPoints[:-nPoints]
        targets_pointPoints = targets_pointPoints[:-nPoints]

        pointPoint_list = []

        polygonList = []
        for i in range(nPoints):
            polygonList.append([])
            pointPoint_list.append([])
            if pointTypes[i] == 1:
                polygonList[i].append(pointCoordinates[i])

        for i in range(len(sources_pointPoints)):
            pointPoint_list[sources_pointPoints[i]].append(targets_pointPoints[i])

        # print(pointPoint_list)

        for i in range(nPoints):
            if pointTypes[i] == 1:
                for point in pointPoint_list[i]:
                    if pointTypes[point] == 1:
                        # print(i,point)
                        polygonList[i].append(0.5 * (pointCoordinates[i] + pointCoordinates[point]))

        for i in range(len(sources)):
            polygonList[targets[i]].append((1 * cellCenters[sources[i]] + 0*pointCoordinates[targets[i]]))

        orderedPolygonList = [self.order_polygon(polygon) for polygon in polygonList]

        return orderedPolygonList
        # print(orderedPolygonList)

    def plot_graphs(self):
        sources_center_point, targets_center_point = self.get_sources_targets_center_point()
        sources_face_point, targets_face_point = self.get_sources_targets_face_point()
        sources_face_center, targets_face_center = self.get_sources_targets_face_center()
        sources_face_edgeCenters, targets_face_edgeCenters = self.get_sources_targets_face_edgeCenters()
        # sources_face_edgeCenters, targets_face_edgeCenters = 1,1
        sources_dual_mesh, targets_dual_mesh = self.get_sources_targets_dual_mesh()
        faceCenters = self.get_face_centers()
        edgeCenters = self.get_edgeCenters_center_point()
        # faceCenters = self.get_face_centers_from_cells()

        plot_graph(sources_dual_mesh, targets_dual_mesh, sources_center_point, targets_center_point,
                   sources_face_point, targets_face_point, sources_face_center, targets_face_center,
                   sources_face_edgeCenters, targets_face_edgeCenters, self.cellCenters,
                   self.pointCoordinates, faceCenters, edgeCenters)

if __name__ == '__main__':
    test_case_path = "/home/justinbrusche/datasets/step1/case_0/level3"
    # test_case_path = "/home/justinbrusche/datasets/foundationalParameters/case_0/level3"
    # test_case_path = "/home/justinbrusche/datasets/foundationalParameters2/case_0/level3"

    # test_case_path = "/home/justinbrusche/test_custom_solver_cases/explicit_small_poissonfoam"

    sides_list = ["Left", "Upper", "Right", "Lower","Cylinder"]
    sides_list = ["Left", "Upper", "Right", "Lower"]

    # sides_list = ["Inlet", "Outlet", "Cylinder"]
    # sides_list = ["Inlet", "Outlet"]


    mesh_processor = GetGraphData(test_case_path)
    mesh_processor.get_boundary_faces_ordened(sides_list)
    # mesh_processor.get_cell_types()
    mesh_processor.get_pointPoints()
    orderedPolygonList = mesh_processor.get_polygons()
    # plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 8))
    # fig, ax = plt.subplots(figsize=(14, 8))
    # fig, ax = plt.subplots(figsize=(21, 12))


    mesh_processor.plot_graphs()
    # mesh_processor.plot_polygons(orderedPolygonList,ax)
    save_path = f"/home/justinbrusche/plots_thesis/aggr_point_center.png"
    save_path = f"/home/justinbrusche/plots_thesis/aggr_center_face.png"
    save_path = f"/home/justinbrusche/plots_thesis/aggr_face_point.png"

    save_path = f"/home/justinbrusche/plots_thesis/aggr_point_point.png"
    save_path = f"/home/justinbrusche/plots_thesis/aggr_point_center.png"
    save_path = f"/home/justinbrusche/plots_thesis/pres15.png"

    # save_path = f"/home/justinbrusche/plots_thesis/aggr_face_point_bounds.png"
    # save_path = f"/home/justinbrusche/plots_thesis/aggr_point_point_bounds.png"

    # save_path = f"/home/justinbrusche/plots_thesis/pooling1.png"
    # save_path = f"/home/justinbrusche/plots_thesis/aggr_distance_red.png"

    # save_path = f"/home/justinbrusche/plots_thesis/mesh_4_small.png"

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.show()

