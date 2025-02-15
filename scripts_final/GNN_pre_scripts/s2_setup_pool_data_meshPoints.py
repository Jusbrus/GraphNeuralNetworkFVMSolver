import numpy as np
import pickle
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor


class SetupPoolDataMeshPoints:
    def __init__(self, test_case_path):
        self.test_case_path = test_case_path

    def create_polygon(self, points):
        return Polygon(points)

    def calculate_overlap(self, refined_polygons, coarser_polygons):
        sources = []
        targets = []
        attributes = []

        # Create polygons and STRtrees for spatial indexing
        refined_polys = [self.create_polygon(polygon) for polygon in refined_polygons]
        coarser_polys = [self.create_polygon(polygon) for polygon in coarser_polygons]
        tree = STRtree(coarser_polys)

        for i, refined_poly in enumerate(refined_polys):
            for j in tree.query(refined_poly):
                coarser_poly = coarser_polys[j]
                intersection = refined_poly.intersection(coarser_poly)
                if not intersection.is_empty:
                    sources.append(i)
                    targets.append(j)
                    attributes.append(intersection.area / refined_poly.area)

        return sources, targets, attributes

    def getPoolData(self):
        file_path = f"{self.test_case_path}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshes = pickle.load(file)

        self.nLevels = len(dataDictMeshes.keys())
        tab_pooling = []
        tab_unpooling = []
        with ProcessPoolExecutor() as executor:
            for i in range(self.nLevels - 1):
                refined_polygons = dataDictMeshes[i]["orderedPolygonList"]
                coarser_polygons = dataDictMeshes[i + 1]["orderedPolygonList"]
                tab_pooling.append(executor.submit(self.calculate_overlap, coarser_polygons, refined_polygons))
                tab_unpooling.append(executor.submit(self.calculate_overlap, refined_polygons, coarser_polygons))

        dataDictPool = {}
        for i in range(self.nLevels - 1):
            targets_pooling, sources_pooling, attributes_pooling = tab_pooling[i].result()
            targets_unpooling, sources_unpooling, attributes_unpooling = tab_unpooling[i].result()
            # print(np.average(attributes_pooling),np.average(attributes_unpooling))


            dataDictPool[i] = {}
            dataDictPool[i]["pooling"] = {}
            dataDictPool[i]["pooling"]["sources"] = np.array(sources_pooling)
            dataDictPool[i]["pooling"]["targets"] = np.array(targets_pooling)
            dataDictPool[i]["pooling"]["attr"] = np.array(attributes_pooling)

            # a = np.where(np.array(targets_pooling) == 10)[0]
            # print(a)
            # print(np.array(attributes_pooling)[a])
            # print(sum(np.array(attributes_pooling)[a]))

            dataDictPool[i]["unpooling"] = {}
            dataDictPool[i]["unpooling"]["sources"] = np.array(sources_unpooling)
            dataDictPool[i]["unpooling"]["targets"] = np.array(targets_unpooling)
            dataDictPool[i]["unpooling"]["attr"] = np.array(attributes_unpooling)

        # Debug prints (optional)
        # for key, value in dataDictPool.items():
        #     print(f"Level {key}:")
        #     print("Pooling sources:", value["pooling"]["sources"])
        #     print("Pooling targets:", value["pooling"]["targets"])
        #     print("Pooling attr:", value["pooling"]["attr"])
        #     print("Unpooling sources:", value["unpooling"]["sources"])
        #     print("Unpooling targets:", value["unpooling"]["targets"])
        #     print("Unpooling attr:", value["unpooling"]["attr"])

        file_path = f"{self.test_case_path}/dataDictPoolMeshPoints.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(dataDictPool, file)

        return dataDictPool

if __name__ == '__main__':
    from GNN_pre_scripts.plots.create_plots import *
    openfoam_source_path = "/home/justinbrusche/Openfoams/OpenFOAM_poissonFoam"

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_BC_0"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/pooling_case"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1_big"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_tau_1"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_2"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_bigger"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3"

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_p4"

    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_step_3_var"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_p_mag"
    test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_large_cyl"

    test_case_path = "/home/justinbrusche/test_cases/cylinder"
    # test_case_path = "/home/justinbrusche/gnn_openfoam/test_case_big_var"

    mesh_name = "mesh_unstructured_square"
    mesh_name = "generatedMesh"


    pool = SetupPoolDataMeshPoints(test_case_path)
    dataDictPool = pool.getPoolData()

    sources = dataDictPool[0]["pooling"]["sources"]
    targets = dataDictPool[0]["pooling"]["targets"]
    edge_attr = dataDictPool[0]["pooling"]["attr"]

    sources = dataDictPool[0]["unpooling"]["sources"]
    targets = dataDictPool[0]["unpooling"]["targets"]
    edge_attr = dataDictPool[0]["unpooling"]["attr"]

    # print(sources)
    # print(targets)
    shape = (max(targets)+1, max(sources)+1)

    from scipy.sparse import coo_matrix

    sparse_matrix = coo_matrix((edge_attr, (targets, sources)), shape=shape)
    # print(sparse_matrix.shape)

    ones_array = np.ones(sparse_matrix.shape[1])

    # Multiply the sparse matrix by the ones array
    result = sparse_matrix.dot(ones_array)

    # Print the result
    # print("Sparse matrix:\n", sparse_matrix.toarray())
    # print("Array of ones:\n", ones_array)
    # print("Result of multiplication:\n", result)




    file_path = f"{test_case_path}/dataDictMeshes.pkl"
    with open(file_path, 'rb') as file:
        dataDictMeshes = pickle.load(file)

    # i = 1
    # # print(dataDictMeshes[i]["facePoints"])
    # # x1 = dataDictMeshes[i]["pointCoordinates"][:, 0]
    # # y1 = dataDictMeshes[i]["pointCoordinates"][:, 1]
    # sources = dataDictMeshes[i]["primalMesh"]["sources"]
    # targets = dataDictMeshes[i]["primalMesh"]["targets"]
    # print(dataDictMeshes[i]["pointCoordinates"])
    # print(len(dataDictMeshes[i]["pointCoordinates"]))
    # print(sources)
    # plt.scatter(dataDictMeshes[i]["pointCoordinates"][:, 0],dataDictMeshes[i]["pointCoordinates"][:, 1])
    # plt.show()
    # plot_primal_mesh(dataDictMeshes[i]["pointCoordinates"], sources, targets, "test", scores=None, point_scores=None)
    # plt.show()
    #
    # i = 0
    # # print(dataDictMeshes[i]["facePoints"])
    # # x1 = dataDictMeshes[i]["pointCoordinates"][:, 0]
    # # y1 = dataDictMeshes[i]["pointCoordinates"][:, 1]
    # sources = dataDictMeshes[i]["primalMesh"]["sources"]
    # targets = dataDictMeshes[i]["primalMesh"]["targets"]
    # print(dataDictMeshes[i]["pointCoordinates"])
    # print(len(dataDictMeshes[i]["pointCoordinates"]))
    # print(sources)
    # plt.scatter(dataDictMeshes[i]["pointCoordinates"][:, 0],dataDictMeshes[i]["pointCoordinates"][:, 1])
    # plt.show()
    # plot_primal_mesh(dataDictMeshes[i]["pointCoordinates"], sources, targets, "test", scores=None, point_scores=None)
    # plt.show()
