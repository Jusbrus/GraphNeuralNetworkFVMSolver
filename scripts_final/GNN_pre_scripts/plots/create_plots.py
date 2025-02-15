import matplotlib.pyplot as plt
import networkx as nx




# plot_graph_pooling
def plot_graph_pooling(sources_dual_mesh, targets_dual_mesh, sources_first_step, targets_first_step, sources_face_point, targets_face_point, sources_face_center, targets_face_center, sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates, faceCenters, edgeCenters):
    G_center_point = nx.DiGraph()
    G_face_points = nx.DiGraph()
    G_dual_mesh = nx.DiGraph()
    G_face_edgeCenters = nx.DiGraph()
    G_face_center = nx.DiGraph()
    G_center_face = nx.DiGraph()
    G_faceCenters = nx.DiGraph()
    G_edgeCenters = nx.DiGraph()

    # Add nodes with positions
    pos_center_point = {}
    for i, (x, y) in enumerate(cellCenters):
        G_center_point.add_node(i, pos=(x, y))
        pos_center_point[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_center_point.add_node(len(cellCenters) + i, pos=(x, y))
        pos_center_point[len(cellCenters) + i] = (x, y)

    pos_face_points = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_points.add_node(i, pos=(x, y))
        pos_face_points[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_points[len(faceCenters) + i] = (x, y)

    pos_face_center = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_center.add_node(i, pos=(x, y))
        pos_face_center[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_center[len(faceCenters) + i] = (x, y)

    pos_center_face = pos_face_center  # The positions remain the same for the reversed graph

    pos_face_edgeCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_edgeCenters.add_node(i, pos=(x, y))
        pos_face_edgeCenters[i] = (x, y)
    for i, (x, y) in enumerate(edgeCenters):
        G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_edgeCenters[len(faceCenters) + i] = (x, y)

    posFaceCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_faceCenters.add_node(i, pos=(x, y))
        posFaceCenters[i] = (x, y)

    posEdgeCenters = {}
    for i, (x, y) in enumerate(edgeCenters):
        G_edgeCenters.add_node(i, pos=(x, y))
        posEdgeCenters[i] = (x, y)

    posDualMesh = {}
    for i, (x, y) in enumerate(cellCenters):
        G_dual_mesh.add_node(i, pos=(x, y))
        posDualMesh[i] = (x, y)

    # Add edges
    for src, tgt in zip(sources_first_step, targets_first_step):
        G_center_point.add_edge(src, len(cellCenters) + tgt)

    for src, tgt in zip(sources_face_point, targets_face_point):
        G_face_points.add_edge(src, len(faceCenters) + tgt)

    for src, tgt in zip(sources_face_center, targets_face_center):
        G_face_center.add_edge(src, len(faceCenters) + tgt)

    # Create the reversed edges for G_center_face (switching the pointing of the arrows)
    for src, tgt in zip(sources_face_center, targets_face_center):
        G_center_face.add_edge(len(faceCenters) + tgt, src)

    for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
        G_dual_mesh.add_edge(src, tgt)

    # Draw the graph
    # plt.figure(figsize=(12, 12))
    # nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='k', edge_color='red', width=0.3, arrows=True, node_size=2, arrowsize=10)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)

    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)


    # nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3, arrows=True, node_size=2)
    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
    # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="purple", width=0.6,arrows=True, node_size=2)

    # Separate source and target nodes for coloring
    source_nodes_first_step = range(len(cellCenters))
    target_nodes_first_step = range(len(cellCenters), len(cellCenters) + len(pointCoordinates))

    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='green', node_size=20)

    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='k', node_size=20)
    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='red', node_size=20)

    # nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='black', node_size=10)

    # plt.title('Graph Visualization')
# plot_graph_point_center
def plot_graph(sources_dual_mesh, targets_dual_mesh, sources_first_step, targets_first_step, sources_face_point, targets_face_point, sources_face_center, targets_face_center, sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates, faceCenters, edgeCenters):
    G_center_point = nx.DiGraph()
    G_face_points = nx.DiGraph()
    G_dual_mesh = nx.DiGraph()
    G_face_edgeCenters = nx.DiGraph()
    G_face_center = nx.DiGraph()
    G_center_face = nx.DiGraph()
    G_faceCenters = nx.DiGraph()
    G_edgeCenters = nx.DiGraph()

    # Add nodes with positions
    pos_center_point = {}
    for i, (x, y) in enumerate(pointCoordinates):
        G_center_point.add_node(i, pos=(x, y))
        pos_center_point[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_center_point.add_node(len(pointCoordinates) + i, pos=(x, y))
        pos_center_point[len(pointCoordinates) + i] = (x, y)



    pos_face_points = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_points.add_node(i, pos=(x, y))
        pos_face_points[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_points[len(faceCenters) + i] = (x, y)

    pos_face_center = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_center.add_node(i, pos=(x, y))
        pos_face_center[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_center[len(faceCenters) + i] = (x, y)

    pos_center_face = pos_face_center  # The positions remain the same for the reversed graph

    pos_face_edgeCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_edgeCenters.add_node(i, pos=(x, y))
        pos_face_edgeCenters[i] = (x, y)
    for i, (x, y) in enumerate(edgeCenters):
        G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_edgeCenters[len(faceCenters) + i] = (x, y)

    posFaceCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_faceCenters.add_node(i, pos=(x, y))
        posFaceCenters[i] = (x, y)

    posEdgeCenters = {}
    for i, (x, y) in enumerate(edgeCenters):
        G_edgeCenters.add_node(i, pos=(x, y))
        posEdgeCenters[i] = (x, y)

    posDualMesh = {}
    for i, (x, y) in enumerate(cellCenters):
        G_dual_mesh.add_node(i, pos=(x, y))
        posDualMesh[i] = (x, y)

    # Add edges
    for src, tgt in zip(targets_first_step,sources_first_step):
        print(src,tgt)
        G_center_point.add_edge(src, len(pointCoordinates) + tgt)

    for src, tgt in zip(sources_face_point, targets_face_point):
        G_face_points.add_edge(src, len(faceCenters) + tgt)

    for src, tgt in zip(sources_face_center, targets_face_center):
        G_face_center.add_edge(src, len(faceCenters) + tgt)

    # Create the reversed edges for G_center_face (switching the pointing of the arrows)
    for src, tgt in zip(sources_face_center, targets_face_center):
        G_center_face.add_edge(tgt + len(faceCenters), src)

    for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
        G_dual_mesh.add_edge(src, tgt)

    # Draw the graph
    # plt.figure(figsize=(12, 12))
    nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='k', edge_color='red', width=0.5, arrows=False, node_size=2, arrowsize=10)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)

    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='b', edge_color="blue", width=1, arrows=True, node_size=0, arrowsize=13)
    # nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='b', edge_color="blue", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color='orange', width=0.3, arrows=True, node_size=2, arrowsize=20)

    # nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3, arrows=True, node_size=2)
    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
    # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="blue", width=0.6,arrows=True, node_size=2)

    # Separate source and target nodes for coloring
    source_nodes_first_step = range(len(pointCoordinates))
    target_nodes_first_step = range(len(pointCoordinates), len(cellCenters) + len(pointCoordinates))

    nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='red', node_size=20)
    nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='green', node_size=20)
    # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='blue', node_size=10)

    # plt.title('Graph Visualization')
    # plt.show()



# plot_graph_point_point
def plot_graph_point_point(sources_dual_mesh, targets_dual_mesh, sources_first_step, targets_first_step, sources_face_point, targets_face_point, sources_face_center, targets_face_center, sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates, faceCenters, edgeCenters):
    G_center_point = nx.DiGraph()
    G_face_points = nx.DiGraph()
    G_dual_mesh = nx.DiGraph()
    G_face_edgeCenters = nx.DiGraph()
    G_face_center = nx.DiGraph()
    G_center_face = nx.DiGraph()
    G_faceCenters = nx.DiGraph()
    G_edgeCenters = nx.DiGraph()

    # Add nodes with positions
    pos_center_point = {}
    for i, (x, y) in enumerate(cellCenters):
        G_center_point.add_node(i, pos=(x, y))
        pos_center_point[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_center_point.add_node(len(cellCenters) + i, pos=(x, y))
        pos_center_point[len(cellCenters) + i] = (x, y)

    pos_face_points = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_points.add_node(i, pos=(x, y))
        pos_face_points[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_points[len(faceCenters) + i] = (x, y)

    pos_face_center = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_center.add_node(i, pos=(x, y))
        pos_face_center[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_center[len(faceCenters) + i] = (x, y)

    pos_center_face = pos_face_center  # The positions remain the same for the reversed graph

    pos_face_edgeCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_edgeCenters.add_node(i, pos=(x, y))
        pos_face_edgeCenters[i] = (x, y)
    for i, (x, y) in enumerate(edgeCenters):
        G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_edgeCenters[len(faceCenters) + i] = (x, y)

    posFaceCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_faceCenters.add_node(i, pos=(x, y))
        posFaceCenters[i] = (x, y)

    posEdgeCenters = {}
    for i, (x, y) in enumerate(edgeCenters):
        G_edgeCenters.add_node(i, pos=(x, y))
        posEdgeCenters[i] = (x, y)

    posDualMesh = {}
    for i, (x, y) in enumerate(cellCenters):
        G_dual_mesh.add_node(i, pos=(x, y))
        posDualMesh[i] = (x, y)

    # Add edges
    for src, tgt in zip(sources_first_step, targets_first_step):
        G_center_point.add_edge(src, len(cellCenters) + tgt)

    for src, tgt in zip(sources_face_point, targets_face_point):
        G_face_points.add_edge(src, len(faceCenters) + tgt)

    for src, tgt in zip(sources_face_center, targets_face_center):
        G_face_center.add_edge(src, len(faceCenters) + tgt)

    # Create the reversed edges for G_center_face (switching the pointing of the arrows)
    for src, tgt in zip(sources_face_center, targets_face_center):
        G_center_face.add_edge(len(faceCenters) + tgt, src)

    for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
        G_dual_mesh.add_edge(src, tgt)

    # Draw the graph
    # plt.figure(figsize=(12, 12))
    nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='green', edge_color='green', width=0.3, arrows=False, node_size=2, arrowsize=10)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)

    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='b', edge_color="blue", width=1, arrows=True, node_size=0, arrowsize=20)


    # nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3, arrows=True, node_size=2)
    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
    # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="purple", width=0.6,arrows=True, node_size=2)

    # Separate source and target nodes for coloring
    source_nodes_first_step = range(len(cellCenters))
    target_nodes_first_step = range(len(cellCenters), len(cellCenters) + len(pointCoordinates))

    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
    nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='black', node_size=10)

    # plt.title('Graph Visualization')
    # plt.show()
#plot_graph_face_point
def plot_graph_face_point(sources_dual_mesh, targets_dual_mesh, sources_first_step, targets_first_step, sources_face_point, targets_face_point, sources_face_center, targets_face_center, sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates, faceCenters, edgeCenters):
    G_center_point = nx.DiGraph()
    G_face_points = nx.DiGraph()
    G_dual_mesh = nx.DiGraph()
    G_face_edgeCenters = nx.DiGraph()
    G_face_center = nx.DiGraph()
    G_center_face = nx.DiGraph()
    G_faceCenters = nx.DiGraph()
    G_edgeCenters = nx.DiGraph()

    # Add nodes with positions
    pos_center_point = {}
    for i, (x, y) in enumerate(cellCenters):
        G_center_point.add_node(i, pos=(x, y))
        pos_center_point[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_center_point.add_node(len(cellCenters) + i, pos=(x, y))
        pos_center_point[len(cellCenters) + i] = (x, y)

    pos_face_points = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_points.add_node(i, pos=(x, y))
        pos_face_points[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_points[len(faceCenters) + i] = (x, y)

    pos_face_center = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_center.add_node(i, pos=(x, y))
        pos_face_center[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_center[len(faceCenters) + i] = (x, y)

    pos_center_face = pos_face_center  # The positions remain the same for the reversed graph

    pos_face_edgeCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_edgeCenters.add_node(i, pos=(x, y))
        pos_face_edgeCenters[i] = (x, y)
    for i, (x, y) in enumerate(edgeCenters):
        G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_edgeCenters[len(faceCenters) + i] = (x, y)

    posFaceCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_faceCenters.add_node(i, pos=(x, y))
        posFaceCenters[i] = (x, y)

    posEdgeCenters = {}
    for i, (x, y) in enumerate(edgeCenters):
        G_edgeCenters.add_node(i, pos=(x, y))
        posEdgeCenters[i] = (x, y)

    posDualMesh = {}
    for i, (x, y) in enumerate(cellCenters):
        G_dual_mesh.add_node(i, pos=(x, y))
        posDualMesh[i] = (x, y)

    # Add edges
    for src, tgt in zip(sources_first_step, targets_first_step):
        G_center_point.add_edge(src, len(cellCenters) + tgt)

    for src, tgt in zip(sources_face_point, targets_face_point):
        G_face_points.add_edge(src, len(faceCenters) + tgt)

    for src, tgt in zip(sources_face_center, targets_face_center):
        G_face_center.add_edge(src, len(faceCenters) + tgt)

    # Create the reversed edges for G_center_face (switching the pointing of the arrows)
    for src, tgt in zip(sources_face_center, targets_face_center):
        G_center_face.add_edge(len(faceCenters) + tgt, src)

    for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
        G_dual_mesh.add_edge(src, tgt)

    # Draw the graph
    # plt.figure(figsize=(12, 12))
    nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='green', edge_color='green', width=0.3, arrows=False, node_size=2, arrowsize=10)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)

    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='b', edge_color="blue", width=1, arrows=True, node_size=0, arrowsize=20)


    # nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3, arrows=True, node_size=2)
    # nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="green", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
    # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="purple", width=0.6,arrows=True, node_size=2)

    # Separate source and target nodes for coloring
    source_nodes_first_step = range(len(cellCenters))
    target_nodes_first_step = range(len(cellCenters), len(cellCenters) + len(pointCoordinates))

    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='green', node_size=100)
    nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='black', node_size=10)

    # plt.title('Graph Visualization')
    # plt.show()

# plot_graph_center_face
def plot_graph_center_face(sources_dual_mesh, targets_dual_mesh, sources_first_step, targets_first_step, sources_face_point, targets_face_point, sources_face_center, targets_face_center, sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates, faceCenters, edgeCenters):
    G_center_point = nx.DiGraph()
    G_face_points = nx.DiGraph()
    G_dual_mesh = nx.DiGraph()
    G_face_edgeCenters = nx.DiGraph()
    G_face_center = nx.DiGraph()
    G_center_face = nx.DiGraph()
    G_faceCenters = nx.DiGraph()
    G_edgeCenters = nx.DiGraph()

    # Add nodes with positions
    pos_center_point = {}
    for i, (x, y) in enumerate(cellCenters):
        G_center_point.add_node(i, pos=(x, y))
        pos_center_point[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_center_point.add_node(len(cellCenters) + i, pos=(x, y))
        pos_center_point[len(cellCenters) + i] = (x, y)

    pos_face_points = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_points.add_node(i, pos=(x, y))
        pos_face_points[i] = (x, y)
    for i, (x, y) in enumerate(pointCoordinates):
        G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_points[len(faceCenters) + i] = (x, y)

    pos_face_center = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_center.add_node(i, pos=(x, y))
        pos_face_center[i] = (x, y)
    for i, (x, y) in enumerate(cellCenters):
        G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_center[len(faceCenters) + i] = (x, y)

    pos_center_face = pos_face_center  # The positions remain the same for the reversed graph

    pos_face_edgeCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_face_edgeCenters.add_node(i, pos=(x, y))
        pos_face_edgeCenters[i] = (x, y)
    for i, (x, y) in enumerate(edgeCenters):
        G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
        pos_face_edgeCenters[len(faceCenters) + i] = (x, y)

    posFaceCenters = {}
    for i, (x, y) in enumerate(faceCenters):
        G_faceCenters.add_node(i, pos=(x, y))
        posFaceCenters[i] = (x, y)

    posEdgeCenters = {}
    for i, (x, y) in enumerate(edgeCenters):
        G_edgeCenters.add_node(i, pos=(x, y))
        posEdgeCenters[i] = (x, y)

    posDualMesh = {}
    for i, (x, y) in enumerate(cellCenters):
        G_dual_mesh.add_node(i, pos=(x, y))
        posDualMesh[i] = (x, y)

    # Add edges
    for src, tgt in zip(sources_first_step, targets_first_step):
        G_center_point.add_edge(src, len(cellCenters) + tgt)

    for src, tgt in zip(sources_face_point, targets_face_point):
        G_face_points.add_edge(src, len(faceCenters) + tgt)

    for src, tgt in zip(sources_face_center, targets_face_center):
        G_face_center.add_edge(src, len(faceCenters) + tgt)

    # Create the reversed edges for G_center_face (switching the pointing of the arrows)
    for src, tgt in zip(sources_face_center, targets_face_center):
        G_center_face.add_edge(len(faceCenters) + tgt, src)

    for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
        G_dual_mesh.add_edge(src, tgt)

    # Draw the graph
    # plt.figure(figsize=(12, 12))
    # nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='k', edge_color='red', width=0.3, arrows=True, node_size=2, arrowsize=10)
    nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)
    # nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3, arrows=True, node_size=2)
    nx.draw(G_center_face, pos_center_face, with_labels=False, node_color='b', edge_color="blue", width=1, arrows=True, node_size=0, arrowsize=20)
    # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
    # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="purple", width=0.6,arrows=True, node_size=2)

    # Separate source and target nodes for coloring
    source_nodes_first_step = range(len(cellCenters))
    target_nodes_first_step = range(len(cellCenters), len(cellCenters) + len(pointCoordinates))

    nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
    # nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='green', node_size=100)
    # nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='green', node_size=20)
    # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='black', node_size=10)

    # plt.title('Graph Visualization')
    # plt.show()


#
# def plot_graph(sources_dual_mesh, targets_dual_mesh,sources_first_step, targets_first_step,sources_face_point, targets_face_point,sources_face_center, targets_face_center,sources_face_edgeCenters, targets_face_edgeCenters, cellCenters, pointCoordinates,faceCenters,edgeCenters):
#     G_center_point = nx.DiGraph()
#     G_face_points = nx.DiGraph()
#     G_dual_mesh = nx.DiGraph()
#     G_face_edgeCenters = nx.DiGraph()
#     G_face_center = nx.DiGraph()
#     G_faceCenters = nx.DiGraph()
#     G_edgeCenters = nx.DiGraph()
#
#     # Add nodes with positions
#     pos_center_point = {}
#     for i, (x, y) in enumerate(cellCenters):
#         G_center_point.add_node(i, pos=(x, y))
#         pos_center_point[i] = (x, y)
#     for i, (x, y) in enumerate(pointCoordinates):
#         G_center_point.add_node(len(cellCenters) + i, pos=(x, y))
#         pos_center_point[len(cellCenters) + i] = (x, y)
#
#     pos_face_points = {}
#     for i, (x, y) in enumerate(faceCenters):
#         G_face_points.add_node(i, pos=(x, y))
#         pos_face_points[i] = (x, y)
#     for i, (x, y) in enumerate(pointCoordinates):
#         G_face_points.add_node(len(faceCenters) + i, pos=(x, y))
#         pos_face_points[len(faceCenters) + i] = (x, y)
#
#     pos_face_center = {}
#     for i, (x, y) in enumerate(faceCenters):
#         G_face_center.add_node(i, pos=(x, y))
#         pos_face_center[i] = (x, y)
#     for i, (x, y) in enumerate(cellCenters):
#         G_face_center.add_node(len(faceCenters) + i, pos=(x, y))
#         pos_face_center[len(faceCenters) + i] = (x, y)
#
#     pos_face_edgeCenters = {}
#     for i, (x, y) in enumerate(faceCenters):
#         G_face_edgeCenters.add_node(i, pos=(x, y))
#         pos_face_edgeCenters[i] = (x, y)
#     for i, (x, y) in enumerate(edgeCenters):
#         G_face_edgeCenters.add_node(len(faceCenters) + i, pos=(x, y))
#         pos_face_edgeCenters[len(faceCenters) + i] = (x, y)
#
#     posFaceCenters = {}
#     for i, (x, y) in enumerate(faceCenters):
#         G_faceCenters.add_node(i, pos=(x, y))
#         posFaceCenters[i] = (x, y)
#
#     posEdgeCenters = {}
#     for i, (x, y) in enumerate(edgeCenters):
#         G_edgeCenters.add_node(i, pos=(x, y))
#         posEdgeCenters[i] = (x, y)
#
#     posDualMesh = {}
#     for i, (x, y) in enumerate(cellCenters):
#         G_dual_mesh.add_node(i, pos=(x, y))
#         posDualMesh[i] = (x, y)
#
#     # Add edges
#     for src, tgt in zip(sources_first_step, targets_first_step):
#         G_center_point.add_edge(src, len(cellCenters) + tgt)
#
#     for src, tgt in zip(sources_face_point, targets_face_point):
#         G_face_points.add_edge(src, len(faceCenters) + tgt)
#
#     for src, tgt in zip(sources_face_center, targets_face_center):
#         G_face_center.add_edge(src, len(faceCenters) + tgt)
#
#     # for src, tgt in zip(sources_face_edgeCenters, targets_face_edgeCenters):
#     #     G_face_edgeCenters.add_edge(src, len(faceCenters) + tgt)
#
#     for src, tgt in zip(sources_dual_mesh, targets_dual_mesh):
#         G_dual_mesh.add_edge(src, tgt)
#
#     # Draw the graph
#     # plt.figure(figsize=(12, 12))
#     # nx.draw(G_center_point, pos_center_point, with_labels=False, node_color='k', edge_color='red', width=0.3, arrows=True, node_size=2, arrowsize=10)
#     nx.draw(G_face_points, pos_face_points, with_labels=False, node_color='r', edge_color="black", width=0.5, arrows=False, node_size=0)
#     nx.draw(G_face_center, pos_face_center, with_labels=False, node_color='k', edge_color="green", width=0.3,arrows=True, node_size=2)
#     # nx.draw(G_face_edgeCenters, pos_face_edgeCenters, with_labels=False, node_color='k', edge_color="blue", width=0.3,arrows=True, node_size=2)
#     # nx.draw(G_dual_mesh, posDualMesh, with_labels=False, node_color='k', edge_color="purple", width=0.6,arrows=True, node_size=2)
#
#     # Separate source and target nodes for coloring
#     source_nodes_first_step = range(len(cellCenters))
#     target_nodes_first_step = range(len(cellCenters), len(cellCenters) + len(pointCoordinates))
#
#     nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=source_nodes_first_step, node_color='red', node_size=20)
#     #nx.draw_networkx_nodes(G_center_point, pos_center_point, nodelist=target_nodes_first_step, node_color='green', node_size=100)
#     nx.draw_networkx_nodes(G_faceCenters, posFaceCenters, nodelist=range(len(faceCenters)), node_color='Green', node_size=20)
#     # nx.draw_networkx_nodes(G_edgeCenters, posEdgeCenters, nodelist=range(len(edgeCenters)), node_color='black', node_size=10)
#
#     plt.title('Graph Visualization')
#     plt.show()


def plot_primal_mesh(coords, sources, targets, title, scores=None, point_scores=None):
    G_primal = nx.DiGraph()

    # Extract node positions from coords
    pos_primal = {i: (coord[0], coord[1]) for i, coord in enumerate(coords)}

    # Add nodes with positions
    for i, (x, y) in pos_primal.items():
        G_primal.add_node(i, pos=(x, y))

    # Add edges with scores as edge attributes
    edge_list = []
    for idx, (src, tgt) in enumerate(zip(sources, targets)):
        if scores is not None:
            G_primal.add_edge(src, tgt, weight=scores[idx], index=idx)
        else:
            G_primal.add_edge(src, tgt, index=idx)
        edge_list.append((src, tgt, idx))

    # Draw the graph
    plt.figure(figsize=(9, 9))

    # Draw nodes and edges
    if scores is not None:
        edges = G_primal.edges()
        edge_colors = [G_primal[u][v]['weight'] for u, v in edges]
        nx.draw(G_primal, pos_primal, with_labels=False, node_color='red', edge_color=edge_colors, edge_cmap=plt.cm.viridis, width=0.8, arrows=True, node_size=50, arrowsize=20)
    else:
        nx.draw(G_primal, pos_primal, with_labels=False, node_color='red', edge_color='blue', width=0.8, arrows=True, node_size=50, arrowsize=20)

    # Add node labels (index and point_scores)
    if point_scores is not None:
        node_labels = {i: f'{i}: {point_scores[i]:.2f}' for i in G_primal.nodes()}
    else:
        node_labels = {i: str(i) for i in G_primal.nodes()}
    nx.draw_networkx_labels(G_primal, pos_primal, labels=node_labels, font_color='black', font_size=12, bbox=dict(facecolor='white', edgecolor='none', alpha=1))

    # Add edge labels (index and scores)
    edge_labels = {}
    if scores is not None:
        for src, tgt, idx in edge_list:
            edge_labels[(src, tgt)] = f'{idx}: {scores[idx]:.2f}'
    else:
        for src, tgt, idx in edge_list:
            edge_labels[(src, tgt)] = f'{idx}'

    nx.draw_networkx_edge_labels(G_primal, pos_primal, edge_labels=edge_labels, font_color='black', font_size=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    plt.title(title)

def plot_primal_mesh_color(coords, sources, targets, title, scores=None):
    G_primal = nx.DiGraph()

    # Extract node positions from coords
    pos_primal = {i: (coord[0], coord[1]) for i, coord in enumerate(coords)}

    # Add nodes with positions
    for i, (x, y) in pos_primal.items():
        G_primal.add_node(i, pos=(x, y))

    # Add edges with scores as edge attributes
    for idx, (src, tgt) in enumerate(zip(sources, targets)):
        if scores is not None:
            G_primal.add_edge(src, tgt, weight=scores[idx])
        else:
            G_primal.add_edge(src, tgt)

    # Draw the graph
    plt.figure(figsize=(9, 9))

    # Draw nodes and edges with edge colors based on scores
    if scores is not None:
        edges = G_primal.edges()
        edge_colors = [G_primal[u][v]['weight'] for u, v in edges]
        nx.draw(G_primal, pos_primal, with_labels=False, node_color='red', edge_color=edge_colors, edge_cmap=plt.cm.viridis, width=0.8, arrows=True, node_size=50, arrowsize=20)
    else:
        nx.draw(G_primal, pos_primal, with_labels=False, node_color='red', edge_color='blue', width=0.8, arrows=True, node_size=50, arrowsize=20)

    # Add node labels (index)
    node_labels = {i: str(i) for i in G_primal.nodes()}
    nx.draw_networkx_labels(G_primal, pos_primal, labels=node_labels, font_color='black')

    plt.title(title)


def plot_primal_pooled_mesh(x, edge_index, pooled_x, pooled_edge_index, title):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot original graph
    G_primal = nx.DiGraph()
    pos_primal = {i: (coord[0].item(), coord[1].item()) for i, coord in enumerate(x)}
    for i, (x, y) in pos_primal.items():
        G_primal.add_node(i, pos=(x, y))
    for src, tgt in edge_index.T.tolist():
        if src != tgt:  # Remove self-loops
            G_primal.add_edge(src, tgt)

    nx.draw(G_primal, pos_primal, with_labels=False, node_color='blue', edge_color='blue', width=0.3, arrows=True, node_size=20, arrowsize=10, ax=ax)

    # Plot pooled graph
    G_pooled = nx.DiGraph()
    pos_pooled = {i: (coord[0].item(), coord[1].item()) for i, coord in enumerate(pooled_x)}
    for i, (x, y) in pos_pooled.items():
        G_pooled.add_node(i, pos=(x, y))
    for src, tgt in pooled_edge_index.T.tolist():
        if src != tgt:  # Remove self-loops
            G_pooled.add_edge(src, tgt)

    nx.draw(G_pooled, pos_pooled, with_labels=False, node_color='black', edge_color='black', width=0.6, arrows=True, node_size=40, arrowsize=15, ax=ax)

    plt.title(title)



