import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx.algorithms.community as nx_comm

globalColors = [
    "#0b5d1e",
    "#004E98",
    "#E8871E",
    "#806443",
    "#725AC1",
    "#5386E4",
    "#ff6700",
    "#49111c",
    "#EF271B",
    "#937666",
    "#B0E298",
    "#1d7874",
    "#da627d",
    "#587B7F",
]


def networkBuildKnn(
    X_net,
    Y_net,
    knn=5,
    ePercentile=None,
    subGraphsConnected=False,
    metric="euclidean",
    neighbors=True,
    modularity=True,
    colors=globalColors,
):
    g = nx.Graph()
    g.graph["knn"] = knn
    g.graph["ePercentile"] = ePercentile
    g.graph["subGraphsConnected"] = subGraphsConnected
    g.graph["metric"] = metric
    g.graph["neighbors"] = neighbors

    lnNet = len(X_net)
    g.graph["classNames"] = list(set(Y_net))
    g.graph["colors"] = colors
    classNodes = [[] for i in g.graph["classNames"]]

    for index, instance in enumerate(X_net):
        label = Y_net[index]
        indexLabel = g.graph["classNames"].index(label)
        classNodes[indexLabel].append(str(index))
        g.add_node(str(index), value=instance, typeNode="net", label=label)
    g.graph["classNodes"] = classNodes

    values = X_net
    if values.ndim == 1:
        values = np.reshape(values, (-1, 1))

    nbrs = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
    nbrs.fit(values)

    distances, indices = nbrs.kneighbors(values)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    for indiceNode, indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if (
                g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]
                or subGraphsConnected
            ):
                g.add_edge(
                    str(indice), str(indiceNode), weight=distances[indiceNode][tmpi]
                )

    if not ePercentile == None:
        eRadius = np.quantile(distances, ePercentile)
        nbrs.set_params(radius=eRadius)
        distances, indices = nbrs.radius_neighbors(values)

        for indiceNode, indicesNode in enumerate(indices):
            for tmpi, indice in enumerate(indicesNode):
                if not str(indice) == str(indiceNode):
                    if (
                        g.nodes()[str(indice)]["label"]
                        == g.nodes()[str(indiceNode)]["label"]
                        or subGraphsConnected
                    ):
                        g.add_edge(
                            str(indice),
                            str(indiceNode),
                            weight=distances[indiceNode][tmpi],
                        )
    g.graph["index"] = lnNet
    if neighbors:
        g.graph["nbrs"] = nbrs

    return g


def insertNode(g, instance, label="?", colors=globalColors):
    nodeIndex = g.graph["index"]
    nbrs = g.graph["nbrs"]
    g.graph["index"] += 1
    colors = g.graph["colors"]
    classNames = g.graph["classNames"]
    classNodes = g.graph["classNodes"]

    g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)
    if label == "?":
        color = "#000000"
    else:
        color = colors[classNames.index(label)]

    # if instance.ndim == 1:
    #     instance = np.reshape(instance, (-1, 1))

    distances, indices = nbrs.kneighbors([instance])
    indices = indices[:, :-1]
    distances = distances[:, :-1]

    for indiceNode, indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            g.add_edge(
                str(indice),
                str(nodeIndex),
                weight=distances[indiceNode][tmpi],
                color=color,
            )

    tmpRadius = g.graph["ePercentile"]
    if not tmpRadius == None:
        distances, indices = nbrs.radius_neighbors(instance)
        for indiceNode, indicesNode in enumerate(indices):
            for tmpi, indice in enumerate(indicesNode):
                if not str(indice) == str(indiceNode):
                    g.add_edge(
                        str(indice),
                        str(nodeIndex),
                        weight=distances[indiceNode][tmpi],
                        color=color,
                    )
    if label == "?":
        for index, e in enumerate(classNames):
            indexLabel = g.graph["classNames"].index(e)
            classNodes[indexLabel].append(str(nodeIndex))
            g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)
    else:
        indexLabel = g.graph["classNames"].index(label)
        classNodes[indexLabel].append(str(nodeIndex))
        g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)


# def networkBuildKnnByClass(
#     X_net,
#     Y_net,
#     knn=5,
#     ePercentile=None,
#     subGraphsConnected=False,
#     metric="euclidean",
#     neighbors=True,
#     modularity=True,
#     colors=globalColors,
# ):
#     g = nx.Graph()
#     g.graph["knn"] = knn
#     g.graph["ePercentile"] = ePercentile
#     g.graph["subGraphsConnected"] = subGraphsConnected
#     g.graph["metric"] = metric
#     g.graph["neighbors"] = neighbors

#     lnNet = len(X_net)
#     g.graph["classNames"] = list(set(Y_net))
#     g.graph["colors"] = colors
#     classNodes = [[] for i in g.graph["classNames"]]

#     for index, instance in enumerate(X_net):
#         label = Y_net[index]
#         indexLabel = g.graph["classNames"].index(label)
#         classNodes[indexLabel].append(str(index))
#         g.add_node(str(index), value=instance, typeNode="net", label=label)
#     g.graph["classNodes"] = classNodes
#     nbrsArray = []
#     for indexesSelected in classNodes:
#         tmp = [int(e) for e in indexesSelected]
#         values = X_net[tmp]
#         if values.ndim == 1:
#             values = np.reshape(values, (-1, 1))

#         nbrs = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
#         nbrs.fit(values)

#         distances, indices = nbrs.kneighbors(values)
#         indices = indices[:, 1:]
#         distances = distances[:, 1:]

#         for indiceNode, indicesNode in enumerate(indices):
#             for tmpi, indice in enumerate(indicesNode):
#                 if (
#                     g.nodes()[str(tmp[indice])]["label"]
#                     == g.nodes()[str(tmp[indiceNode])]["label"]
#                     or subGraphsConnected
#                 ):
#                     g.add_edge(
#                         str(tmp[indice]),
#                         str(tmp[indiceNode]),
#                         weight=distances[indiceNode][tmpi],
#                     )

#         if not ePercentile == None:
#             eRadius = np.quantile(distances, ePercentile)
#             nbrs.set_params(radius=eRadius)
#             distances, indices = nbrs.radius_neighbors(values)

#             for indiceNode, indicesNode in enumerate(indices):
#                 for tmpi, indice in enumerate(indicesNode):
#                     if not str(tmp[indice]) == str(tmp[indiceNode]):
#                         if (
#                             g.nodes()[str(tmp[indice])]["label"]
#                             == g.nodes()[str(tmp[indiceNode])]["label"]
#                             or subGraphsConnected
#                         ):
#                             g.add_edge(
#                                 str(tmp[indice]),
#                                 str(tmp[indiceNode]),
#                                 weight=distances[indiceNode][tmpi],
#                             )
#         nbrsArray.append(nbrs)
#     g.graph["index"] = lnNet
#     if neighbors:
#         g.graph["nbrsArray"] = nbrsArray

#     return g


# def insertNodeByClass(g, instance, label="?", colors=globalColors):
#     nodeIndex = g.graph["index"]
#     nbrs = g.graph["nbrs"]
#     g.graph["index"] += 1
#     colors = g.graph["colors"]
#     classNames = g.graph["classNames"]

#     g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)
#     if label == "?":
#         color = "#000000"
#     else:
#         color = colors[classNames.index(label)]

#     if instance.ndim == 1:
#         instance = np.reshape(instance, (-1, 1))

#     distances, indices = nbrs.kneighbors(instance)
#     indices = indices[:, :-1]
#     distances = distances[:, :-1]

#     for indiceNode, indicesNode in enumerate(indices):
#         for tmpi, indice in enumerate(indicesNode):
#             g.add_edge(
#                 str(indice),
#                 str(nodeIndex),
#                 weight=distances[indiceNode][tmpi],
#                 color=color,
#             )

#     tmpRadius = g.graph["ePercentile"]
#     if not tmpRadius == None:
#         distances, indices = nbrs.radius_neighbors(instance)
#         for indiceNode, indicesNode in enumerate(indices):
#             for tmpi, indice in enumerate(indicesNode):
#                 if not str(indice) == str(indiceNode):
#                     g.add_edge(
#                         str(indice),
#                         str(nodeIndex),
#                         weight=distances[indiceNode][tmpi],
#                         color=color,
#                     )
