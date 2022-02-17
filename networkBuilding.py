import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx.algorithms.community as nx_comm


def getProperty(g):
    bb = nx.betweenness_centrality(g)
    nx.set_node_attributes(g, bb, "betweenness")


def networkBuildKnn(
    X_net,
    Y_net,
    knn,
    eQuartile=0.5,
    labels=False,
    colors=[
        "#a8201a",
        "#46acc2",
        "#47a64e",
        "#99582a",
        "#d81159",
        "#e8e4e1",
        "#e8e4e1",
        "#e8e2e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
    ],
):
    g = nx.Graph()
    lnNet = len(X_net)
    g.graph["lnNet"] = lnNet
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
    # if isinstance(values[0], (int, float, str)):
    #     values = np.reshape(values, (-1, 1))

    nbrs = NearestNeighbors(n_neighbors=knn + 1, metric="euclidean")
    nbrs.fit(values)

    distances, indices = nbrs.kneighbors(values)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    if not eQuartile == 0.0:
        eRadius = np.quantile(distances, eQuartile)
        nbrs.set_params(radius=eRadius)

    for indiceNode, indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if (
                g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]
                or not labels
            ):
                g.add_edge(
                    str(indice), str(indiceNode), weight=distances[indiceNode][tmpi]
                )
    if not eQuartile == 0.0:
        distances, indices = nbrs.radius_neighbors(values)
        for indiceNode, indicesNode in enumerate(indices):
            for tmpi, indice in enumerate(indicesNode):
                if not str(indice) == str(indiceNode):
                    if (
                        g.nodes()[str(indice)]["label"]
                        == g.nodes()[str(indiceNode)]["label"]
                        or not labels
                    ):
                        g.add_edge(
                            str(indice),
                            str(indiceNode),
                            weight=distances[indiceNode][tmpi],
                        )
    g.graph["index"] = lnNet
    return g, nbrs


def insertNode(g, nbrs, instance, label="?", colors=["#bb9457"]):
    nodeIndex = g.graph["index"]
    g.graph["index"] += 1
    g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)
    colors = g.graph["colors"]
    classNames = g.graph["classNames"]
    if label == "?":
        color = colors[0]
    else:
        color = colors[classNames.index(label)]

    if isinstance(instance, (int, float, str)):
        instance = [instance]

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

    tmpRadius = nbrs.get_params()["radius"]
    if not tmpRadius == 0.0:
        distances, indices = nbrs.radius_neighbors([instance])
        for indiceNode, indicesNode in enumerate(indices):
            for tmpi, indice in enumerate(indicesNode):
                if not str(indice) == str(indiceNode):
                    g.add_edge(
                        str(indice),
                        str(nodeIndex),
                        weight=distances[indiceNode][tmpi],
                        color=color,
                    )


def quipusBuildKnn(
    X_net,
    Y_net,
    knn,
    eQuartile=0.5,
    labels=False,
    colors=[
        "#a8201a",
        "#46acc2",
        "#47a64e",
        "#99582a",
        "#d81159",
        "#e8e4e1",
        "#e8e4e1",
        "#e8e2e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
        "#e8e1e1",
    ],
    inside=False,
):
    G = []
    if inside:
        g, nbrs = networkBuildKnn(X_net, Y_net, knn, eQuartile, labels, colors)
        g.graph["nbrs"] = nbrs
        t = g.graph["classNodes"]
        mod = nx_comm.modularity(g, t)
        g.graph["mod"] = mod
        G.append(g)
    for i in range(len(X_net[0])):
        tmpX = X_net[:, [i]]
        nantmp = np.isnan(tmpX)
        notNan = ~nantmp
        X = tmpX[notNan]
        Y = np.reshape(Y_net, (-1, 1))
        Y = Y[notNan]
        Y.flatten()
        g = nx.Graph()
        g, nbrs = networkBuildKnn(X, Y, knn, eQuartile, labels, colors)
        g.graph["nbrs"] = nbrs
        t = g.graph["classNodes"]
        mod = nx_comm.modularity(g, t)
        g.graph["mod"] = mod
        G.append(g)
    return G


def quipusInsert(G, instance, label="?", colors=["#bb9457"], inside =False, accepted=[]):
    j=0
    if(inside):
        j=1
        insertNode(G[0], G[0].graph["nbrs"],instance, label="?", colors=["#bb9457"])
    for i, e in enumerate(instance):
        if(not accepted==[] and not accepted[i]):
            continue
        insertNode(G[i+j], G[i+j].graph["nbrs"], [e], label="?", colors=["#bb9457"])

def quipusInsertByInstance(
    g, nbrsGroup, instance, label="?", colors=["#bb9457"], inside=True, accepted=[]
):
    nodeIndex = g.graph["index"]
    if inside:
        insertNode(g, g.graph["nbrs"], instance, label="?", colors=["#bb9457"])

    for index, nbrs in enumerate(nbrsGroup):
        if(not accepted[index]):
            continue
        # g.add_node(str(nodeIndex), value=instance, typeNode="opt", label=label)
        colors = g.graph["colors"]
        classNames = g.graph["classNames"]
        if label == "?":
            color = colors[0]
        else:
            color = colors[classNames.index(label)]

        # if isinstance(instance, (int, float, str)):
        #     instance = [instance]
        tmpInstance = None
        if isinstance(instance[index], (int, float, str)):
            tmpInstance = np.reshape(instance[index], (-1, 1))
        distances, indices = nbrs.kneighbors(tmpInstance)
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

        tmpRadius = nbrs.get_params()["radius"]
        if not tmpRadius == 0.0:
            distances, indices = nbrs.radius_neighbors(tmpInstance)
            for indiceNode, indicesNode in enumerate(indices):
                for tmpi, indice in enumerate(indicesNode):
                    if not str(indice) == str(indiceNode):
                        g.add_edge(
                            str(indice),
                            str(nodeIndex),
                            weight=distances[indiceNode][tmpi],
                            color=color,
                        )
