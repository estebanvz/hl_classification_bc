import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm


def drawGraph(g, title="", sizeGraph=(10,10), labels=False):
    plt.figure("Graph", figsize=sizeGraph)
    if "mod" in g.graph:
        mod = g.graph["mod"]
        plt.title(title + " Q:" + str(round(mod, 4)))
    pos = nx.spring_layout(g, k=0.40,iterations=50,seed=42)
    # pos = nx.spring_layout(g, k=0.3,iterations=100,seed=42)
    # pos = nx.kamada_kawai_layout(g)
    color_group = g.graph["colors"]
    classNames = g.graph["classNames"]
    node_color = []
    edge_color = []
    # print("CLASES:", classes)
    for node, label in g.nodes(data="label"):
        if g.nodes[node]["typeNode"] == "net":
            node_color.append(color_group[classNames.index(label)])
        if g.nodes[node]["typeNode"] == "opt":
            node_color.append("#000000")
    for node_a, node_b, color in g.edges.data("color", default="#9db4c0"):
        edge_color.append("#a6a6a8")
    nx.draw(
        g,
        pos,
        node_color=node_color,
        edge_color=edge_color,
        width=1.3,
        node_size=50,
        with_labels=labels,
    )
    plt.show()
