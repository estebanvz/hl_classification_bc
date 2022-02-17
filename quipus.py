import networkBuilding as nBuilding
import numpy as np
import networkx as nx
import predict as predict
import tools
import pyswarms as ps
from sklearn.model_selection import train_test_split, cross_val_score


class HLNB_BC:
    knn = None
    X_train = []
    Y_train = []

    def __str__(self):
        return "HLNB_BC"

    def predict(self, X_test, Y_test=[]):
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test,1)
        result = []
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test)
        g, nbrs = nBuilding.networkBuildKnn(
            self.X_train, self.Y_train, self.knn, self.ePercentile, labels=True
        )
        # nBuilding.getProperty(g)
        # draw.drawGraph(g,title="Graph Iris Dataset k="+str(self.knn)+" e="+str(self.ePercentile)+ " b=10 α=0.0" )
        # draw.drawGraph(g,title="" )
        results = []
        for index, instance in enumerate(X_test):
            # CHECK INDEX LNNET WAS REMOVED + index
            indexNode = g.graph["lnNet"]
            if len(Y_test) == 0:
                nBuilding.insertNode(g, nbrs, instance)
            else:
                nBuilding.insertNode(g, nbrs, instance, Y_test[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.prediction(g, self.bnn, self.alpha)
            results.append(tmpResults)
            maxIndex = np.argmax(tmpResults)
            newLabel = g.graph["classNames"][maxIndex]
            result.append(newLabel)
            # g.remove_node(str(indexNode))
            g.nodes[str(indexNode)]["label"] = newLabel
            nn = list(nx.neighbors(g, str(indexNode)))
            for node in nn:
                if g.nodes[str(node)]["label"] != newLabel:
                    g.remove_edge(str(node), str(indexNode))
            # draw.drawGraph(g,"Final Node")
            for edge in g.edges:
                g.edges[edge]["color"] = "#9db4c0"
            g.graph["index"] += 1
        # draw.drawGraph(g,title="")

        if len(Y_test) != 0:
            # print("RESULT:", np.array(result))
            # print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(g.graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append([element, Y_test[index], results[index]])
            acc /= len(X_test)

            # print("ERRORS: ", err)
            print("Accuracy ", round(acc, 2), "%")
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def __init__(self, knn=3, ePercentile=0.5, bnn=3, alpha=1.0):
        self.knn = knn
        self.bnn = bnn
        self.alpha = alpha
        self.ePercentile = ePercentile

    def get_params(self, deep=False):
        return {
            "knn": self.knn,
            "ePercentile": self.ePercentile,
            "bnn": self.bnn,
            "alpha": self.alpha,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X_test, Y_test):
        self.predict(X_test, Y_test)


class EnsambleHLNBBC(HLNB_BC):
    graphs = []

    def __str__(self):
        return "EnsambleHLNBBC"

    def predict(self, X_test, Y_test=[]):
        # self.graphs = nBuilding.quipusBuildKnn(
        #     self.X_train, self.Y_train, self.knn, self.ePercentile, labels=True
        # )
        result = []
        results = []
        self.classes = self.graphs[0].graph["classNames"]
        for index, instance in enumerate(X_test):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_test) == 0:
                nBuilding.quipusInsert(self.graphs, instance)
            else:
                nBuilding.quipusInsert(self.graphs, instance, Y_test[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(self.graphs, self.bnn, self.alpha)
            test = tmpResults * self.w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            test2 = test / np.sum(test)
            results.append(test2)
            result.append(list(set(self.Y_train))[np.argmax(test2)])

        if len(Y_test) != 0:
            print("RESULT:", np.array(result))
            print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(self.graphs[0].graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append(
                        [
                            element,
                            Y_test[index],
                            results[index][self.classes.index(element)],
                            results[index][self.classes.index(Y_test[index])],
                        ]
                    )
            acc /= len(X_test)

            print("ERRORS: ", err)
            print("Accuracy ", round(acc, 4), "%")
        # print(test2)
        # print(tmpResults)
        return result
        # tmpResults = predict.prediction(g, indexNode, self.bnn, self.deepNeighbors)
        # results.append(tmpResults)
        # maxIndex = np.argmax(tmpResults)
        # newLabel = g.graph["classNames"][maxIndex]
        # result.append(newLabel)
        # # g.remove_node(str(indexNode))
        # g.nodes[str(indexNode)]["label"] = newLabel
        # nn = list(nx.neighbors(g, str(indexNode)))
        # for node in nn:
        #     if g.nodes[str(node)]["label"] != newLabel:
        #         g.remove_edge(str(node), str(indexNode))
        # # draw.drawGraph(g,"Final Node")
        # for edge in g.edges:
        #     g.edges[edge]["color"] = "#9db4c0"
        # g.graph["index"] += 1
        # draw.drawGraph(g,title="")

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.dim = (len(self.X_train[0]), len(list(set(self.Y_train))))
        self.w = np.ones(self.dim[0] * self.dim[1])
        G = nBuilding.quipusBuildKnn(
            X_train, Y_train, self.knn, self.ePercentile, labels=True
        )
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        probabilities = []

        for index, instance in enumerate(X_train):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_train) == 0:
                nBuilding.quipusInsert(G, instance)
            else:
                nBuilding.quipusInsert(G, instance, Y_train[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(G, self.bnn, self.alpha)
            probabilities.append(tmpResults)
            # test = tmpResults * self.w
            # test = np.transpose(test)
            # test = [np.sum(e) for e in test]
            # test2 = test / np.sum(test)
            # print(test2)
            # print(tmpResults)
        # print(probabilities)
        max_bound = np.ones(self.dim[0] * self.dim[1]) * self.dim[1]
        # min_bound = np.zeros(self.dim[0] * self.dim[1])
        min_bound = -max_bound
        bounds = (min_bound, max_bound)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=50,
            dimensions=self.dim[0] * self.dim[1],
            options=options,
            bounds=bounds,
        )
        cost, pos = optimizer.optimize(
            optimizacion,
            iters=20,
            probabilidades=probabilities,
            Y=Y_train,
            dim=self.dim,
        )
        self.w = np.reshape(pos, (self.dim))
        self.graphs = G
        # max_bound = [0,0,0,0,0]
        # min_bound = [0,0,0,0,0]
        # bounds = [min_bound,max_bound]
        # max_bound = np.ones(len(self.X_train[0]) * len(list(set(self.Y_train)))).tolist()
        # min_bound = np.zeros(len(self.X_train[0]) * len(list(set(self.Y_train))).tolist()
        # bounds = [min_bound,max_bound]
        # optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
        # return


def optimizacion(particles, probabilidades, Y, dim):
    accs = []
    classes = list(set(Y))
    for w_particle in particles:
        w = np.reshape(w_particle, dim)
        acc = 0
        Y_predicted = []
        for index, instanceProb in enumerate(probabilidades):
            test = instanceProb * w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            Y_predicted.append(classes[np.argmax(test)])
        acc = sum(1 for x, y in zip(Y, Y_predicted) if x == y) / len(Y)
        accs.append(1.0 - acc)
    return accs


class Quipus(HLNB_BC):
    graphs = []

    def __str__(self):
        return "Quipus"

    def predict(self, X_test, Y_test=[]):
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test,1)
        result = []
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test)
        # g, nbrs = nBuilding.networkBuildKnn(
        #     self.X_train, self.Y_train, self.knn, self.ePercentile, labels=True
        # )
        # nBuilding.getProperty(g)
        # draw.drawGraph(g,title="Graph Iris Dataset k="+str(self.knn)+" e="+str(self.ePercentile)+ " b=10 α=0.0" )
        # draw.drawGraph(g,title="" )
        g = self.graph
        results = []
        for index, instance in enumerate(X_test):
            # CHECK INDEX LNNET WAS REMOVED + index
            indexNode = g.graph["lnNet"]
            if len(Y_test) == 0:
                nBuilding.quipusInsertByInstance(g, self.nbrsGroup, instance)
            else:
                nBuilding.quipusInsertByInstance(
                    g, self.nbrsGroup, instance, Y_test[index]
                )
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.prediction(g, self.bnn, self.alpha)
            results.append(tmpResults)
            maxIndex = np.argmax(tmpResults)
            newLabel = g.graph["classNames"][maxIndex]
            result.append(newLabel)
            # g.remove_node(str(indexNode))
            g.nodes[str(indexNode)]["label"] = newLabel
            nn = list(nx.neighbors(g, str(indexNode)))
            for node in nn:
                if g.nodes[str(node)]["label"] != newLabel:
                    g.remove_edge(str(node), str(indexNode))
            # draw.drawGraph(g,"Final Node")
            for edge in g.edges:
                g.edges[edge]["color"] = "#9db4c0"
            g.graph["index"] += 1
        # draw.drawGraph(g,title="")

        if len(Y_test) != 0:
            # print("RESULT:", np.array(result))
            # print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(g.graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append([element, Y_test[index], results[index]])
            acc /= len(X_test)

            # print("ERRORS: ", err)
            print("Accuracy ", round(acc, 2), "%")
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        self.dim = (len(self.X_train[0]), len(list(set(self.Y_train))))
        self.w = np.ones(self.dim[0] * self.dim[1])
        options = {"c1": 0.5, "c2": 0.5, "w": 0.9}

        G = nBuilding.quipusBuildKnn(
            X_train, Y_train, self.knn, self.ePercentile, labels=True, inside=True
        )
        # tools.drawGraphs(G[0])

        graph = nx.Graph()
        # print(G[0].nodes())
        # print(G[0].edges())
        # print("---------------")
        # print(G[1].nodes())
        # print(G[1].edges())
        # print("---------------")
        nbrsGroup = []
        flag = False
        for g in G:
            if flag:
                nbrsGroup.append(g.graph["nbrs"])
            flag = True
            graph = nx.compose(graph, g)
        # probabilities = []
        # print("---------------")
        # print(graph.nodes())
        # print(graph.edges())
        # print("---------------")
        # tools.drawGraphs(graph)
        graph.graph = G[0].graph
        self.graph = graph
        self.nbrsGroup = nbrsGroup
        # for index, instance in enumerate(X_train):
        #     # indexNode = self.graphs[0].graph["lnNet"] + index
        #     if len(Y_train) == 0:
        #         nBuilding.quipusInsert(G, instance)
        #     else:
        #         nBuilding.quipusInsert(G, instance, Y_train[index])
        #     # draw.drawGraph(g,"New Dark Node Inserted")
        #     tmpResults = predict.quipusPrediction(G, self.bnn, self.alpha)
        #     probabilities.append(tmpResults)
        #     # test = tmpResults * self.w
        #     # test = np.transpose(test)
        #     # test = [np.sum(e) for e in test]
        #     # test2 = test / np.sum(test)
        #     # print(test2)
        #     # print(tmpResults)
        # # print(probabilities)
        # max_bound = np.ones(self.dim[0] * self.dim[1]) * self.dim[1]
        # # min_bound = np.zeros(self.dim[0] * self.dim[1])
        # min_bound = -max_bound
        # bounds = (min_bound, max_bound)
        # optimizer = ps.single.GlobalBestPSO(
        #     n_particles=50,
        #     dimensions=self.dim[0] * self.dim[1],
        #     options=options,
        #     bounds=bounds,
        # )
        # cost, pos = optimizer.optimize(
        #     optimizacion,
        #     iters=20,
        #     probabilidades=probabilities,
        #     Y=Y_train,
        #     dim=self.dim,
        # )
        # self.w = np.reshape(pos, (self.dim))
        # self.graphs = G


class Quipus2(HLNB_BC):
    graphs = []
    graph = None
    accepted = []

    def __str__(self):
        return "Quipus2"

    def predict(self, X_test, Y_test=[]):
        result = []
        results = []
        self.classes = self.graphs[0].graph["classNames"]
        for index, instance in enumerate(X_test):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_test) == 0:
                nBuilding.quipusInsert(self.graphs, instance, inside=True)
            else:
                nBuilding.quipusInsert(
                    self.graphs, instance, Y_test[index], inside=True
                )
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(self.graphs, self.bnn, self.alpha)
            test = tmpResults * self.w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            test2 = test / np.sum(test)
            results.append(test2)
            result.append(list(set(self.Y_train))[np.argmax(test2)])

        if len(Y_test) != 0:
            print("RESULT:", np.array(result))
            print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(self.graphs[0].graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append(
                        [
                            element,
                            Y_test[index],
                            results[index][self.classes.index(element)],
                            results[index][self.classes.index(Y_test[index])],
                        ]
                    )
            acc /= len(X_test)

            print("ERRORS: ", err)
            print("Accuracy ", round(acc, 4), "%")
        # print(test2)
        # print(tmpResults)
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.dim = (len(self.X_train[0]) + 1, len(list(set(self.Y_train))))
        self.w = np.ones(self.dim[0] * self.dim[1])
        options = {"c1": 0.5, "c2": 0.1, "w": 0.9}

        X_net, X_opt, Y_net, Y_opt = train_test_split(
            X_train, Y_train, test_size=0.20, stratify=Y_train
        )

        G = nBuilding.quipusBuildKnn(
            X_net, Y_net, self.knn, self.ePercentile, labels=True, inside=True
        )

        probabilities = []

        for index, instance in enumerate(X_opt):
            nBuilding.quipusInsert(G, instance, inside=True)
            tmpResults = predict.quipusPrediction(G, self.bnn, self.alpha)
            probabilities.append(tmpResults)
        initial = np.zeros(self.dim[0])
        initial[0] = 1
        initPos = initial
        for i in range(self.dim[1] - 1):
            initPos = np.append(initPos, initial)
        min_bound = -np.ones(self.dim[0] * self.dim[1])
        max_bound = np.ones(self.dim[0] * self.dim[1])
        bounds = (min_bound, max_bound)
        nParticles = 10
        initPosParticles = []
        for _ in range(nParticles):
            initPosParticles.append(initPos)
        initPosParticles = np.array(initPosParticles)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=nParticles,
            dimensions=self.dim[0] * self.dim[1],
            options=options,
            bounds=bounds,
            init_pos=initPosParticles,
        )
        cost, pos = optimizer.optimize(
            optimization,
            iters=500,
            n_processes=None,
            probabilidades=probabilities,
            Y=Y_opt,
            dim=self.dim,
        )
        self.w = np.reshape(pos, (self.dim))
        self.graphs = nBuilding.quipusBuildKnn(
            X_train, Y_train, self.knn, self.ePercentile, labels=True, inside=True
        )


def optimization(particles, probabilidades, Y, dim):
    accs = []
    classes = list(set(Y))
    for w_particle in particles:
        w = np.reshape(w_particle, dim)
        acc = 0
        Y_predicted = []
        for index, instanceProb in enumerate(probabilidades):
            test = instanceProb * w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            Y_predicted.append(classes[np.argmax(test)])
        acc = sum(1 for x, y in zip(Y, Y_predicted) if x == y) / len(Y)
        accs.append(1.0 - acc)
    return accs


class Quipus3(HLNB_BC):
    graphs = []
    graph = None
    accepted = []

    def __str__(self):
        return "Quipus3"

    def predict(self, X_test, Y_test=[]):
        result = []
        results = []
        self.classes = self.graphs[0].graph["classNames"]
        for index, instance in enumerate(X_test):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_test) == 0:
                nBuilding.quipusInsert(
                    self.graphs, instance, inside=True, accepted=self.accepted
                )
            else:
                nBuilding.quipusInsert(
                    self.graphs,
                    instance,
                    Y_test[index],
                    inside=True,
                    accepted=self.accepted,
                )
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(
                self.graphs, self.bnn, self.alpha, accepted=self.accepted
            )
            test = np.transpose(tmpResults) * self.w

            test = [np.sum(e) for e in test]
            test2 = test / np.sum(test)
            results.append(test2)
            result.append(list(set(self.Y_train))[np.argmax(test2)])

        if len(Y_test) != 0:
            print("RESULT:", np.array(result))
            print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(self.graphs[0].graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append(
                        [
                            element,
                            Y_test[index],
                            results[index][self.classes.index(element)],
                            results[index][self.classes.index(Y_test[index])],
                        ]
                    )
            acc /= len(X_test)

            print("ERRORS: ", err)
            print("Accuracy ", round(acc, 4), "%")
        # print(test2)
        # print(tmpResults)
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.dim = (len(self.X_train[0]) + 1, len(list(set(self.Y_train))))
        self.w = np.ones(self.dim[0] * self.dim[1])
        options = {"c1": 0.5, "c2": 0.1, "w": 0.9}

        X_net, X_opt, Y_net, Y_opt = train_test_split(
            X_train, Y_train, test_size=0.20, stratify=Y_train
        )

        G = nBuilding.quipusBuildKnn(
            X_net, Y_net, self.knn, self.ePercentile, labels=False, inside=True
        )
        minQ = G[0].graph["mod"] - 0.1
        accepted = []
        nDimAccepted = 0
        for g in G:
            # print(g.graph["mod"])
            if g.graph["mod"] >= minQ:
                accepted.append(True)
                nDimAccepted += 1
            else:
                accepted.append(False)

        accepted.pop(0)
        probabilities = []

        for index, instance in enumerate(X_opt):
            nBuilding.quipusInsert(G, instance, inside=True, accepted=accepted)
            tmpResults = predict.quipusPrediction(
                G, self.bnn, self.alpha, accepted=accepted
            )
            probabilities.append(tmpResults)
        # initial = np.zeros(self.dim[0])
        # initial[0] = 1
        # initPos = initial

        # for i in range(self.dim[1] - 1):
        #     initPos = np.append(initPos, initial)
        probabilities = np.array(probabilities)
        self.dim = probabilities.shape[1]
        self.w = np.ones(self.dim)
        if(not self.dim==1):
            min_bound = np.zeros(self.dim)
            max_bound = np.ones(self.dim)

            bounds = (min_bound, max_bound)
            nParticles = 10

            # initPosParticles = []
            # for _ in range(nParticles):
            #     initPosParticles.append(initPos)
            # initPosParticles = np.array(initPosParticles)

            optimizer = ps.single.GlobalBestPSO(
                n_particles=nParticles, dimensions=self.dim, options=options, bounds=bounds,
            )
            cost, pos = optimizer.optimize(
                optimizationQ3,
                iters=500,
                n_processes=None,
                probabilidades=probabilities,
                Y=Y_opt,
                dim=self.dim,
            )
            self.w = np.reshape(pos, (self.dim))            
        self.accepted = accepted
        self.graphs = nBuilding.quipusBuildKnn(
            X_train, Y_train, self.knn, self.ePercentile, labels=True, inside=True
        )


def optimizationQ3(particles, probabilidades, Y, dim):
    accs = []
    classes = list(set(Y))
    for w in particles:
        # w = np.reshape(w_particle, dim)
        acc = 0
        Y_predicted = []
        for index, instanceProb in enumerate(probabilidades):
            test = instanceProb.T * w
            # test = np.transpose(test)
            test = [np.sum(e) for e in test]
            Y_predicted.append(classes[np.argmax(test)])
        acc = sum(1 for x, y in zip(Y, Y_predicted) if x == y) / len(Y)
        accs.append(1.0 - acc)
    return accs
