import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import drawGraph as draw


def drawGraphs(G, title="", sizeGraph=(10, 10),labels=False):
    if isinstance(G, list):
        for i in range(len(G)):
            draw.drawGraph(G[i], title, sizeGraph=sizeGraph,labels=labels)
    else:
        draw.drawGraph(G, title, sizeGraph=sizeGraph,labels=labels)


def drawGraphByClass(G, title=""):
    for indexClassName, className in enumerate(G.graph["classNames"]):
        classNodes = [e for e in G.graph["classNodes"][indexClassName]]
        subG = G.subgraph(classNodes)
        draw.drawGraph(subG, title, sizeGraph=(4, 4))


def getDataFromCSV(url, className="Class"):
    dataset = {}
    data = pd.read_csv(url, keep_default_na=False, na_values=np.nan)
    if len(data.values[0]) == 1:
        data = pd.read_csv(url, ";", keep_default_na=False, na_values=np.nan)
    dataset["target"] = data[className].values
    dataset["data"] = data.drop(className, axis=1).values
    return dataset


def normalizations(train_data, test_data, normType=4):
    if normType == 1:
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    if normType == 2:
        scaler = preprocessing.MinMaxScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    if normType == 3:
        scaler = preprocessing.Normalizer(norm="l2").fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    if normType == 4:
        M = np.mean(train_data, axis=0)
        S = np.std(train_data, axis=0)
        S[S == 0] = M[S == 0] + 10e-10
        (train_data, test_data) = (train_data - M) / S, (test_data - M) / S
    return train_data, test_data


def split(X, Y):
    return train_test_split(
        X, Y, test_size=0.5, train_size=0.5, random_state=123, stratify=Y
    )

