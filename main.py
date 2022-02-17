#%%
from quipus import HLNB_BC
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV


strNameDataset='moon25'
dataset = tools.getDataFromCSV("./datasets/"+strNameDataset+".csv")
print("DATASET: "+strNameDataset)
print('---------')
#%%
grid_param = {"knn": range(5, 15), "ePercentile": [0.0,0.5], "bnn": [1], "alpha": [0.0]}
quipusClass = HLNB_BC()
gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
)
gd_sr.fit(dataset["data"], dataset["target"])
best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print('---------')

#%%

test = 10
total = []
knnTest = gd_sr.best_params_['knn']
eRadiusTest = gd_sr.best_params_['ePercentile']
bnnTest = gd_sr.best_params_['bnn']
alpha =  gd_sr.best_params_['alpha']
print("knn: ", knnTest)
print("e-radius: ", eRadiusTest)
print("bnnTest: ", bnnTest)
print("alpha: ", alpha)

for i in range(test):
    quipusClass = HLNB_BC(knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha)
    kfold = KFold(n_splits=10, random_state=i + 42, shuffle=True)
    scores = cross_val_score(
        quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
    )
    total.append(scores)
    asd = scores.tostring()
total = np.array(total)
print("--------")
print(total.mean(), total.std() * 2)
print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
print("--------")
# %%
