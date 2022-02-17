from sklearn import preprocessing
import numpy
def preprocess(train_data,test_data, normType=4):
    if(normType==1):
        scaler=preprocessing.StandardScaler().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==2):
        scaler=preprocessing.MinMaxScaler().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==3):
        scaler=preprocessing.Normalizer(norm='l2').fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==4):
        M = numpy.mean(train_data, axis=0)
        S = numpy.std(train_data, axis=0)
        S[S == 0] = M[S == 0] + 10e-10  # Controling devision by zero
        (train_data, test_data) = (train_data - M) / S, (test_data - M) / S
    return train_data, test_data