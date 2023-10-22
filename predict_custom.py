import numpy as np
import pickle
from sklearn import linear_model
import timeit

samples = 1
feature_size = 100
X = np.random.rand(samples, 100)

def unserialize_predict():
    with open("custom.bin", "rb") as fp:
        str_= fp.read()

    loaded = pickle.loads(str_)
    reg = linear_model.LinearRegression()
    reg.coef_ = loaded["coef"]
    reg.intercept_ = loaded["intercept"]
    return reg.predict(X)

t = unserialize_predict()
print(t)
result = timeit.timeit(unserialize_predict, number=100_000)
print(result)
