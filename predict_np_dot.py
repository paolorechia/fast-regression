import numpy as np
import pickle
from sklearn import linear_model
import timeit

X = np.random.rand(100, 100)
with open("custom.bin", "rb") as fp:
    str_= fp.read()

def unserialize_predict():
    loaded = pickle.loads(str_)
    coef_: np.array = loaded["coef"]
    intercept_: np.array = loaded["intercept"]
    return np.dot(X, coef_.T) + intercept_

# t = unserialize_predict()
# print(t)
result = timeit.timeit(unserialize_predict, number=100000)
print(result)
