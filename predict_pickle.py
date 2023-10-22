import numpy as np
import pickle
from sklearn import linear_model
import timeit

X = np.random.rand(1, 100)


def unserialize_predict():
    with open("linear.pkl", "rb") as fp:
        str_= fp.read()
    reg: linear_model.LinearRegression = pickle.loads(str_)
    reg.predict(X)

result = timeit.timeit(unserialize_predict, number=100_000)
print(result)
