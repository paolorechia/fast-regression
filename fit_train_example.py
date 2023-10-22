import numpy as np
from sklearn import linear_model
import pickle
import timeit

X, Y = np.random.rand(100, 100), np.random.rand(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)
reg.coef_

with open("linear.pkl", "wb") as fp:
    str_ = pickle.dumps(reg)
    fp.write(str_)