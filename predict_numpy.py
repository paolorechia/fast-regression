import numpy as np

import io
from sklearn import linear_model
import timeit

X = np.random.rand(100, 100)
with open("numpy_coef.bin", "rb") as fp:
    serialized = fp.read()

with open("numpy_intercept.bin", "rb") as fp:
    serialized2 = fp.read()

def unserialize_predict():
    reg = linear_model.LinearRegression()
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)

    memfile2 = io.BytesIO()
    memfile2.write(serialized2)
    memfile2.seek(0)

    reg.coef_ = np.load(memfile)
    reg.intercept_ = np.load(memfile2)
    return reg.predict(X)

# t = unserialize_predict()
# print(t)
result = timeit.timeit(unserialize_predict, number=10000)
print(result)
