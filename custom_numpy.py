import numpy as np
from sklearn import linear_model
import pickle
import io

X, Y = np.random.rand(100, 100), np.random.rand(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)
reg.coef_

memfile = io.BytesIO()
np.save(memfile, reg.coef_)

with open("numpy_coef.bin", "wb") as fp:
    fp.write(memfile.getvalue())

memfile2 = io.BytesIO()
np.save(memfile2, reg.intercept_)

with open("numpy_intercept.bin", "wb") as fp:
    fp.write(memfile2.getvalue())
