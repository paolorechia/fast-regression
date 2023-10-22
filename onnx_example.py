import numpy as np
from sklearn import linear_model

X, Y = np.random.rand(100, 100), np.random.rand(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)
reg.coef_

# Convert into ONNX format.
from skl2onnx import to_onnx

onx = to_onnx(reg, X[:1])
with open("linear.onnx", "wb") as f:
    f.write(onx.SerializeToString())



