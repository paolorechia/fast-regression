import numpy as np
from sklearn import linear_model
import pickle

X, Y = np.random.rand(100, 100), np.random.rand(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)
reg.coef_

save = {
    "coef": reg.coef_,
    "intercept": reg.intercept_,
}
with open("custom.bin", "wb") as fp:
    fp.write(pickle.dumps(save))
