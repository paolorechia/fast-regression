import numpy as np
from sklearn import linear_model
import pickle
import tqdm

X, Y = np.random.rand(100, 100), np.random.rand(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)
reg.coef_

save = {
    "coef": reg.coef_,
    "intercept": reg.intercept_,
}
for i in tqdm.tqdm(range(1_000_000)):
    with open(f"models/linear_{i}.pkl", "wb") as fp:
        str_ = pickle.dumps(reg)
        fp.write(str_)

    with open(f"models/custom_{i}.bin", "wb") as fp:
        fp.write(pickle.dumps(save))
