import numpy as np
import pandas as pd

samples = 1_000_000
feature_size = 100

X = np.random.rand(samples, feature_size)
X.tolist()
pyX = [x.tolist() for x in X]
data = []
for x in pyX:
    data.append({"X": x})

to_predict = pd.DataFrame(data)
to_predict.to_parquet("to_predict.parquet")
