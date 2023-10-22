import ray
import numpy as np
from typing import Dict
import pickle
from sklearn import linear_model
from datetime import datetime


ds = ray.data.read_parquet("data/to_predict.parquet")
class LinearRegressionPredictor:
    def __init__(self):
        with open("data/linear.pkl", "rb") as fp:
            str_= fp.read()
            self.model= linear_model.LinearRegression = pickle.loads(str_)

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = batch["X"]
        reshaped_batch = []
        for x in X:
            reshaped_batch.append(x)
        reshaped = np.vstack(reshaped_batch)
        output = self.model.predict(reshaped)
        return {"output": output}
        

scale = ray.data.ActorPoolStrategy(size=2)

t0 = datetime.now()
# Step 3: Map the Predictor over the Dataset to get predictions.
predictions = ds.map_batches(LinearRegressionPredictor, compute=scale)

# Step 4: Show one prediction output.
result = predictions.take_batch()

t1 = datetime.now()
elapsed = t1 - t0
print(result)
print(elapsed)