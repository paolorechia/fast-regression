import ray
import numpy as np
from typing import Dict
import pickle
from sklearn import linear_model
from datetime import datetime


ds = ray.data.read_parquet("data/to_predict.parquet")
class LinearRegressionPredictor:
    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = batch["X"]
        output_list = []
        for i, x in enumerate(X):
            with open(f"models/linear_{i}.pkl", "rb") as fp:
                str_= fp.read()
            self.model = linear_model.LinearRegression = pickle.loads(str_)    
            output_list.append(self.model.predict(x.reshape(1, -1)))
        return {"output": np.array(output_list)}


scale = ray.data.ActorPoolStrategy(size=2)

t0 = datetime.now()
# Step 3: Map the Predictor over the Dataset to get predictions.
predictions = ds.map_batches(LinearRegressionPredictor, compute=scale)

# Step 4: Show one prediction output.
result = predictions.take_batch()

print(result)


t1 = datetime.now()
elapsed = t1 - t0
print(elapsed)