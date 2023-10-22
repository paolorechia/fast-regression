## Only works on Linux machine
import ray

from pyspark import sql
import numpy as np
from typing import Dict
import pickle
from sklearn import linear_model
from datetime import datetime

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES


MAX_NUM_WORKER_NODES = 4

spark: sql.SparkSession = (
    sql.SparkSession
        .builder
        .master("local[*]")
        .appName("Spark") \
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.task.cpus", "4") \
        .getOrCreate()
)
setup_ray_cluster(num_worker_nodes=MAX_NUM_WORKER_NODES, head_node_options={"disable_usage_stats": None})

ray.init()
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

ds.write_parquet("local:///home/paolo/dev/fast-regression/results/spark_ray_1m.parquet")
t1 = datetime.now()
elapsed = t1 - t0

print(elapsed)
shutdown_ray_cluster()
