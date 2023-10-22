import numpy as np
from pyspark import sql
from pyspark.sql import functions as F, types as T
import json

from datetime import datetime
# Loading trained model
with open("intercept.json") as fp:
    intercept = json.load(fp)["intercept"]

print(intercept)
spark: sql.SparkSession = sql.SparkSession.builder.master("local").getOrCreate()

coef: sql.DataFrame = spark.read.format("parquet").load("coef.parquet", schema=T.StructType([
        T.StructField(
            name="coef", dataType=T.ArrayType(T.DoubleType())
        )
    ]))
coef.show()

samples = 10_000
feature_size = 100
# Creating synthetic data
X = np.random.rand(samples, feature_size)
X.tolist()
pyX = [x.tolist() for x in X]
data = []
for x in pyX:
    data.append({"X": x})

to_predict: sql.DataFrame = spark.createDataFrame(
    data,
    schema=T.StructType([
        T.StructField(
            name="X", dataType=T.ArrayType(T.DoubleType())
        )
    ])
)
to_predict.show()

# Applying dot product

t0 = datetime.now()
joined = to_predict.join(coef, how="full")
joined.show()

def predict(features, coef, intercept):
    zipped = zip(features, coef)
    result = 0.0
    for f, c in zipped:
        result += f * c
    return result + intercept

predict_udf = F.udf(predict, returnType=T.DoubleType())
result = joined.withColumn("prediction", predict_udf(F.col("X"), F.col("coef"), F.lit(intercept)))

result.show()
t1 = datetime.now()
elapsed = t1 - t0
print(elapsed)
