import numpy as np
from sklearn import linear_model
from pyspark import sql
from pyspark.sql import functions as F, types as T
import pandas as pd
import json

from datetime import datetime

# Loading trained model
with open("intercept.json") as fp:
    intercept = json.load(fp)["intercept"]

print(intercept)
spark: sql.SparkSession = sql.SparkSession.builder.master("local").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

coef: sql.DataFrame = spark.read.format("parquet").load("coef.parquet", schema=T.StructType([
        T.StructField(
            name="coef", dataType=T.ArrayType(T.DoubleType())
        )
    ]))
coef.show()

samples = 100_000
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
to_predict = to_predict.withColumn("row_number", F.row_number().over(sql.Window.orderBy("X")))
to_predict.show()

t0 = datetime.now()
joined = to_predict.join(coef, how="full")
joined.show()

def unserialize_predict(df: pd.DataFrame) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")

    result = []
    for row in df.itertuples():
        reg = linear_model.LinearRegression()
        reg.coef_ = row.coef
        reg.intercept_ = intercept
        result.append(reg.predict(row.X.reshape(1, -1)))

    return pd.DataFrame({"prediction": result})

t0 = datetime.now()

result = (
    joined
        .groupBy("row_number")
        .applyInPandas(
            unserialize_predict,
            schema=T.StructType([
            T.StructField(
                name="prediction", dataType=T.ArrayType(T.DoubleType())
            )])
        )
)

result.show()
t1 = datetime.now()
elapsed = t1 - t0
print(elapsed)
