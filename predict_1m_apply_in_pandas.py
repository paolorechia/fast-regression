from sklearn import linear_model
from pyspark import sql
from pyspark.sql import functions as F, types as T
import pickle
import pandas as pd

from datetime import datetime

spark: sql.SparkSession = (
    sql.SparkSession
        .builder
        .master("local[*]")
        .appName("Spark") \
        .config("spark.executor.memory", "16g")
        .config("spark.driver.memory", "16g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


to_predict: sql.DataFrame = spark.read.format("parquet").load(
    "data/to_predict.parquet",
    schema=T.StructType([
        T.StructField(
            name="X", dataType=T.ArrayType(T.DoubleType())
        )
    ])
)

to_predict = to_predict.withColumn("row_number", F.row_number().over(sql.Window.orderBy("X")))
to_predict.show()

def unserialize_predict(df: pd.DataFrame) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")

    result = []
    for idx, row in enumerate(df.itertuples()):
        with open(f"models/custom_{idx}.bin", "rb") as fp:
            str_= fp.read()

        loaded = pickle.loads(str_)
        reg = linear_model.LinearRegression()
        reg.coef_ = loaded["coef"]
        reg.intercept_ = loaded["intercept"]
        result.append(reg.predict(row.X.reshape(1, -1)))

    return pd.DataFrame({"prediction": result})

t0 = datetime.now()

result = (
    to_predict
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