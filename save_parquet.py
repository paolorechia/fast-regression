import pandas as pd
import pickle
import json

with open("custom.bin", "rb") as fp:
    str_= fp.read()

loaded = pickle.loads(str_)
intercept = loaded.pop("intercept")
print("intercept", intercept)
df = pd.DataFrame(data={"coef": [loaded["coef"]]})
print(df)
df.to_parquet("coef.parquet")
with open("intercept.json", "w") as fp:
    json.dump({"intercept": intercept}, fp)