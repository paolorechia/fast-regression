#!/bin/bash
# First run setup
kubectl get service raycluster-kuberay-head-svc
kubectl port-forward --address 0.0.0.0 service/raycluster-kuberay-head-svc 8265:8265 &

# The following job's logs will show the Ray cluster's total resource capacity, including 2 CPUs.
# ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources());"
# ray job submit --address http://localhost:8265 -- python -c "import os; print('/data:', os.listdir('/data'))"
ray job submit --address http://localhost:8265 -- python -c "import requests; requests.get('http://192.168.2.105:8000/data/coef.parquet')"