# 1. Create cluster
kind create cluster --image=kindest/node:v1.23.0 --config kind_config.yaml

# 2. kuberay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0
kubectl get pods

# 3. RayCluster
# Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
helm install raycluster kuberay/ray-cluster --version 1.0.0-rc.0

# Once the RayCluster CR has been created, you can view it by running:
kubectl get rayclusters