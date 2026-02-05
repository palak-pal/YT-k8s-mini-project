# YT-k8s-mini-project — ClusterLens (Unsupervised ML + Kubernetes)

End-to-end unsupervised ML app with a clean UI: upload a CSV, pick features, train a clustering or anomaly model, visualize results (PCA projection), and export predictions + a reusable model artifact.

## Features
- CSV upload + sample dataset
- Automatic preprocessing (impute, one-hot, scale)
- Clustering: K-Means, DBSCAN, Gaussian Mixture
- Anomaly detection: Isolation Forest, Local Outlier Factor
- Interactive plots (Plotly) + downloadable results
- Docker + Kubernetes manifests included

## Local run
```bash
make install
make run
```
Then open `http://localhost:8501`.

## Tests
```bash
make test
```

## Docker
```bash
docker build -t paldoc4/kubo_pro:01 .
docker run --rm -p 8501:8501 paldoc4/kubo_pro:01
```

## GitHub
Clone:
```bash
git clone https://github.com/palak-pal/YT-k8s-mini-project.git
cd YT-k8s-mini-project
```

Push changes:
```bash
git add -A
git commit -m "Update"
git push
```

### GitHub Actions (optional)
- CI runs tests on every PR/push to `main`.
- Docker publish runs when you push a git tag like `v0.1.0` and pushes:
  - `paldoc4/kubo_pro:<tag>`, `paldoc4/kubo_pro:latest`, `paldoc4/kubo_pro:sha-<sha>`

Secrets required (GitHub repo → Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Kubernetes (Minikube)
1. Build + push your image:
```bash
docker build -t paldoc4/kubo_pro:01 .
docker push paldoc4/kubo_pro:01
```

2. Apply manifests (namespace is `default`):
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
# Optional (requires an ingress controller enabled):
kubectl apply -f k8s/ingress.yaml
```

3. Get the URL:
```bash
minikube service cluster-lens -n default --url
```
