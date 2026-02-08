# AI Movie Genre Prediction — AIF Final Project (2025)

## Overview

This project implements an end-to-end AI system for movie genre prediction and discovery, based on:

- **movie posters** (CNN-based models),
- **movie plots** (NLP models),
- **multimodal similarity search** using CLIP + Annoy.

The system exposes:
- a Flask REST API
- an interactive Gradio web interface

The application can be run locally or deployed using Docker and Google Cloud Compute Engine.
All model paths, weights and datasets are handled dynamically via `config.yaml`.

## Features

Poster genre prediction (single or batch)

Poster validation using OOD detection

Plot-based genre prediction + movie recommendations

Natural language movie discovery (CLIP + Annoy)

REST API + Web interface

Docker-ready & cloud-deployable

## Prerequisites

- Python ≥ 3.10
- Docker & Docker Compose
- Git
- Google Cloud account (for cloud deployment)

---

## Local Installation (without Docker)

**Clone the repository**

```bash
git clone <repository-url>
cd aif_project_2025
```

**Install dependencies**

```bash
pip install -r requirements.txt
```
**Download weights and dataset**

```bash
# install tools
pip install gdown
pip install --upgrade huggingface-hub

# download + extract dataset
gdown 1rlpHuLlKoSZzHwGh7zVK9nIt2KnG7yW5 -O data.zip
unzip data.zip -d temp_data
rm data.zip
mkdir -p data
mv temp_data/* data/
rm -rf temp_data
gdown 1RpYKQFutnboeQNCqWO5y-bUfWTSZW1Vg -O plot_embeddings.ann
gdown --folder 1Xnt-tW7yutxsy4UBCNRL2c-QAoTV5v7R --remaining-ok
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AnhQuan003/aif', local_dir='.', allow_patterns=['weights_12/*', 'weights_3/*'])"
```
or if you use Windows
```powershell
pip install gdown
pip install --upgrade huggingface-hub

gdown 1rlpHuLlKoSZzHwGh7zVK9nIt2KnG7yW5 -O data.zip
Expand-Archive data.zip temp_data
Remove-Item data.zip

New-Item -ItemType Directory -Force data
Move-Item temp_data\* data\
Remove-Item temp_data -Recurse -Force

gdown 1RpYKQFutnboeQNCqWO5y-bUfWTSZW1Vg -O plot_embeddings.ann
gdown --folder 1Xnt-tW7yutxsy4UBCNRL2c-QAoTV5v7R --remaining-ok

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AnhQuan003/aif', local_dir='.', allow_patterns=['weights_12/*', 'weights_3/*'])"
```

**Configure paths**
Edit `config.yaml` to specify:
- model paths
- weights directories
- datasets locations

**Run the API**

```bash
python movie_api.py --config config.yaml
```

API available at:
http://localhost:5075

**Run the Gradio interface**

```bash
python movie_gradio.py
```

Web interface available at:
http://localhost:7860

---

## Docker Usage (Local)

**Build and run containers**

```bash
docker-compose up --build
```

Or detached mode:

```bash
docker-compose up -d
```

**Useful Docker commands**

```bash
docker-compose ps
docker-compose logs -f
docker-compose down
```

---

## Deployment on Google Cloud (Compute Engine)

This section describes how to deploy the project on a Google Cloud VM using Docker.

### 1. Create a VM Instance

- Go to Google Cloud Console → Compute Engine → VM Instances
- Create a new instance (Ubuntu / Debian-based)
- Allow HTTP traffic
- Note the zone (e.g. us-central1-f)
- Reserve a static external IP

### 2. Connect to the VM

From your local machine:

```bash
gcloud compute ssh instance-20251117-134913 --zone us-central1-f
```

### 3. Install Docker on the VM

On the VM:

```bash
sudo apt update
sudo apt install -y docker.io
sudo curl -SL https://github.com/docker/compose/releases/download/v2.40.3/docker-compose-linux-x86_64 \
	-o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo systemctl enable docker
sudo systemctl start docker
```

Verify:

```bash
docker --version
docker-compose --version
```

### 4. Transfer the Project to the VM

From your local machine (recommended: only deploy folder):

```bash
tar -czf projet_aif_deploy.tar.gz deploy
gcloud compute scp projet_aif_deploy.tar.gz instance-20251117-134913:/home/$USER/ --zone us-central1-f
```

On the VM:

```bash
rm -rf ~/projet_aif
mkdir -p ~/projet_aif
tar -xzf ~/projet_aif_deploy.tar.gz -C ~/projet_aif
cd ~/projet_aif/deploy
```

### 5. Build and Run with Docker Compose

On the VM:

```bash
sudo docker-compose up -d --build
```

Check status:

```bash
sudo docker-compose ps
```

### 6. Access the Application

Retrieve the external IP of the VM:

```bash
curl ifconfig.me
```

Then open in a browser:

Gradio UI  
http://<EXTERNAL_IP>:7860

http://34.170.114.145:7860

API  
http://<EXTERNAL_IP>:5075

### 7. Updating the Application

After modifying code or assets:

```bash
sudo docker-compose down
sudo docker-compose up -d --build
```

### 8. Stopping the VM (Important)

When finished, always stop the VM from Google Cloud Console to avoid billing.

---

## API Usage Examples

**List available genres**

```bash
curl http://localhost:5075/genres
```

**Predict poster genre**

```bash
curl -X POST -F "file=@poster.jpg" http://localhost:5075/predict
```

**Poster validation (OOD)**

```bash
curl -X POST -F "file=@poster.jpg" http://localhost:5075/validate_poster
```

**Health check**

```bash
curl http://localhost:5075/health
```

---

## Configuration

- All paths are defined in `config.yaml`
- Switching model or dataset requires no code change
- Just edit `config.yaml` and restart the containers

---

## Project Structure

```
projet_aif/
├── movie_api.py
├── movie_gradio.py
├── config.yaml
├── docker-compose.yml
├── requirements.txt
├── models/
├── weights/
├── data/
│   ├── content/
│   └── movie_plots.csv
├── assets/
├── README.md
```

---

## Notes & Best Practices

- Use Docker volumes only when necessary
- Do not copy raw datasets unnecessarily
- For production use, replace Flask dev server with Gunicorn/Uvicorn
- Always stop cloud instances after usage

