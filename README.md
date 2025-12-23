# AI Movie Genre Prediction - Partie 1

## Description
Système de prédiction de genre de films à partir de leurs affiches (posters) utilisant un réseau de neurones convolutif.

## Installation

### Prérequis
- Python 3.8+
- Docker et Docker Compose
- Git

### Option 1: Installation locale

#### 1. Cloner le projet
```bash
git clone <votre-repo>
cd movie-genre-project
```

#### 2. Installer les dépendances
```bash
pip install -r requirements-api.txt
pip install -r requirements_gradio.txt
```

#### 3. Vérifier la structure des données
```bash
python check_structure.py
```

#### 4. Entraîner le modèle
```bash
python train.py --epochs=20 --batch_size=32
```

#### 5. Lancer l'API
```bash
python movie_api.py
```
> Note: Tester l'API avec le fichier `test_api.ipynb` si nécessaire

#### 6. Lancer l'interface Gradio
```bash
python movie_gradio.py
```
> Accéder à l'interface via: http://0.0.0.0:7860

### Option 2: Installation avec Docker

#### 1. Vérifier l'installation de Docker
```bash
docker --version
docker-compose --version
```

#### 2. Construire et lancer les containers
```bash
docker-compose up --build
```
> Note: Cette étape peut prendre quelques minutes. Arrêter avec `Ctrl+C`

#### 3. Commandes Docker utiles

**Voir le statut des containers:**
```bash
docker-compose ps
```

**Voir les logs en temps réel:**
```bash
docker-compose logs -f
```

**Démarrer en mode détaché:**
```bash
docker-compose up -d
```

**Tester l'API:**
```bash
# Voir tous les genres disponibles
curl http://localhost:5075/genres

# Prédire le genre d'un poster 
curl -X POST -F "file=@some_image.jpg" http://localhost:5075/predict

# Tester la santé de l'API
curl http://localhost:5075/health

# Valider si une image est un poster (Partie 2 - OOD)
curl -X POST -F "file=@some_image.jpg" http://localhost:5075/validate_poster

# Image → CNN → logits → softmax → entropie
#                           ↓
#                     score OOD
#                           ↓
#                 score ≤ seuil ?
#                     /        \
#                  oui          non
#               poster        image rejetée
#               valide         (OOD)


```

## Utilisation

### Interface Web
Ouvrir un navigateur et accéder à: http://localhost:7860

> Rafraîchir la page si nécessaire pour assurer la connexion avec l'API

## Structure du projet
```
movie-genre-project/
├── movie_api.py
├── movie_gradio.py
├── train.py
├── check_structure.py
├── test_api.ipynb
├── requirements-api.txt
├── requirements_gradio.txt
├── Dockerfile-api
├── Dockerfile-gradio
└── docker-compose.yml
```