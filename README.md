# AIF Project 2025 - Movie Recommendation System with CLIP

Système de recommandation de films utilisant CLIP (Contrastive Language-Image Pre-training) pour la recherche multimodale de films par texte ou image.

## 🎯 Fonctionnalités

### 1. Recherche par Image
- Upload d'une image de poster de film
- Recherche de films similaires dans la base de données
- Recherche inversée pour identifier un film à partir d'une image

### 2. Recherche par Texte
- Description textuelle d'un film (plot, scène, ambiance)
- Recherche de films correspondants via embeddings CLIP
- Support de recherche batch (multiple queries)

### 3. API REST
- Endpoints Flask pour intégration facile
- Support des embeddings texte et image
- Serveur de fichiers pour les posters

## 📁 Structure du Projet

```
aif_project_2025/
├── clip_annoy.py          # Construction de l'index Annoy avec embeddings CLIP
├── api_file.py            # API Flask REST
├── gradio_file.py         # Interface utilisateur Gradio
├── config.py              # Mappings de genres et constantes
├── config.yaml            # Configuration générale
├── requirements.txt       # Dépendances Python
├── assets/                # Index Annoy et métadonnées (générés)
│   ├── clip_movies.ann
│   └── clip_meta.json
└── models/                # Modèles pré-entraînés
```

## 🚀 Installation

### 1. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Préparer les données

Assurez-vous d'avoir :
- `movie_plots.csv` : fichier CSV avec les plots de films
- `content/sorted_movie_posters_paligema/` : dossier contenant les posters

## 🔧 Configuration

Modifier `config.yaml` selon vos besoins :

```yaml
clip:
  model_name: "ViT-B/32"  # Modèle CLIP à utiliser
  embedding_dim: 512       # Dimension des embeddings

data:
  plots_csv: "movie_plots.csv"
  poster_prefix: "content/sorted_movie_posters_paligema"

index:
  annoy_path: "assets/clip_movies.ann"
  meta_path: "assets/clip_meta.json"
  num_trees: 50
  n_candidates: 2000

server:
  api_host: "0.0.0.0"
  api_port: 5075
  gradio_host: "0.0.0.0"
  gradio_port: 7860
```

## 📊 Utilisation

### 1. Construire l'index Annoy

```bash
python clip_annoy.py
```

Cela va :
- Charger le modèle CLIP
- Encoder tous les plots (texte) et posters (images)
- Construire un index Annoy pour la recherche rapide
- Sauvegarder les métadonnées

### 2. Lancer l'API

```bash
python api_file.py
```

L'API sera disponible sur `http://localhost:5075`

### 3. Lancer l'interface Gradio

```bash
python gradio_file.py
```

L'interface sera disponible sur `http://localhost:7860`

## 🔌 Endpoints API

### GET /health
Vérification de l'état du serveur

### GET /stats
Statistiques sur l'index et les données

### POST /retrieve
Recherche de films par texte
```json
{
  "query": "a romantic movie in paris",
  "k": 5
}
```

### POST /retrieve_image
Recherche de films par image
```
multipart/form-data with "file" field
```

### POST /chat
Interface conversationnelle
```json
{
  "message": "find me a thriller movie",
  "k": 5
}
```

### GET /poster/<movie_id>
Récupération d'un poster

## 🛠️ Détails Techniques

### Architecture

- **CLIP Model**: `ViT-B/32` pour l'encodage multimodal
- **Index**: Annoy (Approximate Nearest Neighbors Oh Yeah) avec métrique angular
- **Backend**: Flask pour l'API
- **Frontend**: Gradio pour l'interface utilisateur

### Device Support

Le code détecte automatiquement le meilleur device disponible :
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPU)
3. CPU (fallback)

### Encodage Robuste

- Découpage intelligent des textes longs
- Gestion des erreurs de tokenisation CLIP
- Normalisation des embeddings

## 📝 Notes

- Les embeddings plot et poster sont stockés séparément dans l'index Annoy
- Index item = `2 * movie_id` pour plot, `2 * movie_id + 1` pour poster
- La recherche agrège les résultats des deux modalités

## 🐛 Troubleshooting

### API non disponible
Vérifier que `api_file.py` est lancé et que le port 5075 est libre

### Index introuvable
Exécuter `clip_annoy.py` pour créer l'index

### Posters manquants
Vérifier le chemin `poster_prefix` dans `config.yaml`

## 📄 License

MIT

