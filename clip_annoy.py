import os, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from annoy import AnnoyIndex
import clip
import yaml
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

CLIP_NAME = cfg["clip"]["model_name"]
DIM = int(cfg["clip"]["embedding_dim"])

PLOTS_CSV = cfg["data"]["plots_csv"]
PLOT_COL_CANDS = cfg["data"]["plot_col_candidates"]
POSTER_PREFIX = cfg["data"]["poster_prefix"]

ANNOY_PATH = cfg["index"]["annoy_path"]
META_PATH  = cfg["index"]["meta_path"]
NUM_TREES  = int(cfg["index"].get("num_trees", 50))

os.makedirs(Path(ANNOY_PATH).parent, exist_ok=True)

if not os.path.exists(PLOTS_CSV):
    raise FileNotFoundError(
        f"CSV introuvable: {PLOTS_CSV}\n"
        f"➡️ Mets ton fichier au bon endroit ou modifie data.plots_csv dans config.yaml"
    )

df = pd.read_csv(PLOTS_CSV)

# Détection colonne plot
plot_col = None
for c in PLOT_COL_CANDS:
    if c in df.columns:
        plot_col = c
        break
if plot_col is None:
    raise ValueError(f"Aucune colonne plot trouvée. Colonnes: {list(df.columns)}")

# Colonnes attendues pour poster/genre
POSTER_COL = "movie_poster_path"
GENRE_COL = "movie_category"
TITLE_COL = "title" if "title" in df.columns else None

if POSTER_COL not in df.columns:
    raise ValueError(f"Colonne {POSTER_COL} manquante dans le CSV.")

model, preprocess = clip.load(CLIP_NAME, device=device)
model.eval()

index = AnnoyIndex(DIM, "angular")
meta = {}

def unit_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

def encode_text_safe(text: str, max_words=60) -> np.ndarray:
    """
    Encodage CLIP texte robuste: découpe en chunks et réduit si tokenize trop long.
    On moyenne puis on renormalise.
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(DIM, dtype=np.float32)

    words = text.split()
    embs = []
    i = 0
    while i < len(words):
        j = min(i + max_words, len(words))
        chunk_words = words[i:j]

        while True:
            chunk = " ".join(chunk_words)
            try:
                tokens = clip.tokenize([chunk]).to(device)
                break
            except RuntimeError:
                if len(chunk_words) <= 5:
                    tokens = None
                    break
                chunk_words = chunk_words[: len(chunk_words)//2]

        if tokens is not None:
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=1, keepdim=True)
            embs.append(emb[0].detach().cpu().numpy())

        i = j

    if not embs:
        return np.zeros(DIM, dtype=np.float32)

    mean = np.mean(np.stack(embs, axis=0), axis=0)
    return unit_norm(mean)

def encode_image(path: str) -> np.ndarray:
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)

missing_images = 0

for movie_id, row in df.iterrows():
    plot_text = str(row[plot_col])
    poster_rel = str(row[POSTER_COL])
    poster_path = os.path.join(POSTER_PREFIX, poster_rel)

    # (A) plot embedding
    t_emb = encode_text_safe(plot_text, max_words=60)
    index.add_item(2 * movie_id, t_emb)

    # (B) poster embedding
    try:
        i_emb = encode_image(poster_path)
        index.add_item(2 * movie_id + 1, i_emb)
    except Exception:
        missing_images += 1

    meta[str(movie_id)] = {
        "title": str(row[TITLE_COL]) if TITLE_COL else f"movie_{movie_id}",
        "genre": str(row[GENRE_COL]) if GENRE_COL in df.columns else None,
        "poster_path": poster_rel,
        "plot": plot_text
    }

index.build(NUM_TREES)
index.save(ANNOY_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

print(f"Built single CLIP Annoy index: {ANNOY_PATH}")
print(f"Meta saved: {META_PATH}")
print(f"Movies: {len(df)} | Missing posters: {missing_images}")
