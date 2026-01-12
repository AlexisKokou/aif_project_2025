import os, json
import pandas as pd
import torch
from PIL import Image
from annoy import AnnoyIndex
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

CSV_PATH = os.getenv("PLOTS_CSV", "data/movie_plots.csv")
POSTER_PREFIX = os.getenv("POSTER_PREFIX", "data/content/sorted_movie_posters_paligema")

OUT_IMG_INDEX = os.getenv("CLIP_IMG_INDEX", "clip_img.ann")
OUT_TXT_INDEX = os.getenv("CLIP_TXT_INDEX", "clip_txt.ann")
OUT_META = os.getenv("CLIP_META", "clip_meta.json")

df = pd.read_csv(CSV_PATH)

# Adapte au nom réel de ta colonne plot si besoin
PLOT_COL_CANDIDATES = ["plot", "movie_plot", "description", "synopsis"]
plot_col = next((c for c in PLOT_COL_CANDIDATES if c in df.columns), None)
if plot_col is None:
    raise ValueError(f"Aucune colonne plot trouvée. Colonnes dispo: {list(df.columns)}")

poster_col = "movie_poster_path"
genre_col = "movie_category"
title_col = "title" if "title" in df.columns else None

model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(CLIP_NAME)
dim = model.config.projection_dim  # 512

img_index = AnnoyIndex(dim, "angular")
txt_index = AnnoyIndex(dim, "angular")

meta = {}

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

with torch.no_grad():
    for i, row in df.iterrows():
        rel_poster = str(row[poster_col])
        poster_path = os.path.join(POSTER_PREFIX, rel_poster)

        # --- Image embedding ---
        try:
            img = Image.open(poster_path).convert("RGB")
            inp = processor(images=img, return_tensors="pt").to(DEVICE)
            v_img = model.get_image_features(**inp)
            v_img = l2norm(v_img)[0].cpu().numpy().tolist()
            img_index.add_item(i, v_img)
        except Exception:
            # si un poster est cassé, on skip l'image (mais on garde le texte)
            pass

        # --- Text embedding (plot) ---
        plot_text = str(row[plot_col])[:2000]  # limite propre
        inp = processor(text=[plot_text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        v_txt = model.get_text_features(**inp)
        v_txt = l2norm(v_txt)[0].cpu().numpy().tolist()
        txt_index.add_item(i, v_txt)

        meta[str(i)] = {
            "title": str(row[title_col]) if title_col else f"movie_{i}",
            "genre": str(row[genre_col]) if genre_col in df.columns else None,
            "poster_path": rel_poster,  # relatif (comme tu fais déjà côté Gradio)
            "plot": plot_text
        }

img_index.build(50)
txt_index.build(50)
img_index.save(OUT_IMG_INDEX)
txt_index.save(OUT_TXT_INDEX)

with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

print("OK:", OUT_IMG_INDEX, OUT_TXT_INDEX, OUT_META, "dim=", dim)
