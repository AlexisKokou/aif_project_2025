import os
import json
import argparse

import numpy as np
import torch
import yaml
from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
from annoy import AnnoyIndex

# --- Poster classifier (Part 2) ---
import torchvision.transforms as transforms
from logit_scores import entropy
from model_12 import MoviePosterNet

# --- Plot classifier (Part 3) ---
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- CLIP retrieval (Part 4) ---
import clip


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> str:
    """
    Resolve a path robustly:
    - if p exists as-is, use it
    - else try prefixed by 'data/' (common in your project)
    - else return p (will fail later with clear error)
    """
    if not p:
        return p
    if os.path.exists(p):
        return p
    alt = os.path.join("data", p)
    if os.path.exists(alt):
        return alt
    return p


def safe_join(base_dir: str, rel_path: str) -> str:
    base = os.path.abspath(base_dir)
    full = os.path.abspath(os.path.join(base, rel_path))
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError("Invalid path")
    return full


# ---------------------------
# Config + App
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.getenv("CONFIG_PATH", "config.yaml"))
args = parser.parse_args()

cfg = load_yaml(args.config)
app = Flask(__name__)

device = get_device()
print(f"[unified_api] device={device}")

# ---------------------------
# (A) Poster model (model_12)
# ---------------------------
poster_cfg = cfg.get("poster_model", {})
MODEL_12_PATH = resolve_path(poster_cfg.get("model_path_12", "weights_12/movie_poster_net.pth"))

poster_model_loaded = False
POSTER_GENRES = []
model_12 = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

try:
    print(f"[poster] loading from: {MODEL_12_PATH}")
    checkpoint = torch.load(MODEL_12_PATH, map_location=device)
    model_12 = MoviePosterNet(checkpoint["num_genres"]).to(device)
    model_12.load_state_dict(checkpoint["model_state_dict"])
    model_12.eval()
    POSTER_GENRES = checkpoint.get("classes", [])
    poster_model_loaded = True
    print(f"[poster] loaded. classes={len(POSTER_GENRES)}")
except Exception as e:
    print(f"[poster] FAILED to load: {e}")

# ---------------------------
# (B) Plot model + Annoy (Part 3)
# ---------------------------
plot_cfg = cfg.get("plot_model", {})
MODEL_3_PATH = resolve_path(plot_cfg.get("model_path_3", "weights_3"))
PLOTS_PATH = resolve_path(cfg.get("data", {}).get("plots_path", "data/movie_plots.csv"))
PLOT_ANNOY_PATH = resolve_path(plot_cfg.get("ann_path", "plot_embeddings.ann"))
PLOT_EMB_DIM = int(plot_cfg.get("embedding_dim", 768))

plot_model_loaded = False
tokenizer = None
model_3 = None
plot_df = None
plot_ann = None

DATA_CFG = cfg.get("data", {})

try:
    import pandas as pd
    plot_df = pd.read_csv(PLOTS_PATH)
    plot_ann = AnnoyIndex(PLOT_EMB_DIM, "angular")
    if not plot_ann.load(PLOT_ANNOY_PATH):
        raise RuntimeError(f"Cannot load plot annoy index: {PLOT_ANNOY_PATH}")

    from model_3.model import PlotClassification

    if os.path.exists(os.path.join(MODEL_3_PATH, "config.json")):
        base_model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_3_PATH,
            output_attentions=True,
            output_hidden_states=True
        )
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_3_PATH)
    else:
        base_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=10,
            output_attentions=True,
            output_hidden_states=True
        )
        ckpt = torch.load(MODEL_3_PATH, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model_3 = PlotClassification(base_model).to(device)
    model_3.eval()
    plot_model_loaded = True
    print("[plot] loaded.")
except Exception as e:
    print(f"[plot] FAILED to load: {e}")

# ---------------------------
# (C) OOD config
# ---------------------------
ood_cfg = cfg.get("ood", {})
OOD_METHOD = ood_cfg.get("method", "entropy")
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", ood_cfg.get("threshold", 1.05)))
OOD_CONFIG_PATH = resolve_path(ood_cfg.get("config_path", "ood_config.json"))

ood_config = {
    "method": OOD_METHOD,
    "threshold": OOD_THRESHOLD,
    "decision_rule": "valid_if_score <= threshold",
}
if os.path.exists(OOD_CONFIG_PATH):
    try:
        with open(OOD_CONFIG_PATH, "r", encoding="utf-8") as f:
            ood_config.update(json.load(f))
    except Exception as e:
        print(f"[ood] failed to load {OOD_CONFIG_PATH}: {e}")

# ---------------------------
# (D) CLIP + Annoy (Part 4)
# ---------------------------
clip_cfg = cfg.get("clip", {})
CLIP_NAME = clip_cfg.get("model_name", "ViT-B/32")
CLIP_DIM = int(clip_cfg.get("embedding_dim", 512))

idx_cfg = cfg.get("index", {})
CLIP_ANNOY_PATH = resolve_path(idx_cfg.get("annoy_path", "assets/clip_movies.ann"))
CLIP_META_PATH = resolve_path(idx_cfg.get("meta_path", "assets/clip_meta.json"))
N_CANDIDATES = int(idx_cfg.get("n_candidates", 2000))
ANNOY_SEARCH_K = int(idx_cfg.get("annoy_search_k", -1))

POSTER_PREFIX = resolve_path(DATA_CFG.get("poster_prefix", "content/sorted_movie_posters_paligema"))

clip_loaded = False
clip_index_loaded = False
clip_model = None
clip_preprocess = None
clip_ann = None
clip_meta = None

try:
    clip_model, clip_preprocess = clip.load(CLIP_NAME, device=device)
    clip_model.eval()
    clip_loaded = True
    print(f"[clip] loaded: {CLIP_NAME}")
except Exception as e:
    print(f"[clip] FAILED to load: {e}")

try:
    if not os.path.exists(CLIP_ANNOY_PATH) or not os.path.exists(CLIP_META_PATH):
        raise RuntimeError("Missing CLIP annoy/meta. Run: python clip_annoy.py")

    clip_ann = AnnoyIndex(CLIP_DIM, "angular")
    if not clip_ann.load(CLIP_ANNOY_PATH):
        raise RuntimeError(f"Cannot load CLIP annoy index: {CLIP_ANNOY_PATH}")

    with open(CLIP_META_PATH, "r", encoding="utf-8") as f:
        clip_meta = json.load(f)

    clip_index_loaded = True
    print("[clip] annoy+meta loaded.")
except Exception as e:
    print(f"[clip] index/meta FAILED: {e}")


def encode_text_query(text: str, max_words=60) -> np.ndarray:
    text = (text or "").strip()
    if not text or not clip_loaded:
        return np.zeros(CLIP_DIM, dtype=np.float32)

    try:
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb[0].detach().cpu().numpy().astype(np.float32)
    except (TypeError, RuntimeError):
        pass

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
                chunk_words = chunk_words[: len(chunk_words) // 2]

        if tokens is not None:
            with torch.no_grad():
                emb = clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=1, keepdim=True)
            embs.append(emb[0].detach().cpu().numpy())
        i = j

    if not embs:
        return np.zeros(CLIP_DIM, dtype=np.float32)

    mean = np.mean(np.stack(embs, axis=0), axis=0)
    norm = np.linalg.norm(mean) + 1e-12
    return (mean / norm).astype(np.float32)


def encode_image_query(img_pil: Image.Image) -> np.ndarray:
    if not clip_loaded:
        return np.zeros(CLIP_DIM, dtype=np.float32)
    img = clip_preprocess(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def retrieve_movies(q_emb: np.ndarray, k=5):
    if not clip_index_loaded:
        return []
    n_items = clip_ann.get_n_items()
    n = min(n_items, max(N_CANDIDATES, 2000, 100 * k))

    ids, dists = clip_ann.get_nns_by_vector(
        q_emb,
        n,
        search_k=ANNOY_SEARCH_K,
        include_distances=True
    )

    best = {}
    for item_id, dist in zip(ids, dists):
        movie_id = item_id // 2
        modality = "plot" if (item_id % 2 == 0) else "poster"
        if movie_id not in best or dist < best[movie_id]["distance"]:
            best[movie_id] = {"distance": float(dist), "matched_on": modality}

    ranked = sorted(best.items(), key=lambda x: x[1]["distance"])

    results = []
    for movie_id, info in ranked:
        m = clip_meta.get(str(movie_id), {})

        title = m.get("title") or m.get("titre") or f"movie_{movie_id}"
        genre = m.get("genre")
        poster_path = m.get("poster_path") or m.get("chemin_")
        plot = m.get("plot") or ""
        plot_snippet = (plot[:300] + "...") if len(plot) > 300 else plot

        results.append({
            "movie_id": int(movie_id),
            "distance": info["distance"],
            "matched_on": info["matched_on"],
            "title": title,
            "genre": genre,
            "poster_path": poster_path,
            "plot_snippet": plot_snippet
        })
        if len(results) >= k:
            break
    return results


def add_poster_urls(results, base_url: str):
    base = base_url.rstrip("/")
    for r in results:
        if r.get("poster_path"):
            r["poster_url"] = f"{base}/poster/{r['movie_id']}"
        else:
            r["poster_url"] = None
    return results


def maybe_rerank_for_comedy(results, user_text: str):
    t = (user_text or "").lower()
    wants_comedy = any(w in t for w in ["funny", "comedy", "humor", "humour", "hilarious"])
    if not wants_comedy:
        return results

    def key(r):
        g = str(r.get("genre") or "").lower()
        is_comedy = (g == "comedy")
        return (not is_comedy, r.get("distance", 1e9))

    return sorted(results, key=key)


# ---------------------------
# Routes (system)
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "poster_model_loaded": poster_model_loaded,
        "plot_model_loaded": plot_model_loaded,
        "clip_loaded": clip_loaded,
        "clip_index_loaded": clip_index_loaded,
        "clip_model": CLIP_NAME if clip_loaded else None,
        "clip_dim": CLIP_DIM,
    })


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "poster_classes": len(POSTER_GENRES),
        "plot_df_rows": int(len(plot_df)) if plot_df is not None else 0,
        "plot_ann_items": int(plot_ann.get_n_items()) if plot_ann is not None else 0,
        "clip_meta_movies": int(len(clip_meta)) if clip_meta is not None else 0,
        "clip_ann_items": int(clip_ann.get_n_items()) if clip_ann is not None else 0,
        "paths": {
            "poster_prefix": os.path.abspath(POSTER_PREFIX),
            "clip_annoy": os.path.abspath(CLIP_ANNOY_PATH),
            "clip_meta": os.path.abspath(CLIP_META_PATH),
            "plot_csv": os.path.abspath(PLOTS_PATH),
            "plot_annoy": os.path.abspath(PLOT_ANNOY_PATH),
        }
    })


# ---------------------------
# Routes (Part 2): Poster classification + OOD
# ---------------------------
@app.route("/predict_poster", methods=["POST"])
def predict_poster():
    if not poster_model_loaded:
        return jsonify({"error": "Poster model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        img_pil = Image.open(file.stream).convert("RGB")
        tensor = transform(img_pil).to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model_12(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0, predicted_idx].item()

        all_predictions = {g: float(probabilities[0, i]) for i, g in enumerate(POSTER_GENRES)}
        sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "prediction": POSTER_GENRES[predicted_idx] if POSTER_GENRES else int(predicted_idx),
            "confidence": round(float(confidence), 4),
            "top_3_predictions": [
                {"genre": g, "confidence": round(float(c), 4)} for g, c in sorted_preds[:3]
            ],
            "all_predictions": all_predictions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict_poster", methods=["POST"])
def batch_predict_poster():
    if not poster_model_loaded:
        return jsonify({"error": "Poster model not loaded"}), 500
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    try:
        for f in files:
            if not f.filename:
                continue
            img_pil = Image.open(f.stream).convert("RGB")
            tensor = transform(img_pil).to(device).unsqueeze(0)

            with torch.no_grad():
                outputs = model_12(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = outputs.argmax(1).item()
                confidence = probabilities[0, predicted_idx].item()

            results.append({
                "filename": f.filename,
                "prediction": POSTER_GENRES[predicted_idx] if POSTER_GENRES else int(predicted_idx),
                "confidence": round(float(confidence), 4),
            })

        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/validate_poster", methods=["POST"])
def validate_poster():
    if not poster_model_loaded:
        return jsonify({"error": "Poster model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        img_pil = Image.open(file.stream).convert("RGB")
        tensor = transform(img_pil).to(device).unsqueeze(0)

        with torch.no_grad():
            logits = model_12(tensor)

            if OOD_METHOD == "entropy":
                score = float(entropy(logits)[0])
            else:
                return jsonify({"error": f"Unsupported OOD method: {OOD_METHOD}"}), 500

            valid = bool(score <= OOD_THRESHOLD)

            probas = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(1).item()
            confidence = float(probas[0, pred_idx].item())

        return jsonify({
            "valid": valid,
            "ood_method": OOD_METHOD,
            "ood_score": round(score, 6),
            "threshold": round(OOD_THRESHOLD, 6),
            "rule": "valid_if_score <= threshold",
            "prediction_if_valid": (POSTER_GENRES[pred_idx] if POSTER_GENRES else int(pred_idx)) if valid else None,
            "confidence_if_valid": round(confidence, 4) if valid else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Routes (Part 3): Plot classification + similar posters
# ---------------------------
@app.route("/predict_plot", methods=["POST"])
def predict_plot():
    if not plot_model_loaded or tokenizer is None or plot_df is None or plot_ann is None:
        return jsonify({"error": "Plot model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    plot = (data.get("plot") or "").strip()
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    inputs = tokenizer(plot, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits, _, hidden_states = model_3(inputs["input_ids"], inputs["attention_mask"])
        pred_idx = int(torch.argmax(logits, dim=1).item())
        cls_emb = hidden_states[-1][:, 0, :]

    pred_label = None
    try:
        id2label = getattr(model_3.base_model.config, "id2label", None)
        if isinstance(id2label, dict) and pred_idx in id2label:
            pred_label = id2label[pred_idx]
    except Exception:
        pass

    vec = cls_emb[0].detach().cpu().numpy()
    nearest_indices = plot_ann.get_nns_by_vector(vec, 5)

    poster_col = DATA_CFG.get("poster_col", "movie_poster_path")
    genre_col = DATA_CFG.get("genre_col", "movie_category")
    title_col = DATA_CFG.get("title_col", "title")

    def safe_get(col, idx):
        return plot_df.iloc[idx][col] if col in plot_df.columns else None

    paths = [safe_get(poster_col, i) for i in nearest_indices]
    genres = [safe_get(genre_col, i) for i in nearest_indices]
    titles = [safe_get(title_col, i) for i in nearest_indices]

    return jsonify({
        "predicted_class": pred_idx,
        "predicted_label": pred_label,
        "similar_posters": paths,
        "genres": genres,
        "titles": titles,
    })


@app.route("/batch_predict_plot", methods=["POST"])
def batch_predict_plot():
    if not plot_model_loaded or tokenizer is None:
        return jsonify({"error": "Plot model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    plots = data.get("plots", [])
    plots = [p.strip() for p in plots if isinstance(p, str) and p.strip()]
    if not plots:
        return jsonify({"error": "No plots provided"}), 400

    inputs = tokenizer(plots, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits, _, _ = model_3(inputs["input_ids"], inputs["attention_mask"])
        predicted = torch.argmax(logits, dim=1).tolist()

    return jsonify({"predicted_classes": predicted})


# ---------------------------
# Routes (Part 4): CLIP retrieval
# ---------------------------
@app.route("/poster/<int:movie_id>", methods=["GET"])
def serve_clip_poster(movie_id: int):
    if not clip_index_loaded:
        abort(404)

    m = clip_meta.get(str(movie_id))
    if not m:
        abort(404)

    poster_rel = m.get("poster_path") or m.get("chemin_")
    if not poster_rel:
        abort(404)

    try:
        path = safe_join(POSTER_PREFIX, str(poster_rel))
    except ValueError:
        abort(404)

    if not os.path.exists(path):
        abort(404)

    return send_file(path)


@app.route("/retrieve", methods=["POST"])
def retrieve_text():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    k = int(data.get("k", 5))
    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not clip_index_loaded:
        return jsonify({"error": "CLIP index not loaded (run clip_annoy.py)"}), 500

    query_for_clip = f"a movie poster or plot about: {query}"
    q_emb = encode_text_query(query_for_clip)

    results = retrieve_movies(q_emb, k=k)
    results = add_poster_urls(results, request.host_url)
    return jsonify({"query": query, "results": results})


@app.route("/retrieve_image", methods=["POST"])
def retrieve_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    if not clip_index_loaded:
        return jsonify({"error": "CLIP index not loaded (run clip_annoy.py)"}), 500

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    k = int(request.form.get("k", 5))

    q_emb = encode_image_query(img)
    results = retrieve_movies(q_emb, k=k)
    results = add_poster_urls(results, request.host_url)
    return jsonify({"results": results})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = (data.get("message") or "").strip()
    k = int(data.get("k", 5))
    if not message:
        return jsonify({"error": "No message provided"}), 400
    if not clip_index_loaded:
        return jsonify({"error": "CLIP index not loaded (run clip_annoy.py)"}), 500

    query_for_clip = f"a movie poster or plot about: {message}"
    q_emb = encode_text_query(query_for_clip)

    results = retrieve_movies(q_emb, k=k)
    results = maybe_rerank_for_comedy(results, message)
    results = add_poster_urls(results, request.host_url)

    if not results:
        return jsonify({"answer": "I couldn’t find anything very close. Add one more detail 🙂", "results": []})

    lines = [
        f"You asked for: **{message}**\n",
        "Here’s what I found (with plot snippets):\n"
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        genre = r.get("genre", "N/A")
        matched = r.get("matched_on", "unknown")
        snip = r.get("plot_snippet", "")
        lines.append(f"**{i}) {title} — {genre}**  _(match: {matched})_")
        if snip:
            lines.append(f"> {snip}")
        if r.get("poster_url"):
            lines.append(f"Poster: {r['poster_url']}")
        lines.append("")
    lines.append("Want me to filter by genre (horror/thriller/action/romance)?")

    return jsonify({"answer": "\n".join(lines), "results": results})


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    host = cfg.get("server", {}).get("api_host", "0.0.0.0")
    port = int(cfg.get("server", {}).get("api_port", 5075))
    print(f"[unified_api] http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
