import os, json
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
from annoy import AnnoyIndex
import clip
import yaml


device = torch.device('cpu')

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

CLIP_NAME = cfg["clip"]["model_name"]
DIM = int(cfg["clip"]["embedding_dim"])

ANNOY_PATH = cfg["index"]["annoy_path"]
META_PATH = cfg["index"]["meta_path"]

# Combien de voisins on récupère dans Annoy avant d’agréger (plus grand => meilleur recall)
N_CANDIDATES = int(cfg["index"].get("n_candidates", 2000))
ANNOY_SEARCH_K = int(cfg["index"].get("annoy_search_k", -1))


POSTER_PREFIX = cfg["data"]["poster_prefix"]

app = Flask(__name__)

# --- Load CLIP (TP style)
model, preprocess = clip.load(CLIP_NAME, device=device)
model.eval()

# --- Load Annoy + meta
ann = AnnoyIndex(DIM, "angular")
if not ann.load(ANNOY_PATH):
    raise RuntimeError(f"Cannot load annoy index: {ANNOY_PATH}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)


# --- helper to support old/new meta keys
def meta_title(m):  return m.get("title") or m.get("titre") or None
def meta_genre(m):  return m.get("genre") or None
def meta_poster(m): return m.get("poster_path") or m.get("chemin_") or None
def meta_plot(m):   return m.get("plot") or None


def encode_text_query(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros(DIM, dtype=np.float32)

    try:
        tokens = clip.tokenize([text], truncate=True).to(device)
    except TypeError:
        text = text[:200]
        tokens = clip.tokenize([text]).to(device)
    except RuntimeError:
        text = text[:200]
        tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def encode_image_query(img_pil: Image.Image) -> np.ndarray:
    img = preprocess(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def retrieve_movies(q_emb: np.ndarray, k=5):
    # On demande LARGE pour éviter de tomber toujours sur le même film
    n = min(ann.get_n_items(), max(N_CANDIDATES, 2000, 100 * k))

    ids, dists = ann.get_nns_by_vector(
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
        m = meta.get(str(movie_id), {})
        plot = meta_plot(m) or ""
        results.append({
            "movie_id": int(movie_id),
            "distance": info["distance"],
            "matched_on": info["matched_on"],
            "title": meta_title(m) or f"movie_{movie_id}",
            "genre": meta_genre(m),
            "poster_path": meta_poster(m),
            "plot_snippet": (plot[:300] + "...") if len(plot) > 300 else plot
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device), "clip_model": CLIP_NAME, "annoy_dim": DIM})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "meta_movies": len(meta),
        "annoy_items": ann.get_n_items(),
        "annoy_path": os.path.abspath(ANNOY_PATH),
        "meta_path": os.path.abspath(META_PATH),
        "poster_prefix": os.path.abspath(POSTER_PREFIX),
        "n_candidates": N_CANDIDATES,
        "annoy_search_k": ANNOY_SEARCH_K,
    })


# ✅ Serve posters over HTTP (needed for Gradio web display)
@app.route("/poster/<int:movie_id>", methods=["GET"])
def poster(movie_id: int):
    m = meta.get(str(movie_id))
    if not m:
        abort(404)

    poster_rel = meta_poster(m)
    if not poster_rel:
        abort(404)

    path = os.path.abspath(os.path.join(POSTER_PREFIX, poster_rel))
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

    q_emb = encode_text_query(query)
    results = retrieve_movies(q_emb, k=k)
    results = add_poster_urls(results, request.host_url)
    return jsonify({"query": query, "results": results})


@app.route("/retrieve_image", methods=["POST"])
def retrieve_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

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

    q_emb = encode_text_query(message)
    results = retrieve_movies(q_emb, k=k)
    results = add_poster_urls(results, request.host_url)

    if not results:
        answer = "Je ne trouve rien de très proche. Ajoute un détail (genre, ambiance, personnage, lieu) 🙂"
    else:
        lines = ["Voici des suggestions basées sur ta description :\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}) {r.get('title','')} — {r.get('genre','')} (match: {r.get('matched_on')})")
        lines.append("\nTu veux que je filtre par genre (horror/thriller/action/romance) ?")
        answer = "\n".join(lines)

    return jsonify({"answer": answer, "results": results})

@app.route("/debug_ids", methods=["POST"])
def debug_ids():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query"}), 400

    q_emb = encode_text_query(query)
    ids = ann.get_nns_by_vector(q_emb, 30)
    movie_ids = [i // 2 for i in ids]
    return jsonify({
        "query": query,
        "emb_norm": float(np.linalg.norm(q_emb)),
        "ids": ids,
        "movie_ids": movie_ids,
        "unique_movies": len(set(movie_ids)),
        "n_candidates_cfg": N_CANDIDATES,
        "annoy_search_k_cfg": ANNOY_SEARCH_K
    })



if __name__ == "__main__":
    host = cfg["server"]["api_host"]
    port = int(cfg["server"]["api_port"])
    print(f"Starting Part4 API on http://{host}:{port} (device={device})")
    app.run(host=host, port=port, debug=True)
