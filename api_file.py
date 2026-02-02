import os
import json
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
from annoy import AnnoyIndex
import clip
import yaml


# ---------------------------
# Device
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()

# ---------------------------
# Config
# ---------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

CLIP_NAME = cfg["clip"]["model_name"]
DIM = int(cfg["clip"]["embedding_dim"])

ANNOY_PATH = cfg["index"]["annoy_path"]
META_PATH = cfg["index"]["meta_path"]

# runtime recall/latency knobs
N_CANDIDATES = int(cfg["index"].get("n_candidates", 2000))
ANNOY_SEARCH_K = int(cfg["index"].get("annoy_search_k", -1))

POSTER_PREFIX = cfg["data"]["poster_prefix"]

if not os.path.exists(ANNOY_PATH):
    raise RuntimeError(f"Missing Annoy index: {ANNOY_PATH}")
if not os.path.exists(META_PATH):
    raise RuntimeError(f"Missing meta json: {META_PATH}")

# ---------------------------
# App
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load CLIP (TP style)
# ---------------------------
model, preprocess = clip.load(CLIP_NAME, device=device)
model.eval()

# ---------------------------
# Load Annoy + meta
# ---------------------------
ann = AnnoyIndex(DIM, "angular")
if not ann.load(ANNOY_PATH):
    raise RuntimeError(f"Cannot load annoy index: {ANNOY_PATH}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)


# ---------------------------
# Helpers
# ---------------------------
def safe_join(base_dir: str, rel_path: str) -> str:
    """
    Build a safe absolute path under base_dir (avoid path traversal).
    """
    base = os.path.abspath(base_dir)
    full = os.path.abspath(os.path.join(base, rel_path))
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError("Invalid path")
    return full


def encode_text_query(text: str, max_words=60) -> np.ndarray:
    """
    Robust CLIP text encoding:
    - try tokenize(truncate=True) if available
    - else chunking to avoid "context length 77"
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(DIM, dtype=np.float32)

    # quick path
    try:
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb[0].detach().cpu().numpy().astype(np.float32)
    except (TypeError, RuntimeError):
        pass

    # fallback chunking
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
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=1, keepdim=True)
            embs.append(emb[0].detach().cpu().numpy())

        i = j

    if not embs:
        return np.zeros(DIM, dtype=np.float32)

    mean = np.mean(np.stack(embs, axis=0), axis=0)
    norm = np.linalg.norm(mean) + 1e-12
    return (mean / norm).astype(np.float32)


def encode_image_query(img_pil: Image.Image) -> np.ndarray:
    img = preprocess(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def retrieve_movies(q_emb: np.ndarray, k=5):
    """
    One Annoy index containing 2 items per movie:
      item 2*movie_id   = plot embedding
      item 2*movie_id+1 = poster embedding
    We retrieve many neighbors then aggregate by movie_id, keeping best distance.
    """
    n_items = ann.get_n_items()
    n = min(n_items, max(N_CANDIDATES, 2000, 100 * k))

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
    """
    Very light heuristic: if user asks something funny, boost comedy.
    Keeps CLIP distance as tie-breaker.
    """
    t = (user_text or "").lower()
    wants_comedy = any(w in t for w in ["drôle", "drole", "comique", "humour", "funny", "comedy"])
    if not wants_comedy:
        return results

    def key(r):
        g = str(r.get("genre") or "").lower()
        is_comedy = (g == "comedy")
        # sort: comedy first (False), then by distance
        return (not is_comedy, r.get("distance", 1e9))

    return sorted(results, key=key)


# ---------------------------
# Routes
# ---------------------------
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


@app.route("/poster/<int:movie_id>", methods=["GET"])
def poster(movie_id: int):
    m = meta.get(str(movie_id))
    if not m:
        abort(404)

    poster_rel = m.get("poster_path") or m.get("chemin_")
    if not poster_rel:
        abort(404)

    try:
        path = safe_join(POSTER_PREFIX, poster_rel)
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

    # ✅ prompt boost (helps CLIP for vague queries)
    query_for_clip = f"a movie poster or plot about: {query}"
    q_emb = encode_text_query(query_for_clip)

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
    """
    RAG minimal mais explicite:
    - retrieval CLIP/Annoy
    - réponse augmentée avec plot_snippet des films récupérés
    - re-ranking léger si user demande une comédie
    """
    data = request.json or {}
    message = (data.get("message") or "").strip()
    k = int(data.get("k", 5))
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # ✅ prompt boost (helps CLIP for vague queries)
    query_for_clip = f"a movie poster or plot about: {message}"
    q_emb = encode_text_query(query_for_clip)

    results = retrieve_movies(q_emb, k=k)
    results = maybe_rerank_for_comedy(results, message)
    results = add_poster_urls(results, request.host_url)

    if not results:
        answer = "Je ne trouve rien de très proche. Ajoute un détail (genre, ambiance, personnage, lieu) 🙂"
        return jsonify({"answer": answer, "results": []})

    lines = [
        f"Tu as demandé : **{message}**\n",
        "Voici ce que j’ai trouvé (avec extraits de plots) :\n"
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

    lines.append("Tu veux que je filtre par genre (horror/thriller/action/romance) ?")
    answer = "\n".join(lines)

    return jsonify({"answer": answer, "results": results})


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    host = cfg["server"]["api_host"]
    port = int(cfg["server"]["api_port"])
    print(f"Starting Part4 API on http://{host}:{port} (device={device})")
    app.run(host=host, port=port, debug=False)
