import os
import io
import yaml
import requests
import gradio as gr
from PIL import Image
import inspect


CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

API_URL = os.getenv("API_URL", f"http://127.0.0.1:{cfg['server']['api_port']}")

# Used for showing similar posters returned by /predict_plot (local file paths)
poster_prefix = cfg.get("data", {}).get("poster_prefix", "content/sorted_movie_posters_paligema")
if not os.path.exists(poster_prefix) and os.path.exists(os.path.join("data", poster_prefix)):
    poster_prefix = os.path.join("data", poster_prefix)

genres_mapping = cfg.get("genres", {}).get("mapping", {})
genres_inv = {v: k for k, v in genres_mapping.items()}


# -------------------------
# Helpers
# -------------------------
def _img_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _safe_md(text: str) -> str:
    return (text or "").replace("|", "\\|")


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return "API disponible"
    except Exception:
        pass
    return "API non disponible"


# -------------------------
# Part 1 — Poster: Genre prediction
# -------------------------
def predict_genre_single(image):
    if image is None:
        return "Please upload an image."
    yield "### ⏳ Processing image..."
    try:
        files = {"file": ("poster.jpg", _img_to_jpeg_bytes(image), "image/jpeg")}
        r = requests.post(f"{API_URL}/predict_poster", files=files, timeout=20)
        if r.status_code != 200:
            yield f"**API Error:** {r.status_code}\n```\n{r.text}\n```"
            return

        result = r.json()
        output = f"### Predicted Genre: **{result.get('prediction')}**\n"
        output += f"**Confidence:** {float(result.get('confidence', 0)):.2%}\n\n"
        output += "#### Top 3 Predictions:\n"
        for i, pred in enumerate(result.get("top_3_predictions", []), 1):
            bar = "█" * int(float(pred["confidence"]) * 20)
            output += f"{i}. **{pred['genre']}**: {float(pred['confidence']):.2%} `{bar}`\n"
        yield output
    except Exception as e:
        yield f"**Error:** {e}"


def predict_genre_batch(files):
    if not files:
        return "Please select at least one image."
    yield "### ⏳ Processing images..."

    try:
        req_files = []
        for f in files:
            img = Image.open(f.name).convert("RGB")
            req_files.append(("files[]", (os.path.basename(f.name), _img_to_jpeg_bytes(img), "image/jpeg")))

        r = requests.post(f"{API_URL}/batch_predict_poster", files=req_files, timeout=120)
        if r.status_code != 200:
            yield f"**API Error:** {r.status_code}\n```\n{r.text}\n```"
            return

        preds = r.json().get("predictions", [])
        md = ["### Batch Poster Genre Prediction Results\n"]
        md.append("| # | File | Predicted Genre | Confidence |")
        md.append("|---:|------|----------------|-----------:|")

        for i, p in enumerate(preds, 1):
            fn = _safe_md(p.get("filename", ""))
            pred = _safe_md(str(p.get("prediction", "N/A")))
            conf = p.get("confidence", None)
            conf_str = f"{float(conf):.2%}" if isinstance(conf, (int, float)) else "N/A"
            md.append(f"| {i} | {fn} | {pred} | {conf_str} |")

        yield "\n".join(md)

    except Exception as e:
        yield f"**Error:** {e}"


# -------------------------
# Part 2 — Poster: Validation (OOD)
# -------------------------
def validate_poster_single(image):
    if image is None:
        return "Please upload an image."
    yield "### ⏳ Processing image..."

    try:
        files = {"file": ("image.jpg", _img_to_jpeg_bytes(image), "image/jpeg")}
        r = requests.post(f"{API_URL}/validate_poster", files=files, timeout=20)
        if r.status_code != 200:
            yield f"**API Error:** {r.status_code}\n```\n{r.text}\n```"
            return

        data = r.json()
        if data.get("valid", False):
            pred = data.get("prediction_if_valid", "N/A")
            conf = data.get("confidence_if_valid", None)
            conf_str = f"{float(conf):.2%}" if isinstance(conf, (int, float)) else "N/A"
            yield (
                "### ✅ Valid Poster\n\n"
                f"**OOD Method:** {data.get('ood_method', 'entropy')}\n"
                f"**OOD Score:** {data.get('ood_score')}\n"
                f"**Threshold:** {data.get('threshold')}\n\n"
                "---\n"
                f"**Predicted Genre:** {pred}\n"
                f"**Confidence:** {conf_str}\n"
            )
        else:
            yield (
                "### ❌ Rejected Image\n\n"
                f"**OOD Method:** {data.get('ood_method', 'entropy')}\n"
                f"**OOD Score:** {data.get('ood_score')}\n"
                f"**Threshold:** {data.get('threshold')}\n\n"
                f"*Rule: {data.get('rule', 'valid_if_score <= threshold')}*\n"
            )

    except Exception as e:
        yield f"**Error:** {e}"


def validate_poster_batch(files):
    if not files:
        return "Please select at least one image."
    yield "### ⏳ Validating images (batch)..."

    rows = []
    try:
        for idx, f in enumerate(files, 1):
            img = Image.open(f.name).convert("RGB")
            files_payload = {"file": (os.path.basename(f.name), _img_to_jpeg_bytes(img), "image/jpeg")}
            r = requests.post(f"{API_URL}/validate_poster", files=files_payload, timeout=30)

            if r.status_code != 200:
                rows.append({
                    "i": idx,
                    "file": os.path.basename(f.name),
                    "valid": "❌",
                    "score": "N/A",
                    "thr": "N/A",
                    "pred": f"API error {r.status_code}",
                    "conf": "N/A",
                })
                continue

            data = r.json()
            valid = "✅" if data.get("valid", False) else "❌"
            score = data.get("ood_score", "N/A")
            thr = data.get("threshold", "N/A")
            pred = data.get("prediction_if_valid", "—") if data.get("valid", False) else "—"
            conf = data.get("confidence_if_valid", None)
            conf_str = f"{float(conf):.2%}" if isinstance(conf, (int, float)) else "—"

            rows.append({
                "i": idx,
                "file": os.path.basename(f.name),
                "valid": valid,
                "score": score,
                "thr": thr,
                "pred": pred,
                "conf": conf_str,
            })

        md = ["### Batch Poster Validation Results\n"]
        md.append("| # | File | Valid | OOD Score | Threshold | Predicted Genre (if valid) | Confidence |")
        md.append("|---:|------|:-----:|----------:|----------:|---------------------------|-----------:|")

        for r in rows:
            md.append(
                f"| {r['i']} | {_safe_md(r['file'])} | {r['valid']} | {r['score']} | {r['thr']} | "
                f"{_safe_md(str(r['pred']))} | {r['conf']} |"
            )

        yield "\n".join(md)

    except Exception as e:
        yield f"**Error:** {e}"


# -------------------------
# Part 3 — Plot
# -------------------------
import base64
from io import BytesIO

def predict_plot_single(plot_text):
    if not plot_text or not plot_text.strip():
        return "Please enter a movie plot.", None, None, None, None, None
    try:
        r = requests.post(f"{API_URL}/predict_plot", json={"plot": plot_text}, timeout=30)
    except requests.exceptions.ConnectionError:
        return "**Unable to connect to API.** Check that the API server is running.", None, None, None, None, None

    if r.status_code != 200:
        return f"**API Error:** {r.status_code}\n```\n{r.text}\n```", None, None, None, None, None

    data = r.json()
    label_idx = data.get("predicted_class")
    label_name = data.get("predicted_label")
    if label_name is None and isinstance(label_idx, int) and label_idx in genres_inv:
        label_name = genres_inv[label_idx]
    header = f"### Predicted Genre: **{label_name if label_name else label_idx}**"

    poster_images = data.get("poster_images", []) or []
    posters = []
    for img_data in poster_images[:5]:
        if img_data and img_data.startswith("data:image"):
            # Decode base64 to PIL Image
            base64_str = img_data.split(",")[1]
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_bytes))
            posters.append(img)
        else:
            posters.append(None)
    while len(posters) < 5:
        posters.append(None)

    return header, posters[0], posters[1], posters[2], posters[3], posters[4]


def batch_predict_plot(plots_text):
    plots = [p.strip() for p in (plots_text or "").split("\n") if p.strip()]
    if not plots:
        return "Please enter one plot per line."
    r = requests.post(f"{API_URL}/batch_predict_plot", json={"plots": plots}, timeout=30)
    if r.status_code != 200:
        return f"**API Error:** {r.status_code}\n```\n{r.text}\n```"
    preds = r.json().get("predicted_classes", [])
    lines = ["### Batch Plot Genre Prediction Results\n"]
    for i, c in enumerate(preds, 1):
        lines.append(f"**Plot {i}:** {genres_inv.get(c, c)}\n")
    return "\n".join(lines)


# -------------------------
# Part 4 — Discovery (CLIP)
# -------------------------
def text_search(query, k):
    if not query or not query.strip():
        return "Please enter a description.", []
    try:
        r = requests.post(f"{API_URL}/retrieve", json={"query": query, "k": int(k)}, timeout=30)
    except Exception as e:
        return f"❌ Cannot reach API at {API_URL}\n\n{e}", []
    if r.status_code != 200:
        return f"❌ API error {r.status_code}:\n\n{r.text}", []
    results = r.json().get("results", [])
    md = "### 🎯 Results\n" + "\n".join(
        [f"- **{x.get('title','')}** ({x.get('genre','')}) — match: {x.get('matched_on')}" for x in results]
    )
    gallery = [(x["poster_url"], f"{x.get('title','')} | {x.get('genre','')} | match={x.get('matched_on')}")
               for x in results if x.get("poster_url")]
    return md, gallery


def image_search(image, k):
    if image is None:
        return "Please upload an image.", []
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        files = {"file": ("query.jpg", buf.getvalue(), "image/jpeg")}
        r = requests.post(f"{API_URL}/retrieve_image", files=files, data={"k": int(k)}, timeout=60)
    except Exception as e:
        return f"❌ Cannot reach API at {API_URL}\n\n{e}", []
    if r.status_code != 200:
        return f"❌ API error {r.status_code}:\n\n{r.text}", []
    results = r.json().get("results", [])
    md = "### 🖼️ Image Search Results\n" + "\n".join(
        [f"- **{x.get('title','')}** ({x.get('genre','')}) — match: {x.get('matched_on')}" for x in results]
    )
    gallery = [(x["poster_url"], f"{x.get('title','')} | {x.get('genre','')} | match={x.get('matched_on')}")
               for x in results if x.get("poster_url")]
    return md, gallery


def chat_send(history, message, k):
    history = history or []
    if not message or not message.strip():
        return history, "", []
    try:
        r = requests.post(f"{API_URL}/chat", json={"message": message, "k": int(k)}, timeout=60)
        if r.status_code != 200:
            answer = f"❌ API error {r.status_code}: {r.text}"
            gallery = []
        else:
            data = r.json()
            answer = data.get("answer", "")
            results = data.get("results", [])
            gallery = [(res["poster_url"], f"{res.get('title','Unknown')} ({res.get('genre','N/A')})")
                       for res in results if res.get("poster_url")]
    except Exception as e:
        answer = f"❌ API not reachable: {e}"
        gallery = []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, "", gallery


# -------------------------
# Theme + App
# -------------------------
THEME = gr.themes.Soft()

APP_TITLE = "🎬 Final Project — AI Tools for a Movie Streaming Platform"

with gr.Blocks(title=APP_TITLE, theme=THEME) as demo:
    gr.Markdown(f"# {APP_TITLE}")

    gr.Markdown(
        """
    This interface covers the full final project end-to-end.

    Upload a poster to predict its genre (single or batch).
    Validate that an image is a real movie poster using OOD detection (single or batch).
    Paste a movie plot to predict its genre and get similar movie recommendations.
    Discover movies with natural language: search by text, search by image, or chat with the system.

    All features run through the REST API and can be tested directly from the tabs below.
    """
    )
    with gr.Row():
        api_status = gr.Textbox(label="🔌 API Status", value=check_api_health(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh")
    refresh_btn.click(fn=check_api_health, outputs=api_status)

    with gr.Tabs():
        # -------------------------
        # HOME
        # -------------------------
        with gr.TabItem("Home"):
            gr.Markdown("## Project Overview")
            gr.Markdown(
                "- **Part 1:** Predict movie genre from posters\n"
                "- **Part 2:** Validate whether an image is a real movie poster (OOD)\n"
                "- **Part 3:** Predict genre from plot + recommend similar movies (Annoy)\n"
                "- **Part 4:** Natural language discovery using CLIP + Annoy\n\n"
                "Use the tabs above to test each part."
            )

        # -------------------------
        # PART 1
        # -------------------------
        with gr.TabItem("Part 1 — Poster Genre"):
            with gr.Tabs():
                with gr.TabItem("Single"):
                    img1 = gr.Image(type="pil", label="Poster image (JPG/PNG)", height=320)
                    btn1 = gr.Button("🎭 Predict Genre", variant="primary")
                    out1 = gr.Markdown()
                    btn1.click(fn=predict_genre_single, inputs=img1, outputs=out1, show_progress="full")

                with gr.TabItem("Batch"):
                    files1 = gr.File(file_count="multiple", file_types=["image"], label="Poster images")
                    btn1b = gr.Button("🚀 Run Batch Prediction", variant="primary")
                    out1b = gr.Markdown()
                    btn1b.click(fn=predict_genre_batch, inputs=files1, outputs=out1b, show_progress="full")

        # -------------------------
        # PART 2
        # -------------------------
        with gr.TabItem("Part 2 — Poster Validation"):
            with gr.Tabs():
                with gr.TabItem("Single"):
                    img2 = gr.Image(type="pil", label="Image to validate (JPG/PNG)", height=320)
                    btn2 = gr.Button("✅ Validate Poster (OOD)", variant="primary")
                    out2 = gr.Markdown()
                    btn2.click(fn=validate_poster_single, inputs=img2, outputs=out2, show_progress="full")

                with gr.TabItem("Batch"):
                    files2 = gr.File(file_count="multiple", file_types=["image"], label="Images to validate")
                    btn2b = gr.Button("🚀 Run Batch Validation", variant="primary")
                    out2b = gr.Markdown()
                    btn2b.click(fn=validate_poster_batch, inputs=files2, outputs=out2b, show_progress="full")

        # -------------------------
        # PART 3
        # -------------------------
        with gr.TabItem("Part 3 — Plot Genre + Recommendations"):
            with gr.Tabs():
                with gr.TabItem("Single"):
                    plot_input = gr.Textbox(lines=8, label="Movie Plot", placeholder="Paste a movie plot here...")
                    predict_btn = gr.Button("🎯 Predict Genre", variant="primary")
                    result_output = gr.Markdown()
                    gr.Markdown("### 🎞️ Similar Movies (posters)")
                    with gr.Row():
                        poster1 = gr.Image(height=240, show_label=False)
                        poster2 = gr.Image(height=240, show_label=False)
                        poster3 = gr.Image(height=240, show_label=False)
                        poster4 = gr.Image(height=240, show_label=False)
                        poster5 = gr.Image(height=240, show_label=False)

                    predict_btn.click(
                        fn=predict_plot_single,
                        inputs=plot_input,
                        outputs=[result_output, poster1, poster2, poster3, poster4, poster5],
                        show_progress=True
                    )

                with gr.TabItem("Batch"):
                    batch_input = gr.Textbox(lines=10, label="Multiple plots (ONE per line)")
                    batch_btn = gr.Button("🚀 Predict Batch", variant="primary")
                    batch_out = gr.Markdown(label="Batch Results")
                    batch_btn.click(fn=batch_predict_plot, inputs=batch_input, outputs=batch_out, show_progress=True)

        # -------------------------
        # PART 4
        # -------------------------
        with gr.TabItem("Part 4 — Natural Language Discovery"):
            with gr.Tabs():
                with gr.TabItem("Text → Posters"):
                    query = gr.Textbox(
                        lines=2,
                        label="Describe what you want",
                        placeholder="e.g. a masked killer, a love story in Paris, alien spaceship..."
                    )
                    k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    btn = gr.Button("Search", variant="primary")
                    out_md = gr.Markdown()
                    gallery = gr.Gallery(label="Matches", columns=5, height=320)
                    btn.click(fn=text_search, inputs=[query, k], outputs=[out_md, gallery])

                with gr.TabItem("Image → Posters"):
                    img = gr.Image(type="pil", label="Upload an image", height=320)
                    k2 = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    btn2 = gr.Button("Search by image", variant="primary")
                    out_md2 = gr.Markdown()
                    gallery2 = gr.Gallery(label="Matches", columns=5, height=320)
                    btn2.click(fn=image_search, inputs=[img, k2], outputs=[out_md2, gallery2])

                with gr.TabItem("Chat (RAG)"):
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(label="Message", placeholder="I want a thriller with a masked killer")
                    k3 = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    send = gr.Button("Send", variant="primary")
                    chat_gallery = gr.Gallery(label="Recommended Movies", columns=5, height=320)
                    send.click(fn=chat_send, inputs=[chatbot, msg, k3], outputs=[chatbot, msg, chat_gallery])

        # -------------------------
        # DIAGNOSTICS
        # -------------------------
        with gr.TabItem("⚙️ Diagnostics"):
            diag_btn = gr.Button("Get /stats")
            diag_out = gr.JSON()
            diag_btn.click(fn=lambda: requests.get(f"{API_URL}/stats", timeout=5).json(), outputs=diag_out)


if __name__ == "__main__":
    print(f"UI running on http://{cfg['server']['gradio_host']}:{cfg['server']['gradio_port']} (API={API_URL})")

    launch_kwargs = dict(
        server_name=cfg["server"]["gradio_host"],
        server_port=int(cfg["server"]["gradio_port"]),
        share=False,
    )

    # Compatible launch across Gradio versions (some accept theme in launch, some don't)
    if "theme" in inspect.signature(demo.launch).parameters:
        demo.launch(**launch_kwargs, theme=THEME)
    else:
        demo.launch(**launch_kwargs)
