import gradio as gr
import requests
import io
import os
import yaml
from PIL import Image
API_URL = os.getenv("API_URL", "http://localhost:5075")
prefix = "data/content/sorted_movie_posters_paligema"
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
# ---- Genres ----
GENRES_MAPPING = cfg["genres"]["mapping"]
GENRES_MAPPING_INV = {v: k for k, v in GENRES_MAPPING.items()}
def predict_genre(image):
    if image is None:
        return "Please upload an image."
    
    # This will show briefly while processing
    yield "### ⏳ Processing image..."
    
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
        files = {"file": ("poster.jpg", img_bytes, "image/jpeg")}

        try:
            response = requests.post(f"{API_URL}/predict_poster", files=files, timeout=10)
        except requests.exceptions.ConnectionError:
            yield "**Unable to connect to API.** Check that the API server is running."
            return

        if response.status_code == 200:
            result = response.json()
            output = f"### Predicted Genre: **{result['prediction']}**\n"
            output += f"**Confidence:** {result['confidence']:.2%}\n\n"
            output += "#### Top 3 Predictions:\n"
            for i, pred in enumerate(result.get('top_3_predictions', []), 1):
                bar = "█" * int(pred['confidence'] * 20)
                output += f"{i}. **{pred['genre']}**: {pred['confidence']:.2%} `{bar}`\n"
            yield output
            return

        yield f"**API Error:** {response.status_code}\n```\n{response.text}\n```"

    except Exception as e:
        yield f"**Error:** {str(e)}"

def validate_poster(image):
    if image is None:
        return "Please upload an image."
    yield "### ⏳ Processing image..."
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        try:
            response = requests.post(f"{API_URL}/validate_poster", files=files, timeout=10)
        except requests.exceptions.ConnectionError:
            yield "**Unable to connect to API.** Check that the API server is running."

        if response.status_code == 200:
            r = response.json()

            if r.get("valid", False):
                pred = r.get("prediction_if_valid", "N/A")
                conf = r.get("confidence_if_valid", None)
                conf_str = f"{conf:.2%}" if isinstance(conf, (int, float)) else "N/A"

                yield (
                    "### Valid Poster\n\n"
                    f"**OOD Method:** {r.get('ood_method', 'entropy')}\n"
                    f"**OOD Score:** {r.get('ood_score')}\n"
                    f"**Threshold:** {r.get('threshold')}\n\n"
                    "---\n"
                    f"**Genre:** {pred}\n"
                    f"**Confidence:** {conf_str}\n"
                )

            yield (
                "### Image Rejected\n\n"
                f"**OOD Method:** {r.get('ood_method', 'entropy')}\n"
                f"**OOD Score:** {r.get('ood_score')}\n"
                f"**Threshold:** {r.get('threshold')}\n\n"
                f"*{r.get('rule', 'valid_if_score <= threshold')}*\n"
            )
            return

        yield f"**API Error:** {response.status_code}\n```\n{response.text}\n```"

    except Exception as e:
        yield f"**Error:** {str(e)}"


def batch_predict_poster(files):
    if not files:
        return "Please select at least one image."
    yield "### ⏳ Processing images..."
    results = ["### Batch Prediction Results\n"]
    for idx, file in enumerate(files, 1):
        try:
            image = Image.open(file.name).convert("RGB")

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            files_dict = {"files[]": (file.name, img_bytes, "image/jpeg")}
            response = requests.post(f"{API_URL}/batch_predict_poster", files=files_dict, timeout=20)

            if response.status_code == 200:
                batch_result = response.json()
                if batch_result.get('predictions'):
                    pred = batch_result['predictions'][0]
                    results.append(f"**{idx}. {os.path.basename(file.name)}**  \n{pred['prediction']} • {pred['confidence']:.2%}\n")
                else:
                    results.append(f"**{idx}. {os.path.basename(file.name)}**  \nNo prediction returned\n")
            else:
                results.append(f"**{idx}. {os.path.basename(file.name)}**  \nAPI Error {response.status_code}\n")

        except Exception as e:
            results.append(f"**{idx}. {os.path.basename(file.name)}**  \nError: {str(e)}\n")

    yield "\n".join(results) if len(results) > 1 else "No results."
def predict_plot(plot_text):
    if not plot_text or not plot_text.strip():
        return "Please enter a movie plot.", None, None, None, None, None
    
    try:
        r = requests.post(
            f"{API_URL}/predict_plot",
            json={"plot": plot_text},
            timeout=10
        )
    except requests.exceptions.ConnectionError:
        return "**Unable to connect to API.** Check that the API server is running.", None, None, None, None, None
    
    if r.status_code != 200:
        return f"**API Error:** {r.status_code}\n```\n{r.text}\n```", None, None, None, None, None
    
    data = r.json()
    label = data.get("predicted_class")
    paths = data.get("similar_posters", [])
    genres = data.get("genres", [])
    
    posters = [os.path.join(prefix, path) if path else None for path in paths[:5]]
    while len(posters) < 5:
        posters.append(None)
    
    result_text = f"### Predicted Genre: **{GENRES_MAPPING_INV[label]}**"
    
    return result_text, posters[0], posters[1], posters[2], posters[3], posters[4]


def batch_predict_plot(plots_text):
    plots = [p.strip() for p in plots_text.split("\n") if p.strip()]
    
    if not plots:
        return "Please enter one plot per line."
    
    try:
        r = requests.post(
            f"{API_URL}/batch_predict_plot",
            json={"plots": plots},
            timeout=10
        )
    except requests.exceptions.ConnectionError:
        return "**Unable to connect to API.** Check that the API server is running."
    
    if r.status_code != 200:
        return f"**API Error:** {r.status_code}\n```\n{r.text}\n```"
    
    preds = r.json()["predicted_classes"]
    
    results = ["### Batch Prediction Results\n"]
    for i, c in enumerate(preds, 1):
        results.append(f"**Plot {i}:** {GENRES_MAPPING_INV[c]}\n")
    
    return "\n".join(results)

def empty_posters():
    return [None, ""] * 5

def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return "API disponible"
    except Exception:
        pass
    return "API non disponible"

with gr.Blocks(title="Movie Tools", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Movie Tools")
    
    with gr.Row():
        api_status = gr.Textbox(label="🔌 API Status", value=check_api_health(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh")
    
    refresh_btn.click(fn=check_api_health, outputs=api_status)
    
    with gr.Tabs():
        # First Tab: Movie Tools
        with gr.TabItem("🎨 Movie Tools"):
            gr.Markdown("**AI-Powered Movie Poster Analysis**")
            gr.Markdown("""
            **How to use:**
            - **Image tab:** Upload a single movie poster to predict its genre or validate if it's a legitimate poster (OOD detection)
            - **Batch tab:** Upload multiple posters at once for bulk genre prediction
            """)

            with gr.Tabs():
                with gr.TabItem("📷 Image"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(type="pil", label="Image (JPG/PNG)", height=300)
                            validate_btn = gr.Button("✅ Validate Poster (OOD)")
                            predict_btn = gr.Button("🎭 Predict Genre")

                        with gr.Column():
                            validation_output = gr.Markdown(label="✅ Poster Validation")
                            prediction_output = gr.Markdown(label="🎭 Genre Prediction")

                with gr.TabItem("📦 Batch"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(file_count="multiple", file_types=["image"], label="Images")
                            batch_btn = gr.Button("🚀 Predict (batch)")
                        with gr.Column():
                            batch_output = gr.Markdown(label="📦 Batch Results")

            predict_btn.click(fn=predict_genre, inputs=image_input, outputs=prediction_output,show_progress="full")
            validate_btn.click(fn=validate_poster, inputs=image_input, outputs=validation_output,show_progress="full")
            batch_btn.click(fn=batch_predict_poster, inputs=file_input, outputs=batch_output,show_progress="full")

        # Second Tab: Plot Predictor
        with gr.TabItem("📝 Plot Predictor"):
            gr.Markdown("**AI-Powered Plot Genre Prediction**")
            gr.Markdown("""
            **How to use:**
            - **Single Plot tab:** Enter a movie plot to predict its genre and discover similar movies
            - **Batch tab:** Enter multiple plots (one per line) for bulk genre prediction
            """)

            with gr.Tabs():
                with gr.TabItem("📄 Single Plot"):
                    plot_input = gr.Textbox(lines=8, label="Movie Plot", placeholder="Paste a movie plot here...")
                    predict_btn2 = gr.Button("🎯 Predict Genre", variant="primary")
                    result_output = gr.Markdown()
                    gr.Markdown("### 🎞️ Related Movies")

                    with gr.Row():
                        poster1 = gr.Image(height=240, show_label=False)
                        poster2 = gr.Image(height=240, show_label=False)
                        poster3 = gr.Image(height=240, show_label=False)
                        poster4 = gr.Image(height=240, show_label=False)
                        poster5 = gr.Image(height=240, show_label=False)

                with gr.TabItem("📋 Batch"):
                    batch_input = gr.Textbox(lines=10, label="Multiple plots (ONE per line)")
                    batch_btn2 = gr.Button("🚀 Predict Batch", variant="primary")
                    batch_output2 = gr.Markdown(label="📋 Batch Results")

            predict_btn2.click(
                fn=predict_plot, 
                inputs=plot_input, 
                outputs=[result_output, poster1, poster2, poster3, poster4, poster5],
                show_progress=True
            )
            batch_btn2.click(
                fn=batch_predict_plot, 
                inputs=batch_input, 
                outputs=batch_output2,
                show_progress=True
            )
if __name__ == "__main__":
    print(f"Connecting to API at: {API_URL}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
