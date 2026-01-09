import gradio as gr
import requests
import io
import os
from PIL import Image
from config import *
API_URL = os.getenv("API_URL", "http://localhost:5075")
prefix = "data/content/sorted_movie_posters_paligema"

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
            response = requests.post(f"{API_URL}/predict_12", files=files, timeout=10)
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


def batch_predict_12(files):
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
            response = requests.post(f"{API_URL}/batch_predict_12", files=files_dict, timeout=20)

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
        return "Please enter a movie plot.", *empty_posters()
    try:
        r = requests.post(
            f"{API_URL}/predict_3",
            json={"plot": plot_text},
            timeout=10
        )
    except requests.exceptions.ConnectionError:
        return "Cannot connect to API.", *empty_posters()

    if r.status_code != 200:
        return f"API error: {r.text}", *empty_posters()

    data = r.json()
    label = data.get("predicted_class")
    paths = data.get("similar_posters", [])
    genres = data.get("genres", [])
    
    posters = [(os.path.join(prefix, path), genre) for path, genre in zip(paths, genres)]

    return (
        f"🎬 **Predicted genre:** {GENRES_MAPPING_INV[label]}",
        posters[0][0], posters[0][1],
        posters[1][0], posters[1][1],
        posters[2][0], posters[2][1],
        posters[3][0], posters[3][1],
        posters[4][0], posters[4][1],
    )


def batch_predict_3(plots_text):
    plots = [p.strip() for p in plots_text.split("\n") if p.strip()]
    if not plots:
        return "Please enter one plot per line."

    r = requests.post(
        f"{API_URL}/batch_predict_3",
        json={"plots": plots},
        timeout=10
    )

    if r.status_code != 200:
        return f"API error: {r.text}"

    preds = r.json()["predicted_classes"]
    return "\n".join([f"Plot {i+1}: {GENRES_MAPPING_INV[c]}" for i, c in enumerate(preds)])


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
            batch_btn.click(fn=batch_predict_12, inputs=file_input, outputs=batch_output,show_progress="full")

        # Second Tab: Plot Predictor
        with gr.TabItem("📝 Plot Predictor"):
            gr.Markdown("**Movie Genre Predictor**")
            gr.Markdown("Enter a movie plot and get its predicted genre.")

            with gr.Tabs():
                with gr.TabItem("📄 Single Plot"):
                    plot_input = gr.Textbox(lines=8, label="Movie Plot", placeholder="Paste a movie plot here...")
                    predict_btn2 = gr.Button("🎯 Predict Genre", variant="primary")
                    result_output = gr.Markdown()
                    gr.Markdown("### 🎞️ Related Movies")

                    with gr.Row():
                        poster1 = gr.Image(height=240)
                        poster2 = gr.Image(height=240)
                        poster3 = gr.Image(height=240)
                        poster4 = gr.Image(height=240)
                        poster5 = gr.Image(height=240)

                    with gr.Row():
                        genre1 = gr.Textbox(label="Genre", interactive=False)
                        genre2 = gr.Textbox(label="Genre", interactive=False)
                        genre3 = gr.Textbox(label="Genre", interactive=False)
                        genre4 = gr.Textbox(label="Genre", interactive=False)
                        genre5 = gr.Textbox(label="Genre", interactive=False)

                with gr.TabItem("📋 Batch"):
                    batch_input = gr.Textbox(lines=10, label="Multiple plots (ONE per line)")
                    batch_btn2 = gr.Button("🚀 Predict Batch", variant="primary")
                    batch_output2 = gr.Textbox(lines=12, label="Results", interactive=False)

            predict_btn2.click(fn=predict_plot, inputs=plot_input, 
                             outputs=[result_output, poster1, genre1, poster2, genre2, 
                                    poster3, genre3, poster4, genre4, poster5, genre5])
            batch_btn2.click(fn=batch_predict_3, inputs=batch_input, outputs=batch_output2)
if __name__ == "__main__":
    print(f"Connecting to API at: {API_URL}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
