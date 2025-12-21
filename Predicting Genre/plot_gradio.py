import gradio as gr
import requests
import os
from config import *
API_URL = os.getenv("API_URL", "http://localhost:5000")
prefix = "data/content/sorted_movie_posters_paligema"
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return "🟢 API is running"
    except:
        pass
    return "🔴 API is not available"


def predict_plot(plot_text):
    if not plot_text or not plot_text.strip():
        return "Please enter a movie plot.", *empty_posters()

    try:
        r = requests.post(
            f"{API_URL}/predict",
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


def batch_predict(plots_text):
    plots = [p.strip() for p in plots_text.split("\n") if p.strip()]
    if not plots:
        return "Please enter one plot per line."

    r = requests.post(
        f"{API_URL}/batch_predict",
        json={"plots": plots},
        timeout=10
    )

    if r.status_code != 200:
        return f"API error: {r.text}"

    preds = r.json()["predicted_classes"]
    return "\n".join([f"Plot {i+1}: class {c}" for i, c in enumerate(preds)])


def empty_posters():
    return [None, ""] * 5

with gr.Blocks(title="🎬 Movie Genre Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Movie Genre Predictor")
    gr.Markdown("Enter a movie plot and get its predicted genre.")

    with gr.Row():
        api_status = gr.Textbox(
            value=check_api_health(),
            label="API Status",
            interactive=False
        )
        refresh_btn = gr.Button("Refresh")

    with gr.Tabs():

        with gr.TabItem("Single Plot"):
            plot_input = gr.Textbox(
                lines=8,
                label="Movie Plot",
                placeholder="Paste a movie plot here..."
            )
            predict_btn = gr.Button("Predict Genre", variant="primary")

            result_output = gr.Markdown()

            gr.Markdown("### 🎞️ Related Movies")

            with gr.Row():
                poster1 = gr.Image(height=240)
                poster2 = gr.Image(height=240)
                poster3 = gr.Image(height=240)
                poster4 = gr.Image(height=240)
                poster5 = gr.Image(height=240)

            with gr.Row():
                title1 = gr.Textbox(label="Title", interactive=False)
                title2 = gr.Textbox(label="Title", interactive=False)
                title3 = gr.Textbox(label="Title", interactive=False)
                title4 = gr.Textbox(label="Title", interactive=False)
                title5 = gr.Textbox(label="Title", interactive=False)


        with gr.TabItem("Batch Prediction"):
            batch_input = gr.Textbox(
                lines=10,
                label="Multiple plots (ONE per line)"
            )
            batch_btn = gr.Button("Predict Batch", variant="primary")
            batch_output = gr.Textbox(lines=12, label="Results", interactive=False)


    refresh_btn.click(fn=check_api_health, outputs=api_status)

    predict_btn.click(
        fn=predict_plot,
        inputs=plot_input,
        outputs=[
            result_output,
            poster1, title1,
            poster2, title2,
            poster3, title3,
            poster4, title4,
            poster5, title5,
        ]
    )

    batch_btn.click(
        fn=batch_predict,
        inputs=batch_input,
        outputs=batch_output
    )

    gr.Markdown("""
    ### Instructions
    - Start the Flask API first
    - Enter a full movie plot
    - One plot per line for batch mode

    ### API Endpoints
    - POST `/predict`
    - POST `/batch_predict`
    - GET `/health`
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5075,
        share=False
    )
