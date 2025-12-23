import gradio as gr
import requests
import io
import os
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:5075")


def predict_genre(image):
    if image is None:
        return "Veuillez télécharger une image."

    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        files = {"file": ("poster.jpg", img_bytes, "image/jpeg")}

        try:
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        except requests.exceptions.ConnectionError:
            return "Impossible de se connecter à l'API. Vérifiez que le serveur API est démarré."

        if response.status_code == 200:
            result = response.json()
            output = f"Genre prédit : {result['prediction']}\n"
            output += f"Confiance : {result['confidence']:.2%}\n\n"
            output += "Top 3 prédictions :\n"
            for pred in result.get('top_3_predictions', []):
                output += f"- {pred['genre']}: {pred['confidence']:.2%}\n"
            return output

        return f"Erreur API: {response.status_code}\n{response.text}"

    except Exception as e:
        return f"Erreur lors du traitement: {str(e)}"


def validate_poster(image):
    if image is None:
        return "Veuillez télécharger une image."

    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        try:
            response = requests.post(f"{API_URL}/validate_poster", files=files, timeout=10)
        except requests.exceptions.ConnectionError:
            return "Impossible de se connecter à l'API. Vérifiez que le serveur API est démarré."

        if response.status_code == 200:
            r = response.json()

            if r.get("valid", False):
                pred = r.get("prediction_if_valid", "N/A")
                conf = r.get("confidence_if_valid", None)
                conf_str = f"{conf:.2%}" if isinstance(conf, (int, float)) else "N/A"

                return (
                    "Poster valide\n\n"
                    f"Méthode: {r.get('ood_method', 'entropy')}\n"
                    f"Score OOD: {r.get('ood_score')}\n"
                    f"Seuil: {r.get('threshold')}\n\n"
                    f"Genre (si valide): {pred}\n"
                    f"Confiance: {conf_str}\n"
                )

            return (
                "Image rejetée (pas un poster)\n\n"
                f"Méthode: {r.get('ood_method', 'entropy')}\n"
                f"Score OOD: {r.get('ood_score')}\n"
                f"Seuil: {r.get('threshold')}\n\n"
                f"Règle: {r.get('rule', 'valid_if_score <= threshold')}\n"
            )

        return f"Erreur API: {response.status_code}\n{response.text}"

    except Exception as e:
        return f"Erreur lors du traitement: {str(e)}"


def batch_predict(files):
    if not files:
        return "Veuillez sélectionner au moins une image."

    results = []
    for file in files:
        try:
            image = Image.open(file.name).convert("RGB")

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            files_dict = {"files[]": (file.name, img_bytes, "image/jpeg")}
            response = requests.post(f"{API_URL}/batch_predict", files=files_dict, timeout=20)

            if response.status_code == 200:
                batch_result = response.json()
                if batch_result.get('predictions'):
                    pred = batch_result['predictions'][0]
                    results.append(f"{os.path.basename(file.name)}: {pred['prediction']} ({pred['confidence']:.2%})")
                else:
                    results.append(f"{os.path.basename(file.name)}: Pas de prédiction retournée")
            else:
                results.append(f"{os.path.basename(file.name)}: Erreur API {response.status_code}")

        except Exception as e:
            results.append(f"{os.path.basename(file.name)}: {str(e)}")

    return "\n".join(results) if results else "Aucun résultat."


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return "API disponible"
    except Exception:
        pass
    return "API non disponible"


with gr.Blocks(title="Movie Tools") as demo:
    gr.Markdown("# AI Tools for a Movie Streaming Platform")
    gr.Markdown("Partie 1: prédiction de genre. Partie 2: validation poster (OOD).")

    with gr.Row():
        status = gr.Textbox(label="Statut de l'API", value=check_api_health(), interactive=False)
        refresh_btn = gr.Button("Rafraîchir")

    with gr.Tabs():
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Image (JPG/PNG)", height=300)
                    predict_btn = gr.Button("Prédire le genre")
                    validate_btn = gr.Button("Valider le poster (OOD)")

                with gr.Column():
                    prediction_output = gr.Textbox(label="Prédiction", lines=10, interactive=False)
                    validation_output = gr.Textbox(label="Validation poster", lines=10, interactive=False)

        with gr.TabItem("Batch"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(file_count="multiple", file_types=["image"], label="Images")
                    batch_btn = gr.Button("Prédire (batch)")
                with gr.Column():
                    batch_output = gr.Textbox(label="Résultats batch", lines=15, interactive=False)

    refresh_btn.click(fn=check_api_health, outputs=status)
    predict_btn.click(fn=predict_genre, inputs=image_input, outputs=prediction_output)
    validate_btn.click(fn=validate_poster, inputs=image_input, outputs=validation_output)
    batch_btn.click(fn=batch_predict, inputs=file_input, outputs=batch_output)

    gr.Markdown("""
Endpoints API
- POST /predict
- POST /batch_predict
- POST /validate_poster
- GET /health
- GET /genres
""")

if __name__ == "__main__":
    print(f"Connecting to API at: {API_URL}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
