import gradio as gr
import requests
import io
import os
from PIL import Image

# Configuration de l'URL de l'API
API_URL = os.getenv("API_URL", "http://localhost:5075")
# URL pour le client web (JavaScript) - toujours localhost
CLIENT_API_URL = "http://localhost:5075"

def predict_genre(image):
    """
    Prédit le genre d'un poster de film.
    
    Args:
        image: Image téléchargée (format PIL)
    
    Returns:
        str: Résultat de la prédiction
    """
    if image is None:
        return "Veuillez télécharger une image"
    
    try:
        # Convertir l'image en bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
        
        # Envoyer à l'API
        files = {"file": ("poster.jpg", img_bytes, "image/jpeg")}
        
        try:
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        except requests.exceptions.ConnectionError:
            return "Impossible de se connecter à l'API. Vérifiez que le serveur API est démarré."
        
        if response.status_code == 200:
            result = response.json()
            
            # Formater la réponse
            output = f"**Genre prédit : {result['prediction']}**\n"
            output += f"**Confiance : {result['confidence']:.2%}**\n\n"
            
            output += "**Top 3 prédictions :**\n"
            for pred in result['top_3_predictions']:
                output += f"   • {pred['genre']}: {pred['confidence']:.2%}\n"
            
            return output
        else:
            return f"Erreur API: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"Erreur lors du traitement: {str(e)}"

def batch_predict(files):
    """
    Prédit le genre pour plusieurs images.
    """
    if not files:
        return "Veuillez sélectionner au moins une image"
    
    results = []
    for file in files:
        try:
            image = Image.open(file.name)
            
            # Convertir en bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()
            
            # Envoyer à l'API
            files_dict = {"files[]": (file.name, img_bytes, "image/jpeg")}
            response = requests.post(f"{API_URL}/batch_predict", files=files_dict, timeout=10)
            
            if response.status_code == 200:
                batch_result = response.json()
                if batch_result.get('predictions'):
                    pred = batch_result['predictions'][0]
                    results.append(f"{os.path.basename(file.name)}: {pred['prediction']} ({pred['confidence']:.2%})")
            else:
                results.append(f"{os.path.basename(file.name)}: Erreur API")
                
        except Exception as e:
            results.append(f"{os.path.basename(file.name)}: {str(e)}")
    
    return "\n".join(results) if results else "Aucun résultat"

def check_api_health():
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"API disponible\nGenres: {', '.join(data.get('available_genres', []))}"
    except:
        pass
    return "API non disponible"

# Création de l'interface Gradio
with gr.Blocks(title="Prédicteur de Genre de Films", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Prédicteur de Genre de Film")
    gr.Markdown("Téléchargez une affiche de film pour prédire son genre.")
    
    # Section de statut
    with gr.Row():
        status = gr.Textbox(
            label="Statut de l'API",
            value=check_api_health(),
            interactive=False
        )
        refresh_btn = gr.Button("Rafraîchir")
    
    # Onglets
    with gr.Tabs():
        with gr.TabItem("Prédiction simple"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="pil",
                        label="Téléchargez une affiche de film",
                        height=300
                    )
                    predict_btn = gr.Button("Prédire le genre", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(
                        label="Résultat",
                        lines=10,
                        interactive=False
                    )
            
            # Exemples
            gr.Examples(
                examples=[],  # Tu peux ajouter des exemples plus tard
                inputs=image_input,
                label="Exemples (à configurer)"
            )
        
        with gr.TabItem("Prédiction par lot"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Sélectionnez plusieurs images",
                        height=200
                    )
                    batch_btn = gr.Button("Prédire en lot", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Textbox(
                        label="Résultats",
                        lines=15,
                        interactive=False
                    )
    
    # Événements
    refresh_btn.click(
        fn=check_api_health,
        outputs=status
    )
    
    predict_btn.click(
        fn=predict_genre,
        inputs=image_input,
        outputs=output
    )
    
    batch_btn.click(
        fn=batch_predict,
        inputs=file_input,
        outputs=batch_output
    )
    
    # Instructions
    gr.Markdown("""
    ## Instructions
    1. Assurez-vous que l'API est démarrée (statut vert ci-dessus)
    2. Téléchargez une affiche de film au format JPG/PNG
    3. Cliquez sur "Prédire le genre"
    
    ## API Endpoints
    - `POST /predict` : Prédiction single image
    - `POST /batch_predict` : Prédiction multiple images
    - `GET /health` : Vérification statut
    - `GET /genres` : Liste des genres supportés
    """)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    print(f"Connecting to API at: {API_URL}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )