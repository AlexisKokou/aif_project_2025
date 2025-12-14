import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import MoviePosterNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"API running on device: {device}")

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/movie_poster_net.pth', help='model path')
args = parser.parse_args()

print(f"Loading model from: {args.model_path}")
try:
    # Charger le checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Créer le modèle
    model = MoviePosterNet(checkpoint['num_genres']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Récupérer les classes
    genres = checkpoint['classes']
    class_to_idx = checkpoint['class_to_idx']
    
    print(f"Model loaded successfully!")
    print(f"Genres available: {list(genres)}")
    print(f"Class mapping: {class_to_idx}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Transformations (doivent correspondre à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Ouvrir l'image
        img_pil = Image.open(file.stream).convert('RGB')
        
        # Appliquer les transformations
        tensor = transform(img_pil).to(device)
        tensor = tensor.unsqueeze(0)  # Ajouter dimension batch
        
        # Prédiction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Préparer les résultats
        all_predictions = {}
        for i, genre in enumerate(genres):
            all_predictions[genre] = float(probabilities[0][i])
        
        # Trier par confidence décroissante
        sorted_predictions = sorted(
            all_predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return jsonify({
            'prediction': genres[predicted_idx],
            'confidence': round(confidence, 4),
            'top_3_predictions': [
                {'genre': genre, 'confidence': round(conf, 4)} 
                for genre, conf in sorted_predictions[:3]
            ],
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    results = []
    try:
        for file in files:
            if file.filename == '':
                continue
                
            # Traiter chaque image
            img_pil = Image.open(file.stream).convert('RGB')
            tensor = transform(img_pil).to(device)
            tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = outputs.argmax(1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            results.append({
                'filename': file.filename,
                'prediction': genres[predicted_idx],
                'confidence': round(confidence, 4)
            })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'available_genres': list(genres),
        'num_genres': len(genres)
    })

@app.route('/genres', methods=['GET'])
def get_genres():
    """Retourne la liste des genres disponibles"""
    return jsonify({
        'genres': list(genres),
        'class_mapping': class_to_idx
    })

if __name__ == "__main__":
    print("\nStarting Flask API server...")
    print("   Endpoints:")
    print("   - POST /predict : Prédire le genre d'une image")
    print("   - POST /batch_predict : Prédire plusieurs images")
    print("   - GET /health : Vérifier l'état du serveur")
    print("   - GET /genres : Liste des genres disponibles")
    print(f"\nServer running on http://0.0.0.0:5075")
    app.run(host='0.0.0.0', port=5075, debug=False)
