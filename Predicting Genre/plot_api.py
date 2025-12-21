import torch
from flask import Flask, request, jsonify
from model.model import PlotClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse
from annoy import AnnoyIndex
from config import *
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
args = parser.parse_args()
model_path = args.model_path
import pandas as pd
plots_path = "data/movie_plots.csv"
df = pd.read_csv(plots_path)

print(f"Loading model from {model_path}")
try:
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
ann_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
ann_index.load('plot_embeddings.ann')
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    plot = data.get('plot', '')

    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    inputs = tokenizer(plot, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
    nearest_indices = ann_index.get_nns_by_vector(cls_embedding[0], 5)
    paths = df.iloc[nearest_indices]['movie_poster_path'].tolist()
    genres = df.iloc[nearest_indices]['movie_category'].tolist()
    return jsonify({"predicted_class": predicted_class,
                    "similar_posters": paths,
                    "genres": genres})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    plots = data.get('plots', [])

    if not plots:
        return jsonify({"error": "No plots provided"}), 400

    inputs = tokenizer(plots, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1).tolist()

    return jsonify({"predicted_classes": predicted_classes})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy",
                    "model_loaded":True ,
                    "device": str(device)}), 200


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)

