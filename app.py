from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote

app = Flask(__name__)

# Load the model and hyperparameters from the pickle file
model_file_path = 'model.pkl'
with open(model_file_path, 'rb') as f:
    model_dict = pickle.load(f)

transform = model_dict['transform']
sbert_model = model_dict['sbert_model']
model = model_dict['model']
roberta_model = model_dict['roberta_model']
cv_embeddings = model_dict['cv_embeddings']
nlp_embeddings = model_dict['nlp_embeddings']


def search_engine(image_query_path, text_query, image_embeddings, text_embeddings, all_image_paths, top_k=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('convnext_base', pretrained=True).to(device)
    model.eval()

    image = Image.open(image_query_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model(image_tensor).squeeze(0).cpu().numpy()
    text_embedding = sbert_model.encode([text_query])[0]
    image_similarities = cosine_similarity(image_embedding.reshape(1, -1), image_embeddings).flatten()
    text_similarities = cosine_similarity(text_embedding.reshape(1, -1), text_embeddings).flatten()
    combined_similarities = 0.5 * image_similarities + 0.5 * text_similarities
    ranked_indices = np.argsort(-combined_similarities)[:top_k]
    ranked_results = [(all_image_paths[idx], combined_similarities[idx]) for idx in ranked_indices]

    return ranked_results

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/searchengine', methods=['POST'])
def search_route():
        image_query_path = request.files['image_query_path']
        text_query = request.form['text']
        top_k = int(request.form['numImages'])

        data_dir = "static/descriptions"
        all_image_paths = []
        descriptions = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(data_dir, filename)
                df = pd.read_excel(file_path)
                image_paths = ["static/" + path for path in df.iloc[:, 0].tolist()]
                desc = df.iloc[:, 1].tolist()
                descriptions.extend(desc)
                all_image_paths.extend(image_paths)

        ranked_results = search_engine(image_query_path, text_query, cv_embeddings, nlp_embeddings, all_image_paths, top_k)
        encoded_search_results = [(quote(result[0]), result[1]) for result in ranked_results]

        return render_template('display.html', ranked_results=encoded_search_results)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
