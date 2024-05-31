from flask import Flask, request, jsonify, g
from .Client import Client
from .query_result import qresult
from .Collection import Collection
from .embeddings import HuggingFaceEmbedding
from PIL import Image
import io
import base64
import numpy as np

HTTPserver = Flask(__name__)

embedding_model = HuggingFaceEmbedding()

def get_db():
    if 'db' not in g:
        g.db = Client(embedding_model, persistent_path='db.db3').conn
    return g.db

@HTTPserver.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def image_from_base64(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

@HTTPserver.route('/create_collection', methods=['POST'])
def create_collection():
    data = request.json
    collection_name = data['name']
    return jsonify({"message": f"Collection '{collection_name}' created."})

@HTTPserver.route('/add_image', methods=['POST'])
def add_image():
    data = request.json
    collection_name = data['collection']
    file_name = data['file']
    image_base64 = data['image_base64']
    metadata = data.get('metadata', {})

    image = image_from_base64(image_base64)
    collection = Collection(collection_name, get_db(), embedding_model)
    collection.add_image_vector(file_name, image_base64, embedding_model(image), metadata)
    
    return jsonify({"message": "Image added to collection."})

@HTTPserver.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.json
    collection_name = data['collection']
    image_base64 = data['image_base64']
    top_N = data.get('top_N', 5)
    threshold = data.get('threshold', 0.0)
    where = data.get('where', None)

    image = image_from_base64(image_base64)
    collection = Collection(collection_name, get_db(), embedding_model)
    result = collection.find_similar_images(image, top_N, threshold, where)

    return jsonify({
        "n_findings": result.n,
        "scores": result.scores,
        "files": result.files,
        "base64": result.base64,
        "metadata": result.metadata
    })

if __name__ == '__main__':
    HTTPserver.run(debug=True)
