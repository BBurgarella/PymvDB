import sqlite3
import numpy as np
import base64
from .query_result import qresult
from PIL import Image
import io


class Collection:
    def __init__(self, name, conn, embedding_model):
        self.name = name
        self.conn = conn
        self.embedding_model = embedding_model
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            id INTEGER PRIMARY KEY,
            image_base64 TEXT NOT NULL,
            image_file_name TEXT NOT NULL UNIQUE,
            vector BLOB NOT NULL
        )
        ''')
        self.conn.commit()

    def add_image(self, image_path: str):
        image = Image.open(image_path)
        image_base64 = self.image_file_to_base64(image_path)
        vector = self.embedding_model(image)
        self.add_image_vector(image_path, image_base64, vector)

    def add_image_vector(self, path: str, image_base64: str, vector: np.ndarray):
        cursor = self.conn.cursor()
        vector_blob = np.array(vector).tobytes()
        try:
            cursor.execute(f'''
            INSERT INTO {self.name} (image_base64, image_file_name, vector) VALUES (?, ?, ?)
            ''', (image_base64, path, vector_blob))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"An image with the file name '{path}' already exists in the collection '{self.name}'.")

    def get_all_vectors(self):
        cursor = self.conn.cursor()
        cursor.execute(f'''
        SELECT image_base64, image_file_name, vector FROM {self.name}
        ''')
        rows = cursor.fetchall()
        files = [row[0] for row in rows]
        paths = [row[1] for row in rows]
        vectors = [np.frombuffer(row[2], dtype=np.float32) for row in rows]
        return files, paths, vectors

    def find_similar_images(self, target_image, top_N=5, threshold=0.0):
        target_vector = self.embedding_model(target_image)
        base64s, paths, vectors = self.get_all_vectors()
        similarities = [self.calculate_cosine_similarity(target_vector, vec) for vec in vectors]
        
        similar_images = [(paths[i], base64s[i], similarities[i]) for i in range(len(vectors)) if similarities[i] >= threshold]
        similar_images.sort(key=lambda x: x[2], reverse=True)
        
        top_similar_images = similar_images[:top_N]
        
        result = {
            "n_findings": len([sim for sim in similarities if sim >= threshold]),
            "scores": [img[2] for img in top_similar_images],
            "files": [img[0] for img in top_similar_images],
            "base64": [img[1] for img in top_similar_images]
        }
        
        return qresult(**result)


    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def image_file_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def base64_to_image(self, base64_string: str) -> Image.Image:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

