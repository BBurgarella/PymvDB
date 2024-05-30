# database.py
import sqlite3
import numpy as np
import base64
from PIL import Image
import io

class PymvDB:
    """
    A class to manage a vector database for image embeddings.

    Attributes:
        embedding_model (HuggingFaceEmbedding): The embedding model to use.
        conn (sqlite3.Connection): The SQLite database connection.
    """
    def __init__(self, embedding_model, db_path=None):
        """
        Initializes the VectorDatabase with an embedding model and database path.

        Args:
            embedding_model (HuggingFaceEmbedding): The embedding model to use.
            db_path (str): The path to the SQLite database file. Uses an in-memory database if None.
        """
        self.embedding_model = embedding_model
        if db_path is None:
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """
        Creates the vectors table in the database if it does not exist.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY,
            image_base64 TEXT NOT NULL,
            vector BLOB NOT NULL
        )
        ''')
        self.conn.commit()

    def insert_image(self, image_path: str):
        """
        Inserts an image and its embedding into the database.

        Args:
            image_path (str): The path to the image file.
        """
        image = Image.open(image_path)
        image_base64 = self.image_file_to_base64(image_path)
        vector = self.embedding_model.get_embedding(image)
        self.insert_image_vector(image_base64, vector)

    def insert_image_vector(self, image_base64: str, vector: np.ndarray):
        """
        Inserts an image's base64 encoding and embedding vector into the database.

        Args:
            image_base64 (str): The base64 encoding of the image.
            vector (numpy.ndarray): The embedding vector.
        """
        cursor = self.conn.cursor()
        vector_blob = np.array(vector).tobytes()
        cursor.execute('''
        INSERT INTO vectors (image_base64, vector) VALUES (?, ?)
        ''', (image_base64, vector_blob))
        self.conn.commit()

    def get_all_vectors(self):
        """
        Retrieves all image vectors from the database.

        Returns:
            tuple: A tuple containing a list of image base64 strings and a list of vectors.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT image_base64, vector FROM vectors
        ''')
        rows = cursor.fetchall()
        images = [row[0] for row in rows]
        vectors = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        return images, vectors

    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vector1 (numpy.ndarray): The first vector.
            vector2 (numpy.ndarray): The second vector.

        Returns:
            float: The cosine similarity between the vectors.
        """
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def image_file_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to its base64 encoding.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def base64_to_image(self, base64_string: str) -> Image.Image:
        """
        Converts a base64 encoded string to a PIL Image.

        Args:
            base64_string (str): The base64 encoded string of the image.

        Returns:
            PIL.Image.Image: The decoded image.
        """
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def find_similar_images(self, target_vector: np.ndarray, top_N=5, threshold=0.0):
        """
        Finds the most similar images to a target vector.

        Args:
            target_vector (numpy.ndarray): The target embedding vector.
            top_N (int): The number of similar images to return.
            threshold (float): The similarity threshold.

        Returns:
            list: A list of tuples containing the image base64 string and similarity score.
        """
        images, vectors = self.get_all_vectors()
        similarities = [self.calculate_cosine_similarity(target_vector, vec) for vec in vectors]
        similar_images = [(images[i], similarities[i]) for i in range(len(images)) if similarities[i] >= threshold]
        similar_images.sort(key=lambda x: x[1], reverse=True)
        return similar_images[:top_N]
