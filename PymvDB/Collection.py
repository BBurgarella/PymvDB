import sqlite3
import numpy as np
import base64
import requests
import json
from .query_result import qresult
from PIL import Image
import io

class HTTPCollection:
    """
    A class to represent a remote collection of images and their embeddings via HTTP.

    Attributes
    ----------
    name : str
        The name of the collection.
    server_url : str
        The URL of the server.

    Methods
    -------
    add_image(image_path, metadata=None)
        Adds an image to the collection.
    find_similar_images(image_path, top_N=5, threshold=0.0, where=None)
        Finds similar images in the collection.
    image_file_to_base64(image_path)
        Converts an image file to a base64 string.
    base64_to_image(base64_string)
        Converts a base64 string to an image.
    """

    def __init__(self, Name, server_url):
        """
        Parameters
        ----------
        name : str
            The name of the collection.
        server_url : str
            The URL of the server.
        """
        self.name = Name
        self.server_url = server_url

    def add_image(self, image_path: str, metadata: dict = None):
        """
        Adds an image to the collection.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        metadata : dict, optional
            The metadata associated with the image (default is None).
        """
        if metadata is None:
            metadata = {}

        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        url = f"{self.server_url}/add_image"
        data = {
            "collection": self.name,
            "file": image_path,
            "image_base64": image_base64,
            "metadata": metadata
        }
        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to add image: {response.json()}")

    def find_similar_images(self, image: Image.Image, top_N=5, threshold=0.0, where=None):
        """
        Finds similar images in the collection.

        Parameters
        ----------
        image : PIL.Image.Image
            The PIL Image object of the target image.
        top_N : int, optional
            The number of top similar images to return (default is 5).
        threshold : float, optional
            The similarity threshold for filtering results (default is 0.0).
        where : dict, optional
            The metadata conditions for filtering results (default is None).

        Returns
        -------
        qresult
            A query result object containing the similar images and their details.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        url = f"{self.server_url}/find_similar"
        data = {
            "collection": self.name,
            "image_base64": image_base64,
            "top_N": top_N,
            "threshold": threshold,
            "where": where
        }
        response = requests.post(url, json=data)
        result = qresult(**response.json())
        if response.status_code != 200:
            raise Exception(f"Failed to find similar images: {response.json()}")

        return result

class Collection:
    def __init__(self, name, conn_str, embedding_model):
        """
        Parameters
        ----------
        name : str
            The name of the collection.
        conn_str : str
            The SQLite connection string.
        embedding_model : callable
            The model used to generate embeddings from images.
        """
        self.name = name
        self.conn_str = conn_str
        self.embedding_model = embedding_model
        self._create_table()

    def _get_connection(self):
        """Returns a new SQLite connection."""
        return sqlite3.connect(self.conn_str)

    def _create_table(self):
        """Creates the SQLite table for the collection if it does not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.name} (
                id INTEGER PRIMARY KEY,
                image_base64 TEXT NOT NULL,
                image_file_name TEXT NOT NULL UNIQUE,
                vector BLOB NOT NULL,
                metadata TEXT
            )
            ''')
            conn.commit()

    def add_image(self, image_path: str, metadata: dict = None):
        """
        Adds an image to the collection.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        metadata : dict, optional
            The metadata associated with the image (default is None).
        """
        if metadata is None:
            metadata = {}
        image = Image.open(image_path)
        image_base64 = self.image_file_to_base64(image_path)
        vector = self.embedding_model(image)
        self.add_image_vector(image_path, image_base64, vector, metadata)

    def add_image_vector(self, path: str, image_base64: str, vector: np.ndarray, metadata: dict):
        """
        Adds an image vector to the collection.

        Parameters
        ----------
        path : str
            The file path of the image.
        image_base64 : str
            The base64 encoded string of the image.
        vector : np.ndarray
            The embedding vector of the image.
        metadata : dict
            The metadata associated with the image.
        """
        vector_blob = np.array(vector).tobytes()
        metadata_json = json.dumps(metadata)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f'''
                INSERT INTO {self.name} (image_base64, image_file_name, vector, metadata) VALUES (?, ?, ?, ?)
                ''', (image_base64, path, vector_blob, metadata_json))
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    def get_all_vectors(self):
        """
        Retrieves all vectors from the collection.

        Returns
        -------
        files : list of str
            The list of base64 encoded images.
        paths : list of str
            The list of image file paths.
        vectors : list of np.ndarray
            The list of embedding vectors.
        metadata : list of dict
            The list of metadata dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
            SELECT image_base64, image_file_name, vector, metadata FROM {self.name}
            ''')
            rows = cursor.fetchall()
            files = [row[0] for row in rows]
            paths = [row[1] for row in rows]
            vectors = [np.frombuffer(row[2], dtype=np.float32) for row in rows]
            metadata = [json.loads(row[3]) for row in rows]
            return files, paths, vectors, metadata

    def find_similar_images(self, target_image, top_N=5, threshold=0.0, where=None):
        """
        Finds similar images in the collection.

        Parameters
        ----------
        target_image : Image.Image
            The target image to find similarities for.
        top_N : int, optional
            The number of top similar images to return (default is 5).
        threshold : float, optional
            The similarity threshold for filtering results (default is 0.0).
        where : dict, optional
            The metadata conditions for filtering results (default is None).

        Returns
        -------
        qresult
            A query result object containing the similar images and their details.
        """
        target_vector = self.embedding_model(target_image)
        base64s, paths, vectors, metadata = self.get_all_vectors()
        similarities = [self.calculate_cosine_similarity(target_vector, vec) for vec in vectors]
        
        similar_images = [
            (paths[i], base64s[i], similarities[i], metadata[i])
            for i in range(len(vectors))
            if similarities[i] >= threshold and self._metadata_matches(metadata[i], where)
        ]
        similar_images.sort(key=lambda x: x[2], reverse=True)
        
        top_similar_images = similar_images[:top_N]
        
        result = {
            "n_findings": len([sim for sim in similarities if sim >= threshold]),
            "scores": [float(img[2]) for img in top_similar_images],
            "files": [img[0] for img in top_similar_images],
            "base64": [img[1] for img in top_similar_images],
            "metadata": [img[3] for img in top_similar_images]
        }
        
        return qresult(**result)

    def _metadata_matches(self, metadata, where):
        """
        Checks if the metadata matches the given conditions.

        Parameters
        ----------
        metadata : dict
            The metadata dictionary to check.
        where : dict
            The conditions to match against.

        Returns
        -------
        bool
            True if the metadata matches the conditions, False otherwise.
        """
        if not where:
            return True
        for key, value in where.items():
            if metadata.get(key) != value:
                return False
        return True

    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Parameters
        ----------
        vector1 : np.ndarray
            The first vector.
        vector2 : np.ndarray
            The second vector.

        Returns
        -------
        float
            The cosine similarity between the two vectors.
        """
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def image_file_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 string.

        Parameters
        ----------
        image_path : str
            The path to the image file.

        Returns
        -------
        str
            The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def base64_to_image(self, base64_string: str) -> Image.Image:
        """
        Converts a base64 string to an image.

        Parameters
        ----------
        base64_string : str
            The base64 encoded string of the image.

        Returns
        -------
        Image.Image
            The decoded image.
        """
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))