import sqlite3
import numpy as np
import base64
import json
from .query_result import qresult
from PIL import Image
import io


class Collection:
    """
    A class to represent a collection of images and their embeddings.

    Attributes
    ----------
    name : str
        The name of the collection.
    conn : sqlite3.Connection
        The SQLite connection object.
    embedding_model : callable
        The model used to generate embeddings from images.

    Methods
    -------
    add_image(image_path, metadata=None)
        Adds an image to the collection.
    add_image_vector(path, image_base64, vector, metadata)
        Adds an image vector to the collection.
    get_all_vectors()
        Retrieves all vectors from the collection.
    find_similar_images(target_image, top_N=5, threshold=0.0, where=None)
        Finds similar images in the collection.
    calculate_cosine_similarity(vector1, vector2)
        Calculates the cosine similarity between two vectors.
    image_file_to_base64(image_path)
        Converts an image file to a base64 string.
    base64_to_image(base64_string)
        Converts a base64 string to an image.
    """
    def __init__(self, name, conn, embedding_model):
        """
        Parameters
        ----------
        name : str
            The name of the collection.
        conn : sqlite3.Connection
            The SQLite connection object.
        embedding_model : callable
            The model used to generate embeddings from images.
        """
        self.name = name
        self.conn = conn
        self.embedding_model = embedding_model
        self._create_table()

    def _create_table(self):
        """Creates the SQLite table for the collection if it does not exist."""
        cursor = self.conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            id INTEGER PRIMARY KEY,
            image_base64 TEXT NOT NULL,
            image_file_name TEXT NOT NULL UNIQUE,
            vector BLOB NOT NULL,
            metadata TEXT
        )
        ''')
        self.conn.commit()

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
        cursor = self.conn.cursor()
        vector_blob = np.array(vector).tobytes()
        metadata_json = json.dumps(metadata)
        try:
            cursor.execute(f'''
            INSERT INTO {self.name} (image_base64, image_file_name, vector, metadata) VALUES (?, ?, ?, ?)
            ''', (image_base64, path, vector_blob, metadata_json))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"An image with the file name '{path}' already exists in the collection '{self.name}'.")

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
        cursor = self.conn.cursor()
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
            "scores": [img[2] for img in top_similar_images],
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
