import sqlite3
from PIL import Image
from .Collection import Collection, HTTPCollection
import requests
from typing import Optional

class HTTPclient:
    def __init__(self, server_url):
        self.server_url = server_url

    def create_collection(self, Name):
        url = f"{self.server_url}/create_collection"
        response = requests.post(url, json={"name": Name})
        
        if response.status_code == 200 and response.json().get("message") == f"Collection '{Name}' created.":
            return HTTPCollection(Name, self.server_url)
        else:
            raise Exception(f"Failed to create collection: {response.json()}")

   

class Client:
    """
    A class to represent a client that manages collections of images and their embeddings.

    Attributes
    ----------
    embedding_model : callable
        The model used to generate embeddings from images.
    conn_str : str
        The SQLite connection string (path to the SQLite database file).

    Methods
    -------
    create_collection(name)
        Creates a new collection.
    reset_collection(collection)
        Resets the specified collection.
    reset()
        Resets all collections managed by the client.
    """
    def __init__(self, embedding_model: callable, persistent_path: Optional[str] = None):
        """
        Parameters
        ----------
        embedding_model : callable
            The model used to generate embeddings from images.
        persistent_path : str, optional
            The path to the SQLite database file. If None, an in-memory database is used (default is None).
        """
        self.embedding_model = embedding_model
        if persistent_path is None:
            self.conn_str = ':memory:'
        else:
            self.conn_str = persistent_path

    def _get_connection(self):
        """Returns a new SQLite connection."""
        return sqlite3.connect(self.conn_str)

    def create_collection(self, name: str):
        """
        Creates a new collection.

        Parameters
        ----------
        name : str
            The name of the new collection.

        Returns
        -------
        Collection
            A new Collection object.
        """
        return Collection(name, self.conn_str, self.embedding_model)

    def reset_collection(self, collection: Collection):
        """
        Resets the specified collection.

        Parameters
        ----------
        collection : Collection
            The collection to reset.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
            DROP TABLE IF EXISTS {collection.name}
            ''')
            conn.commit()
        collection._create_table()

    def reset(self):
        """
        Resets all collections managed by the client.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            # Drop each table
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
            conn.commit()