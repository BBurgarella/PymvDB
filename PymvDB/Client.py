import sqlite3
from PIL import Image
from .Collection import Collection

class Client:
    """
    A class to represent a client that manages collections of images and their embeddings.

    Attributes
    ----------
    embedding_model : callable
        The model used to generate embeddings from images.
    conn : sqlite3.Connection
        The SQLite connection object.

    Methods
    -------
    create_collection(Name)
        Creates a new collection.
    reset_collection(collection)
        Resets the specified collection.
    reset()
        Resets all collections managed by the client.
    """
    def __init__(self, embedding_model, persistent_path=None):
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
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = sqlite3.connect(persistent_path)

    def create_collection(self, Name: str):
        """
        Creates a new collection.

        Parameters
        ----------
        Name : str
            The name of the new collection.

        Returns
        -------
        Collection
            A new Collection object.
        """
        return Collection(Name, self.conn, self.embedding_model)

    def reset_collection(self, collection: Collection):
        """
        Resets the specified collection.

        Parameters
        ----------
        collection : Collection
            The collection to reset.
        """
        cursor = self.conn.cursor()
        cursor.execute(f'''
        DROP TABLE IF EXISTS {collection.name}
        ''')
        collection._create_table()

    def reset(self):
        """
        Resets all collections managed by the client.
        """
        cursor = self.conn.cursor()
        # Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
