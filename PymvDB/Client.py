import sqlite3
from PIL import Image
from .Collection import Collection

class Client:
    def __init__(self, embedding_model, persistent_path=None):
        self.embedding_model = embedding_model
        if persistent_path is None:
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = sqlite3.connect(persistent_path)

    def create_collection(self, Name: str):
        return Collection(Name, self.conn, self.embedding_model)

    def reset_collection(self, collection: Collection):
        cursor = self.conn.cursor()
        cursor.execute(f'''
        DROP TABLE IF EXISTS {collection.name}
        ''')
        collection._create_table()

    def reset(self):
        cursor = self.conn.cursor()
        # Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
