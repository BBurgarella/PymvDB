"""
This module initializes the package and imports the necessary components.

The following components are imported:
- HuggingFaceEmbedding: Handles embedding operations using HuggingFace's transformers.
- PymvDB: Manages database interactions.
- Client: Manages the vector database and collection creation.
- Collection: Handles operations within individual collections.

The __all__ variable is defined to specify the public API of the package, 
including HuggingFaceEmbedding, PymvDB, Client, and Collection.
"""

from .embeddings import HuggingFaceEmbedding
from .Client import Client
from .Collection import Collection

__all__ = ["HuggingFaceEmbedding", "Client", "Collection"]
__version__ = "0.1.0"
