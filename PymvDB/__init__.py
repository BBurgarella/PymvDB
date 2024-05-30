"""
This module initializes the package and imports the necessary components.

The following components are imported:
- HuggingFaceEmbedding: Handles embedding operations using HuggingFace's transformers.
- PymvDB: Manages database interactions.

The __all__ variable is defined to specify the public API of the package, 
including HuggingFaceEmbedding and PymvDB.
"""


# __init__.py
from .embeddings import HuggingFaceEmbedding
from .database import PymvDB

__all__ = ["HuggingFaceEmbedding", "PymvDB"]
__version__= "0.1.0"