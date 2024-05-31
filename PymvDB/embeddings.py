# embedding.py
import numpy as np
from numpy.linalg import norm
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class HuggingFaceEmbedding:
    """
    A class to generate image embeddings using a Hugging Face model.

    Attributes
    ----------
    processor : AutoImageProcessor
        Processor for image preprocessing.
    model : AutoModel
        Model to generate image embeddings.

    Methods
    -------
    __call__(image)
        Generates an embedding for the given image.
    similarity(vector1, vector2)
        Calculates the cosine similarity between two vectors.
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        Initializes the HuggingFaceEmbedding with the specified model.

        Parameters
        ----------
        model_name : str, optional
            The name of the Hugging Face model to use (default is 'google/vit-base-patch16-224-in21k').
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image) -> np.ndarray:
        """
        Generates an embedding for the given image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to generate an embedding for.

        Returns
        -------
        numpy.ndarray
            The embedding vector.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states[:, 0].detach().numpy()
        return features

    @staticmethod
    def similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Parameters
        ----------
        vector1 : numpy.ndarray
            The first vector.
        vector2 : numpy.ndarray
            The second vector.

        Returns
        -------
        float
            The cosine similarity between the vectors.
        """
        return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
