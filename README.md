# PymvDB

## Description
PymvDB is a Python library designed to create and manage a vector database for images.
it comes with the ability to use Hugging Face image feature extraction models as encoders.

running it with google/vit-base-patch16-224-in21k[https://huggingface.co/google/vit-base-patch16-224-in21k] takes 0.41 seconds to 
encode an image on my machine (Ryzen 7 2700)

## Installation
You can install PymvDB using pip (comming soon):

```sh
pip install pymvdb
```

## Usage
Here are some basic examples of how to use PymvDB:

### Database Operations

```python
import PymvDB.database

# Initialize the database
db = PymvDB.database.Database()

# Example database operations
db.create_table('example_table')
db.insert('example_table', {'column1': 'value1', 'column2': 'value2'})
data = db.query('SELECT * FROM example_table')
print(data)
```

### Embedding Operations

```python
from PymvDB.embeddings import HuggingFaceEmbedding
from PIL import Image
import numpy as np

# Initialize the embedding model
embedding_model = HuggingFaceEmbedding(model_name='google/vit-base-patch16-224-in21k')

# Load an image
image = Image.open('path_to_image.jpg')

# Generate an embedding for the image
embedding = embedding_model.get_embedding(image)
print(embedding)

# Calculate similarity between two embeddings
embedding1 = embedding_model.get_embedding(image1)
embedding2 = embedding_model.get_embedding(image2)
similarity_score = HuggingFaceEmbedding.similarity(embedding1, embedding2)
print(f'Similarity score: {similarity_score}')
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For more information, please contact the project maintainers.
