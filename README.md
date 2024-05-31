# PymvDB
<p align="center">
<img src="https://github.com/BBurgarella/PymvDB/raw/main/Logo.webp" alt="logo" width="400"/>
</p>

## Description
PymvDB is a Python library designed to create and manage a vector database for images.
it comes with the ability to use Hugging Face image feature extraction models as encoders.

running it with [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) takes 0.41 seconds to 
encode an image on my machine (Ryzen 7 2700)

## Installation
You can install PymvDB using pip (coming soon):

```sh
pip install pymvdb
```

while waiting for me to publish it on pip, you can download and install PymvDB using:
```sh
git clone https://github.com/BBurgarella/PymvDB.git
cd PymvDB
pip install .
```

## Local Usage
```python
# Initialize the client with an embedding model
embedding_model = YourEmbeddingModel()  # Replace with your actual embedding model
db = Client(embedding_model, persistent_path='database.sqlite')

# Create a new collection
collection = db.create_collection(Name='my_collection')

# Add an image to the collection
collection.add_image('path/to/image', metadata={"..."})

# Find similar images
target_image = Image.open('path/to/target_image')
similar_images = collection.find_similar_images(target_image, top_N=5)
print(similar_images)
```

You can run a quick example by using the StartingPoint file. All the images used here come from Wikipedia

```sh
python StartingPoint.py Test_car.jpg
```
