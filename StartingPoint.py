import PymvDB
from PIL import Image
import time
import os
import sys

Emb = PymvDB.HuggingFaceEmbedding()
Db = PymvDB.Client(Emb,persistent_path="./db.db3")
Db.reset()
collection = Db.create_collection(Name="Example_Collection")


# Directory containing the images
image_directory = "example_images"

# Loop through all files in the directory
for filename in os.listdir(image_directory):
    # Construct the full file path
    file_path = os.path.join(image_directory, filename)
    
    # Insert the image into the database
    collection.add_image(file_path, metadata={"extension": file_path[-4:]})

if __name__ == '__main__':
    image_path = sys.argv[1]
    New_image = Image.open(image_path)


    t0 = time.time()
    Res = collection.find_similar_images(New_image, 
                                         top_N=2, 
                                         threshold=0.1
                                         # where={"metadata_field": "is_equal_to_this"}, # optional filter
                                         )
    t1 = time.time()

    print(Res)

    print(f"This image looks like a {Res.files[0].split("/")[1][:-4]}, with a score of {Res.scores[0]}")