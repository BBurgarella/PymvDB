from PymvDB import PymvDB, HuggingFaceEmbedding
from PIL import Image
import time
import os
import sys

Emb = HuggingFaceEmbedding()
Db = PymvDB(Emb,db_path="./db.db3")

# Directory containing the images
image_directory = "example_images"

# Loop through all files in the directory
for filename in os.listdir(image_directory):
    # Construct the full file path
    file_path = os.path.join(image_directory, filename)
    
    # Insert the image into the database
    Db.insert_image(file_path)

if __name__ == '__main__':
    image_path = sys.argv[1]
    New_image = Image.open(image_path)


    t0 = time.time()
    Res = Db.find_similar_images(New_image)
    t1 = time.time()

    #Cat = Res[0][0]
    #Dog = Res[1][0]

    print(f"This image looks like a {Res[0][0].split("/")[1][:-4]}, with a score of {Res[0][2]}")
    print(f"The second best is a {Res[1][0].split("/")[1][:-4]}, with a score of {Res[1][2]}")
