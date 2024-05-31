import PymvDB
from PIL import Image
import time
import os
import sys
import argparse

def init_local():
    Emb = PymvDB.HuggingFaceEmbedding()
    Db = PymvDB.Client(Emb, persistent_path="./db.db3")
    Db.reset()
    collection = Db.create_collection(name="Example_Collection")

    # Directory containing the images
    image_directory = "example_images"

    # Loop through all files in the directory
    for filename in os.listdir(image_directory):
        # Construct the full file path
        file_path = os.path.join(image_directory, filename)
        
        # Insert the image into the database
        collection.add_image(file_path, metadata={"extension": file_path[-4:]})

    return Db, collection

def init_http(ip, port=5000):
    address = f"http://{ip}:{port}"
    Db = PymvDB.HTTPclient(address)
    collection = Db.create_collection(Name="Example_Collection")

    # Directory containing the images
    image_directory = "example_images"

    # Loop through all files in the directory
    for filename in os.listdir(image_directory):
        # Construct the full file path
        file_path = os.path.join(image_directory, filename)
        
        # Insert the image into the database
        collection.add_image(file_path, metadata={"extension": file_path[-4:]})


    return Db, collection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('-S', '--Server', help='Start in server mode', action='store_true')
    parser.add_argument('-I', '--Image', help='Path to the image file')
    parser.add_argument('-ip', required=False, help='IP address for remote HTTP server')
    parser.add_argument('-p', '--port', required=False, type=int, help='Port for remote HTTP server')

    args = parser.parse_args()

    if args.Server:
        # Server mode
        server = PymvDB.HTTPserver
        server.run()
    else:
        # Client mode
        if not args.Image:
            parser.error("Client mode requires --Image to be specified.")
        image_path = args.Image
        New_image = Image.open(image_path)

        if args.ip:
            if args.port:
                Db, collection = init_http(args.ip, args.port)
            else:
                Db, collection = init_http(args.ip)
        else:
            Db, collection = init_local()

        t0 = time.time()
        Res = collection.find_similar_images(New_image, 
                                            top_N=2, 
                                            threshold=0.1
                                            # where={"metadata_field": "is_equal_to_this"}, # optional filter
                                            )
        t1 = time.time()

        print(Res)
        print(f"This image looks like a {Res.files[0].split('/')[1][:-4]}, with a score of {Res.scores[0]}")