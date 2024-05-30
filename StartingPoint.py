from PymvDB import PymvDB, HuggingFaceEmbedding
from PIL import Image
import time

Emb = HuggingFaceEmbedding()
Db = PymvDB(Emb)

Im1 = Image.open("image.webp")

print("Starting the embedding operation")
t0 = time.time()
Im1 = Emb.get_embedding(Im1)
t1 = time.time()
print("Done with the embedding operation")
print(f"Sample of the vector: {Im1[0][:10]}, it took {t1-t0} seconds")