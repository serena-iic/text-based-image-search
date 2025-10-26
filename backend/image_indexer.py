# L2 Distance
import os, json, numpy as np, faiss
from PIL import Image
from model_utils import get_image_embedding

image_dir = 'backend/data/images'
embedding_dir = 'backend/data/embeddings'
os.makedirs(embedding_dir, exist_ok=True)

ALLOWED_EXTENSIONS = (".jpg", ".png",".jpeg", ".webp", ".jfif")

embeddings = []
filenames = []

for filename in os.listdir(image_dir):
    if(filename.lower().endswith(ALLOWED_EXTENSIONS)):
        file_path = os.path.join(image_dir, filename)
        
        if filename.lower().endswith('.jfif'):
            img = Image.open(file_path).convert("RGB")
            new_filename = filename.rsplit(".", 1)[0] + ".jpg"
            new_path = os.path.join(image_dir, new_filename)
            img.save(new_path, "JPEG")
            print(f"Converted {filename} → {new_filename}")
            os.remove(file_path)
            filename = new_filename
            file_path = new_path
        else:
            img = Image.open(file_path).convert("RGB")

        emb = get_image_embedding(img)
        embeddings.append(emb)
        filenames.append(filename)

embeddings = np.vstack(embeddings).astype("float32")

# build faiss index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# save index and mapping
faiss.write_index(index, f"{embedding_dir}/image.index")
np.save(f"{embedding_dir}/embeddings.npy", embeddings)
with open(f"{embedding_dir}/mapping.json", "w") as f:
    json.dump(filenames, f)