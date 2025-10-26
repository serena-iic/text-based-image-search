# uses cosine similarity

import os, json, numpy as np, faiss
import re
from PIL import Image
from model_utils import get_image_embedding

image_dir = 'backend/data/images'
embedding_dir = 'backend/data/embeddings'
os.makedirs(embedding_dir, exist_ok = True)

ALLOWED_EXTENSIONS = (".jpg", ".png", '.jfif')

embeddings= [];
filenames = [];

def sanitize_filename(filename: str) -> str:
    # Replace any non-alphanumeric, dot, dash, or underscore characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]+', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized

print("processing images")
for filename in os.listdir(image_dir):

    # Sanitize filename
    clean_filename = sanitize_filename(filename)

    # If filename changed, rename the file
    if clean_filename != filename:
        old_path = os.path.join(image_dir, filename)
        new_path = os.path.join(image_dir, clean_filename)
        os.rename(old_path, new_path)
        filename = clean_filename  # use sanitized name

    if filename.lower().endswith(ALLOWED_EXTENSIONS):
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

print("Normalizing embeddings for cosine similarity...")
faiss.normalize_L2(embeddings)

# ✅ Use IndexFlatIP (Inner Product) instead of IndexFlatL2
# After normalization, Inner Product = Cosine Similarity
print("Building FAISS index with cosine similarity...")
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = Inner Product
index.add(embeddings)

# Save index and mapping
faiss.write_index(index, f"{embedding_dir}/image.index")
np.save(f"{embedding_dir}/embeddings.npy", embeddings)
with open(f"{embedding_dir}/mapping.json", "w") as f:
    json.dump(filenames, f)

print(f"✅ Successfully built index with {len(filenames)} images using cosine similarity!")