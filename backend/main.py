from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from backend.model_utils import get_text_embedding
import faiss, numpy as np, json, os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load index and mappings
embedding_dir = "backend/data/embeddings"
image_dir = "backend/data/images"

index = faiss.read_index(f"{embedding_dir}/image.index")
with open(f"{embedding_dir}/mapping.json") as f:
    mapping = json.load(f)

@app.get("/search")
def search_images(query: str = Query(...)):
    # Get text embeddings
    text_emb = get_text_embedding(query).astype("float32")
    
    # ✅ CRITICAL: Normalize text embedding for cosine similarity
    faiss.normalize_L2(text_emb)
    
    # Search FAISS index (get more candidates for filtering)
    top_k = 50
    D, I = index.search(text_emb, top_k)
    
    # Debug: print actual distances (now they're cosine similarities!)
    print(f"Query: {query}")
    print(f"Similarities (higher=better): {D[0][:10]}")  # Now these are similarity scores
    print(f"Indices: {I[0][:10]}")
    
    results = []
    for similarity, idx in zip(D[0], I[0]):
        if idx != -1:  # Skip invalid indices
            results.append({
                "filename": mapping[idx],
                "similarity": float(similarity),  # Higher = better match
                "distance": float(1 - similarity)  # Convert to distance if needed
            })
    
    # ✅ Filter by quality: keep results within reasonable similarity range
    if results:
        best_similarity = results[0]['similarity']
        # Keep results with similarity >= 70% of best match
        threshold = best_similarity * 0.7
        results = [r for r in results if r['similarity'] >= threshold]
    
    # Return top 30 results
    results = results[:30]
    
    return {
        "query": query,
        "results": [f"/image/{r['filename']}" for r in results],
        "similarities": [r['similarity'] for r in results],  # Higher is better!
        "count": len(results)
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    path = os.path.join(image_dir, filename)
    return FileResponse(path)