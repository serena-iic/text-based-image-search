# from transformers import CLIPProcessor, CLIPModel
# import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def get_image_embedding(image):
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         emb = model.get_image_features(**inputs)
#     return emb.cpu().numpy()

# def get_text_embedding(text):
#     inputs = processor(text=[text], return_tensors='pt', padding=True).to(device)
#     with torch.no_grad():
#         emb = model.get_text_features(**inputs)
#     return emb.cpu().numpy()

# trying cosine similarity -------------------------------------------------------------------------------

from transformers import CLIPProcessor, CLIPModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy()

def get_text_embedding(text):
    enhanced_text = f"a photo of {text}"
    
    inputs = processor(text=[enhanced_text], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb.cpu().numpy()