import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_paths = ["4166.jpg", "10343.jpg"]  
texts = ["there is a umbrella in the left of picture", "people are standing on stage"]


images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]

text_inputs = clip.tokenize(texts).to(device)

image_batch = torch.cat(images, dim=0)

with torch.no_grad():
    image_embeddings = model.encode_image(image_batch)
    text_embeddings = model.encode_text(text_inputs)


image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

similarities = torch.matmul(image_embeddings, text_embeddings.T)

similarities_np = similarities.cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(similarities_np, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Cosine Similarity")
plt.xticks(ticks=np.arange(len(texts)), labels=texts, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(image_paths)), labels=[f"Image {i+1}" for i in range(len(image_paths))])
plt.title("Cosine Similarity between Images and Texts")
plt.xlabel("Text Descriptions")
plt.ylabel("Images")
plt.tight_layout()
plt.show()
