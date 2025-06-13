import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_and_save_embeddings_from_folder(folder_path, model_name, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    all_embeddings = []
    image_paths = []
    print("Extracting embeddings...")
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

                image = Image.open(image_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model.encode_image(image_input)
                    embedding = embedding.cpu().numpy().flatten()

                all_embeddings.append(embedding)

    all_embeddings = np.array(all_embeddings)
    if os.path.exists(output_file):
        existing_embeddings = np.load(output_file)
        all_embeddings = np.vstack([existing_embeddings, all_embeddings])
    np.save(output_file, all_embeddings)
    print(f"Embeddings saved to {output_file}")



# folder_path = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"  
# model_name = "ViT-B/32"  
# output_file = "E:\\Đồ án chuyên ngành\\DACN_modelCLIP-3\\embedding\\image_embeddings.npy"  

# extract_and_save_embeddings_from_folder(folder_path, model_name, output_file)
