import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def extract_and_save_embeddings_from_folder(folder_path, model_name, video_name=None):
    """
    Extract embeddings from all images in a folder and save them to a video-specific file.
    
    Args:
        folder_path: Path to the folder containing images
        model_name: Name of the CLIP model to use
        video_name: Name of the video being processed (for naming the output file)
    
    Returns:
        The path to the saved embeddings file
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    all_embeddings = []
    image_paths = []
    
    # Create embeddings directory if it doesn't exist
    embeddings_dir = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\embedding"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create a video-specific output file name
    if not video_name:
        video_name = Path(folder_path).name
        
    output_file = os.path.join(embeddings_dir, f"{video_name}_embeddings.npy")
    # Đảm bảo đường dẫn nhất quán với dấu gạch chéo ngược
    output_file = output_file.replace('/', '\\')
    
    print(f"Extracting embeddings for {video_name}...")
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

    # Convert to array and save
    all_embeddings = np.array(all_embeddings)
    np.save(output_file, all_embeddings)
    print(f"Embeddings saved to {output_file}")
    
    return output_file



# folder_path = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"  
# model_name = "ViT-B/32"  
# output_file = "E:\\Đồ án chuyên ngành\\DACN_modelCLIP-3\\embedding\\image_embeddings.npy"  

# extract_and_save_embeddings_from_folder(folder_path, model_name, output_file)
