import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import clip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
from tqdm import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Sử dụng lại khai báo model từ training_CLIP.py với các cải tiến
class CLIPFineTuner(nn.Module):
    def __init__(self, num_classes=2, pretrained="ViT-B/32", freeze_clip=True):
        super(CLIPFineTuner, self).__init__()
        self.clip_model, self.preprocess = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.clip_model.visual.output_dim
        self.device = next(self.clip_model.parameters()).device

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Thêm projection heads để cải thiện embedding space
        self.image_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Đơn giản hóa fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )
        
        # Classifier chung cho đặc trưng đã kết hợp
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Thêm classifier riêng cho image và text features
        self.image_classifier = nn.Linear(embedding_dim, num_classes)
        self.text_classifier = nn.Linear(embedding_dim, num_classes)
        
        # Hệ số cho các thành phần loss (điều chỉnh theo cấu hình mới)
        self.alpha = 0.3  # Giảm trọng số cho fusion loss
        self.beta = 0.1   # Trọng số cho image loss
        self.gamma = 0.1  # Trọng số cho text loss

    def forward(self, image_batch, caption_batch=None):
        """Forward pass với cả image và text."""
        # Đảm bảo tắt autocast để tránh lỗi dtype mismatch
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                # Encode image
                image_features = self.clip_model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                # Encode text nếu được cung cấp
                if caption_batch is not None:
                    text_tokens = clip.tokenize(caption_batch).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    
                    # Chuyển về float32
                    image_features = image_features.float()
                    text_features = text_features.float()
                    
                    # Sử dụng projection heads để cải thiện embedding space
                    image_features_proj = self.image_proj(image_features)
                    text_features_proj = self.text_proj(text_features)
                    
                    # Normalize lại sau projection
                    image_features_proj = image_features_proj / image_features_proj.norm(dim=1, keepdim=True)
                    text_features_proj = text_features_proj / text_features_proj.norm(dim=1, keepdim=True)
                    
                    # Tính cosine similarity giữa projected image và text features
                    similarity = torch.matmul(image_features_proj, text_features_proj.t())
                    
                    return {
                        'image_features': image_features,
                        'text_features': text_features,
                        'image_features_proj': image_features_proj,
                        'text_features_proj': text_features_proj,
                        'similarity': similarity
                    }
                
                # Nếu chỉ có image, vẫn áp dụng projection head
                image_features = image_features.float()
                image_features_proj = self.image_proj(image_features)
                image_features_proj = image_features_proj / image_features_proj.norm(dim=1, keepdim=True)
                
                return {
                    'image_features': image_features,
                    'image_features_proj': image_features_proj
                }

class OriginalCLIP:
    """Wrapper cho model CLIP thuần để có cùng interface với CLIPFineTuner."""
    def __init__(self, model_name="ViT-B/32"):
        self.clip_model, self.preprocess = clip.load(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.clip_model.parameters()).device
    
    def forward(self, image_batch, caption_batch=None):
        with torch.no_grad():
            # Encode image
            image_features = self.clip_model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Encode text nếu được cung cấp
            if caption_batch is not None:
                text_tokens = clip.tokenize(caption_batch).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Ép kiểu về float32
                image_features = image_features.float()
                text_features = text_features.float()
                
                # Tính cosine similarity
                similarity = torch.matmul(image_features, text_features.t())
                return {
                    'image_features': image_features,
                    'text_features': text_features,
                    'similarity': similarity
                }
            
            return {
                'image_features': image_features
            }
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None):
        self.frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.frame_paths.extend(glob.glob(os.path.join(frame_dir, ext)))
        self.frame_paths.sort()  # Sắp xếp để đảm bảo thứ tự ổn định
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Lỗi đọc ảnh {frame_path}: {str(e)}")
            image = torch.randn(3, 224, 224)  # fallback
            
        return {
            'image': image,
            'path': frame_path
        }

def load_model(model_path, device):
    """Tải model CLIP fine-tuned"""
    model = CLIPFineTuner(pretrained="ViT-B/32", freeze_clip=True).to(device)
    
    # Tải weights đã lưu
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:  # Nếu load từ checkpoint
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
    else:  # Nếu load từ best_model
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model

def load_original_clip(device, model_name="ViT-B/32"):
    """Tải model CLIP thuần"""
    model = OriginalCLIP(model_name=model_name)
    return model

def encode_frames(model, frame_dir, batch_size=16, device='cuda'):
    """Mã hóa tất cả các frame trong thư mục frame_dir"""
    # Khởi tạo transform giống như khi validate
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Tạo dataset và dataloader
    dataset = FrameDataset(frame_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    frame_features = []
    frame_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đang mã hóa frames"):
            images = batch['image'].to(device)
            outputs = model(images)
            
            # Lưu projected features nếu có, hoặc dùng features gốc
            if 'image_features_proj' in outputs:
                features = outputs['image_features_proj'].cpu().numpy()
            else:
                features = outputs['image_features'].cpu().numpy() 
                
            frame_features.append(features)
            frame_paths.extend(batch['path'])
    
    # Ghép tất cả features lại
    frame_features = np.vstack(frame_features)
    
    print(f"Đã mã hóa {len(frame_paths)} frames")
    return frame_features, frame_paths

def search_frames(model, text_query, frame_features, frame_paths, top_k=9, device='cuda'):
    """Tìm kiếm frames dựa trên text query"""
    with torch.no_grad():
        # Mã hóa text query
        text_tokens = clip.tokenize([text_query]).to(device)
        text_features = model.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.float()
        
        # Sử dụng projection head nếu có
        if hasattr(model, 'text_proj'):
            text_features = model.text_proj(text_features)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
        text_features = text_features.cpu().numpy()
    
    # Tính cosine similarity
    similarities = np.dot(frame_features, text_features.T).squeeze()
    
    # Lấy top-k frames có similarity cao nhất
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'path': frame_paths[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

def search_frames_original_clip(model, text_query, frame_features, frame_paths, top_k=9, device='cuda'):
    """Tìm kiếm frames dựa trên text query với model CLIP thuần"""
    with torch.no_grad():
        # Mã hóa text query
        text_tokens = clip.tokenize([text_query]).to(device)
        text_features = model.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.float().cpu().numpy()
    
    # Tính cosine similarity
    similarities = np.dot(frame_features, text_features.T).squeeze()
    
    # Lấy top-k frames có similarity cao nhất
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'path': frame_paths[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

def display_results(results, text_query, grid_size=(3, 3), figsize=(15, 15), title=None):
    """Hiển thị kết quả tìm kiếm dưới dạng lưới ảnh"""
    plt.figure(figsize=figsize)
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f'Kết quả truy vấn cho: "{text_query}"', fontsize=16)
    
    gs = GridSpec(grid_size[0], grid_size[1], figure=plt.gcf())
    
    for i, result in enumerate(results):
        if i >= grid_size[0] * grid_size[1]:
            break
            
        img = Image.open(result['path'])
        similarity = result['similarity']
        
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        ax.set_title(f"Sim: {similarity:.4f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def display_comparison_results(finetuned_results, original_results, text_query, top_k=5, figsize=(18, 12)):
    """Hiển thị kết quả so sánh giữa model fine-tuned và model thuần"""
    plt.figure(figsize=figsize)
    plt.suptitle(f'So sánh kết quả truy vấn cho: "{text_query}"', fontsize=16)
    
    # Giới hạn số lượng kết quả hiển thị
    finetuned_results = finetuned_results[:top_k]
    original_results = original_results[:top_k]
    
    # Tạo lưới 2 cột, mỗi cột là kết quả của một model
    rows = top_k
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    
    # Tiêu đề cho mỗi cột
    axes[0, 0].set_title("Model CLIP đã fine-tune", fontsize=14)
    axes[0, 1].set_title("Model CLIP thuần", fontsize=14)
    
    # Hiển thị kết quả từ model fine-tuned ở cột trái
    for i, result in enumerate(finetuned_results):
        if i >= rows:
            break
            
        img = Image.open(result['path'])
        similarity = result['similarity']
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sim: {similarity:.4f}")
        axes[i, 0].axis('off')
    
    # Hiển thị kết quả từ model thuần ở cột phải
    for i, result in enumerate(original_results):
        if i >= rows:
            break
            
        img = Image.open(result['path'])
        similarity = result['similarity']
        
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f"Sim: {similarity:.4f}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def save_results(results, output_file):
    """Lưu kết quả tìm kiếm vào file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả vào {output_file}")

def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    # Các thông số cấu hình
    image_dir = "E:\\Đồ án chuyên ngành\\testing\\segment_Video_1"
    fine_tuned_model = "E:\\Đồ án chuyên ngành\\testing\\CLIP_v3_NSFW_RLVSD_50Agu\\last_model.pt"
    output_dir = "E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\clip_retrieval_results"
    top_k = 5
    query = "sexual content"  # Query mặc định
    queries_file = None  # Đường dẫn đến file JSON chứa danh sách queries (nếu có)
    
    # Các text queries mặc định cho sensitive content detection
    default_queries = [
        "violent content",
        "explicit nudity",
        "sexual content",
        "blood and gore",
        "weapon violence",
        "nudity",
        "disturbing scenes"
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Kiểm tra các đường dẫn
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục images tại: {image_dir}")
    
    if not os.path.exists(fine_tuned_model):
        raise FileNotFoundError(f"Không tìm thấy model tại: {fine_tuned_model}")
        
    # Kiểm tra CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    
    # Tải model fine-tuned
    print(f"Đang tải model fine-tuned từ {fine_tuned_model}...")
    finetuned_model = load_model(fine_tuned_model, device)
    
    # Tải model CLIP thuần
    print("Đang tải model CLIP thuần...")
    original_model = load_original_clip(device)
    
    # Xác định danh sách queries
    text_queries = []
    if queries_file and os.path.exists(queries_file):
        with open(queries_file, 'r') as f:
            text_queries = json.load(f)
        print(f"Loaded {len(text_queries)} queries from {queries_file}")
    else:
        # Sử dụng query đã chỉ định hoặc danh sách mặc định
        if query:
            text_queries = [query]
        else:
            text_queries = default_queries
            
    # Mã hóa frames cho model fine-tuned
    finetuned_features_path = os.path.join(output_dir, "finetuned_features.npz")
    if os.path.exists(finetuned_features_path):
        print(f"Đang tải features fine-tuned từ {finetuned_features_path}...")
        cached_data = np.load(finetuned_features_path, allow_pickle=True)
        finetuned_frame_features = cached_data['features']
        frame_paths = cached_data['paths']
    else:
        print(f"Đang mã hóa frames từ {image_dir} với model fine-tuned...")
        finetuned_frame_features, frame_paths = encode_frames(
            finetuned_model, image_dir, batch_size=64, device=device
        )
        
        # Lưu features nếu cần
        print(f"Đang lưu features fine-tuned vào {finetuned_features_path}...")
        np.savez_compressed(
            finetuned_features_path,
            features=finetuned_frame_features,
            paths=frame_paths
        )
    
    # Mã hóa frames cho model CLIP thuần
    original_features_path = os.path.join(output_dir, "vanilla_features.npz")
    if os.path.exists(original_features_path):
        print(f"Đang tải features CLIP thuần từ {original_features_path}...")
        cached_data = np.load(original_features_path, allow_pickle=True)
        original_frame_features = cached_data['features']
    else:
        print(f"Đang mã hóa frames từ {image_dir} với model CLIP thuần...")
        original_frame_features, _ = encode_frames(
            original_model, image_dir, batch_size=64, device=device
        )
        
        # Lưu features
        print(f"Đang lưu features CLIP thuần vào {original_features_path}...")
        np.savez_compressed(
            original_features_path,
            features=original_frame_features,
            paths=frame_paths
        )
    
    # Lặp qua từng query và tìm kiếm
    all_results = {}
    for query in text_queries:
        print(f"\nĐang xử lý query: '{query}'")
        
        # Tìm kiếm frames với model fine-tuned
        print(f"Đang tìm kiếm frames phù hợp với model fine-tuned...")
        finetuned_results = search_frames(
            finetuned_model, query, finetuned_frame_features, frame_paths, top_k=top_k, device=device
        )
        
        # Hiển thị kết quả từ model fine-tuned
        print("\nKết quả tìm kiếm từ model fine-tuned:")
        for i, result in enumerate(finetuned_results):
            print(f"{i+1}. {os.path.basename(result['path'])} - Similarity: {result['similarity']:.4f}")
        
        # Tìm kiếm frames với model CLIP thuần
        print(f"Đang tìm kiếm frames phù hợp với model CLIP thuần...")
        original_results = search_frames_original_clip(
            original_model, query, original_frame_features, frame_paths, top_k=top_k, device=device
        )
        
        # Hiển thị kết quả từ model CLIP thuần
        print("\nKết quả tìm kiếm từ model CLIP thuần:")
        for i, result in enumerate(original_results):
            print(f"{i+1}. {os.path.basename(result['path'])} - Similarity: {result['similarity']:.4f}")
    
        # Lưu kết quả cho query hiện tại
        query_results = {
            'query': query,
            'finetuned_results': finetuned_results,
            'original_results': original_results
        }
        all_results[query] = query_results
        
        # Lưu ảnh so sánh
        output_image = os.path.join(output_dir, f"compare_{query.replace(' ', '_')}.png")
        fig = plt.figure(figsize=(18, 10))
        plt.suptitle(f'So sánh kết quả truy vấn cho: "{query}"', fontsize=16)
        
        # Tiêu đề cho mỗi cột
        rows = min(5, top_k)
        fig, axes = plt.subplots(rows, 2, figsize=(18, rows*3))
        axes[0, 0].set_title("Model CLIP đã fine-tune", fontsize=14)
        axes[0, 1].set_title("Model CLIP thuần", fontsize=14)
        
        # Hiển thị kết quả từ model fine-tuned ở cột trái
        for i, result in enumerate(finetuned_results):
            if i >= rows:
                break
                
            try:
                img = Image.open(result['path'])
                folder_name = os.path.basename(os.path.dirname(result['path']))
                similarity = result['similarity']
                
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Sim: {similarity:.4f}\n{folder_name}")
                axes[i, 0].axis('off')
            except Exception as e:
                print(f"Error displaying image {result['path']}: {str(e)}")
        
        # Hiển thị kết quả từ model thuần ở cột phải
        for i, result in enumerate(original_results):
            if i >= rows:
                break
                
            try:
                img = Image.open(result['path'])
                folder_name = os.path.basename(os.path.dirname(result['path']))
                similarity = result['similarity']
                
                axes[i, 1].imshow(img)
                axes[i, 1].set_title(f"Sim: {similarity:.4f}\n{folder_name}")
                axes[i, 1].axis('off')
            except Exception as e:
                print(f"Error displaying image {result['path']}: {str(e)}")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_image)
        plt.close()
        print(f"Saved comparison image to {output_image}")
    
    # Lưu tất cả kết quả vào một file JSON
    results_file = os.path.join(output_dir, "all_retrieval_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu tất cả kết quả vào {results_file}")
    
    return all_results

if __name__ == '__main__':
    main() 