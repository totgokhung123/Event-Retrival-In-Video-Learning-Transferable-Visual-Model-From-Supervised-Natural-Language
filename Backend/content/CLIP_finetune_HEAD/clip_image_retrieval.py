import os
import argparse
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

# Sử dụng lại khai báo model từ training_CLIP.py
class CLIPFineTuner(nn.Module):
    def __init__(self, num_classes=2, pretrained="ViT-B/32", freeze_clip=True):
        super(CLIPFineTuner, self).__init__()
        self.clip_model, self.preprocess = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.clip_model.visual.output_dim
        self.device = next(self.clip_model.parameters()).device

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Thêm fusion layer để kết hợp đặc trưng ảnh và text
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),  # Kết hợp image + text features
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier chung cho đặc trưng đã kết hợp
        self.classifier = nn.Linear(512, num_classes)
        
        # Thêm classifier riêng cho image và text features (cho additional loss)
        self.image_classifier = nn.Linear(embedding_dim, num_classes)
        self.text_classifier = nn.Linear(embedding_dim, num_classes)
        
        # Hệ số cho các thành phần loss
        self.alpha = 0.7  # Trọng số cho fusion loss
        self.beta = 0.15  # Trọng số cho image loss
        self.gamma = 0.15  # Trọng số cho text loss

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
                    
                    # Ép kiểu về float32 để đảm bảo tính toán chính xác
                    image_features = image_features.float()
                    text_features = text_features.float()
                    
                    # Tính cosine similarity giữa image và text
                    similarity = torch.matmul(image_features, text_features.t())
                    return {
                        'image_features': image_features,
                        'text_features': text_features,
                        'similarity': similarity
                    }
                
                # Nếu chỉ có image, trả về image features để lưu trữ
                return {
                    'image_features': image_features
                }

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
        model.load_state_dict(state_dict['model_state_dict'])
    else:  # Nếu load từ best_model
        model.load_state_dict(state_dict)
        
    model.eval()
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
            
            # Lưu features và đường dẫn
            frame_features.append(outputs['image_features'].cpu().numpy())
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

def display_results(results, text_query, grid_size=(3, 3), figsize=(15, 15)):
    """Hiển thị kết quả tìm kiếm dưới dạng lưới ảnh"""
    plt.figure(figsize=figsize)
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

def save_results(results, output_file):
    """Lưu kết quả tìm kiếm vào file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả vào {output_file}")

def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    # Thay đổi các giá trị này trực tiếp trong code nếu cần
    model_path = 'D:/code/projects/NCKH/model_CLIP_fine_tune/best_model_v2_CoAgu_20p.pt'  # Đường dẫn đến model đã fine-tune
    frame_dir = "D:/code/projects/NCKH/video_frame/video_test_training_CLIP/segment_Video_1"  # Thư mục chứa frames
    text_query ="a group of teenagers were fighting, pulling each other's hair"  # Text query để tìm kiếm (nếu để trống sẽ nhận từ input)
    top_k = 10  # Số lượng frames kết quả trả về
    output_file = 'results.json'  # File để lưu kết quả (JSON), để trống nếu không lưu
    batch_size = 32  # Batch size khi xử lý frames
    cache_features = True  # Có lưu features đã tính toán hay không
    features_path = 'frame_features.npz'  # File để lưu/tải features
    display_results_flag = True  # Có hiển thị kết quả bằng matplotlib hay không
    
    # Kiểm tra các đường dẫn
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
    
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục frames tại: {frame_dir}")
        
    # Kiểm tra CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    
    # Tải model
    print(f"Đang tải model từ {model_path}...")
    model = load_model(model_path, device)
    
    # Lấy text query
    if not text_query:
        text_query = input("Nhập text query để tìm kiếm frames: ")
    
    # Mã hóa frames hoặc tải từ cache
    if os.path.exists(features_path) and cache_features:
        print(f"Đang tải features từ {features_path}...")
        cached_data = np.load(features_path, allow_pickle=True)
        frame_features = cached_data['features']
        frame_paths = cached_data['paths']
    else:
        print(f"Đang mã hóa frames từ {frame_dir}...")
        frame_features, frame_paths = encode_frames(
            model, frame_dir, batch_size=batch_size, device=device
        )
        
        # Lưu features nếu cần
        if cache_features:
            print(f"Đang lưu features vào {features_path}...")
            np.savez_compressed(
                features_path,
                features=frame_features,
                paths=frame_paths
            )
    
    # Tìm kiếm frames dựa trên text query
    print(f"Đang tìm kiếm frames phù hợp với query: '{text_query}'...")
    results = search_frames(
        model, text_query, frame_features, frame_paths, top_k=top_k, device=device
    )
    
    # Hiển thị kết quả
    print("\nKết quả tìm kiếm:")
    for i, result in enumerate(results):
        print(f"{i+1}. {os.path.basename(result['path'])} - Similarity: {result['similarity']:.4f}")
    
    # Lưu kết quả nếu cần
    if output_file:
        save_results(results, output_file)
    
    # Hiển thị kết quả bằng matplotlib nếu cần
    if display_results_flag:
        display_results(results, text_query)
    
    return results

if __name__ == '__main__':
    main() 