import os
import json
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
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPFineTuner(nn.Module):
    def __init__(self, num_classes=2, pretrained="ViT-B/32", training_phase=3):
        """Model CLIPFineTuner từ train_CLIP_v3.py với fusion layer"""
        super(CLIPFineTuner, self).__init__()
        self.clip_model, self.preprocess = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.clip_model.visual.output_dim
        self.device = next(self.clip_model.parameters()).device
        self.training_phase = training_phase
        
        # Khai báo blocks để tương thích với checkpoint của train_CLIP_v3.py
        if hasattr(self.clip_model, 'visual') and hasattr(self.clip_model.visual, 'transformer'):
            # Đối với ViT
            self.visual_blocks = self.clip_model.visual.transformer.resblocks
            self.text_blocks = self.clip_model.transformer.resblocks
        
        # Đóng băng CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Layer normalization và dropout ở các lớp mới
        self.image_norm = nn.LayerNorm(embedding_dim)
        self.text_norm = nn.LayerNorm(embedding_dim)
        
        # Fusion layer cải tiến với normalization và dropout
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Image và text classifiers riêng
        self.image_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes)
        )
        
        logger.info(f"Khởi tạo CLIPFineTuner với training_phase={training_phase}")
    
    def load_state_dict(self, state_dict, strict=False):
        """
        Override load_state_dict để xử lý trường hợp state_dict có khóa không khớp.
        Đặt strict=False để bỏ qua các khóa không khớp.
        """
        return super().load_state_dict(state_dict, strict=strict)
    
    def forward(self, image_batch=None, caption_batch=None):
        """Forward pass với cả image và text."""
        outputs = {}
        
        # Encode image nếu có
        if image_batch is not None:
            with torch.no_grad():  # Luôn sử dụng no_grad vì đã đóng băng CLIP
                image_features = self.clip_model.encode_image(image_batch.float())
                
            # Chuyển sang float32 để tránh vấn đề dtype
            image_features = image_features.float()
            
            # Normalize image features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            image_features = self.image_norm(image_features)
            
            # Lưu features và tính logits
            outputs['image_features'] = image_features
            outputs['image_logits'] = self.image_classifier(image_features)
        
        # Encode text nếu có
        if caption_batch is not None:
            text_tokens = clip.tokenize(caption_batch).to(self.device)
            with torch.no_grad():  # Luôn sử dụng no_grad vì đã đóng băng CLIP
                text_features = self.clip_model.encode_text(text_tokens)
                
            # Chuyển sang float32
            text_features = text_features.float()
            
            # Normalize text features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = self.text_norm(text_features)
            
            # Lưu features và tính logits
            outputs['text_features'] = text_features
            outputs['text_logits'] = self.text_classifier(text_features)
        
        # Fusion của image và text features nếu cả hai đều có
        if image_batch is not None and caption_batch is not None:
            combined_features = torch.cat([image_features, text_features], dim=1)
            fused_features = self.fusion(combined_features)
            outputs['fused_features'] = fused_features
            outputs['fused_logits'] = self.classifier(fused_features)
        
        return outputs

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
    """Tải model CLIP multimodal đã fine-tuned"""
    model = CLIPFineTuner(
        pretrained="ViT-B/32", 
        training_phase=3  # Set phase 3 để tương thích với model đã train
    ).to(device)
    
    # Tải weights đã lưu
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:  # Nếu load từ checkpoint
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
    else:  # Nếu load từ best_model
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    logger.info(f"Đã tải model CLIPFineTuner từ {model_path}")
    return model

def encode_frames(model, frame_dir, batch_size=32, device='cuda', memory_efficient=True):
    """Mã hóa tất cả các frame trong thư mục frame_dir bằng model multimodal"""
    # Khởi tạo transform cho validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Tạo dataset và dataloader
    dataset = FrameDataset(frame_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Lưu features và logits
    frame_features = []     # image features từ CLIP
    frame_image_logits = [] # logits từ image classifier
    frame_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đang mã hóa frames"):
            # Giải phóng bộ nhớ cache định kỳ
            if memory_efficient and torch.cuda.is_available() and batch_size > 1:
                torch.cuda.empty_cache()
                
            images = batch['image'].to(device)
            outputs = model(images)
            
            # Lưu features, logits và đường dẫn
            frame_features.append(outputs['image_features'].cpu().numpy())
            frame_image_logits.append(outputs['image_logits'].cpu().numpy())
            frame_paths.extend(batch['path'])
    
    # Ghép tất cả lại
    frame_features = np.vstack(frame_features)
    frame_image_logits = np.vstack(frame_image_logits)
    
    # Tính softmax cho logits để có probability
    frame_probs = softmax(frame_image_logits, axis=1)
    
    print(f"Đã mã hóa {len(frame_paths)} frames")
    
    return {
        'features': frame_features,
        'logits': frame_image_logits,
        'probs': frame_probs,
        'paths': frame_paths
    }

def softmax(x, axis=None):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def search_frames(model, text_query, encoded_frames, top_k=10, device='cuda', 
                  violence_filter=None, retrieval_mode='multimodal'):
    """
    Tìm kiếm frames dựa trên text query và có thể lọc theo xác suất bạo lực
    
    Args:
        violence_filter: None hoặc float trong [0, 1]. Nếu là float, chỉ giữ lại các frame có
                         xác suất bạo lực >= violence_filter
        retrieval_mode: 'multimodal' (dùng cả image và text) hoặc 'classification' (sắp xếp theo xác suất violence)
    """
    frame_features = encoded_frames['features']
    frame_probs = encoded_frames['probs']
    frame_paths = encoded_frames['paths']
    
    # Áp dụng bộ lọc violence nếu cần
    filtered_indices = np.arange(len(frame_paths))
    if violence_filter is not None and 0 <= violence_filter <= 1:
        # Lấy column thứ 1 (Violence) từ frame_probs
        violence_probs = frame_probs[:, 1]
        filtered_indices = np.where(violence_probs >= violence_filter)[0]
        
        if len(filtered_indices) == 0:
            print(f"Không có frame nào có xác suất bạo lực >= {violence_filter}")
            return []
        
        print(f"Đã lọc: còn {len(filtered_indices)}/{len(frame_paths)} frames với xác suất bạo lực >= {violence_filter}")
    
    # Tạo danh sách paths và features sau khi lọc
    filtered_paths = [frame_paths[i] for i in filtered_indices]
    filtered_features = frame_features[filtered_indices]
    filtered_probs = frame_probs[filtered_indices]
    
    # Nếu mode là classification, trả về frames có xác suất bạo lực cao nhất
    if retrieval_mode == 'classification':
        violence_probs = filtered_probs[:, 1]  # Cột thứ hai là probability của Violence
        top_indices = np.argsort(-violence_probs)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'path': filtered_paths[idx],
                'violence_prob': float(filtered_probs[idx, 1]),
                'nonviolence_prob': float(filtered_probs[idx, 0])
            })
        
        return results
    
    # Nếu không, tiếp tục với multimodal retrieval
    with torch.no_grad():
        # Mã hóa text query
        outputs = model(caption_batch=[text_query])
        text_features = outputs["text_features"].cpu().numpy()
    
    # Tính cosine similarity giữa image features và text features
    similarities = np.dot(filtered_features, text_features.T).squeeze()
    
    # Lấy top-k frames có similarity cao nhất
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'path': filtered_paths[idx],
            'similarity': float(similarities[idx]),
            'violence_prob': float(filtered_probs[idx, 1]),
            'nonviolence_prob': float(filtered_probs[idx, 0])
        })
    
    return results

def display_results(results, text_query=None, grid_size=(3, 4), figsize=(15, 12), 
                   show_violence_prob=True, retrieval_mode='multimodal'):
    """Hiển thị kết quả tìm kiếm dưới dạng lưới ảnh"""
    plt.figure(figsize=figsize)
    
    if text_query:
        title = f'Kết quả truy vấn cho: "{text_query}"'
    else:
        title = 'Frames có xác suất bạo lực cao nhất' if retrieval_mode == 'classification' else 'Kết quả tìm kiếm'
    
    plt.suptitle(title, fontsize=16)
    
    gs = GridSpec(grid_size[0], grid_size[1], figure=plt.gcf())
    
    for i, result in enumerate(results):
        if i >= grid_size[0] * grid_size[1]:
            break
            
        img = Image.open(result['path'])
        
        # Tạo tiêu đề cho ảnh
        if retrieval_mode == 'multimodal':
            similarity = result['similarity']
            title = f"Sim: {similarity:.3f}"
        else:
            title = ""
            
        if show_violence_prob and 'violence_prob' in result:
            violence_prob = result['violence_prob']
            title += f"\nViolence: {violence_prob:.2f}"
        
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def save_results(results, output_file):
    """Lưu kết quả tìm kiếm vào file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả vào {output_file}")

def main():
    # Tối ưu hóa bộ nhớ CUDA
    memory_efficient = True  # Bật các tối ưu hóa bộ nhớ
    if memory_efficient and torch.cuda.is_available():
        # Giải phóng bộ nhớ cache
        torch.cuda.empty_cache()
        
        # Thiết lập các thông số phân bổ CUDA để tránh phân mảnh
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Đặt max_split_size để tránh phân mảnh
        torch.cuda.set_per_process_memory_fraction(0.8)  # Chỉ sử dụng 80% bộ nhớ GPU
        logger.info("Đã áp dụng các tối ưu hóa bộ nhớ CUDA")
    
    # Khai báo cứng các tham số thay vì dùng argparse
    # Thay đổi các giá trị này trực tiếp trong code nếu cần
    model_path = 'D:/code/projects/NCKH/model_CLIP_fine_tune/v3/best_model_phase2_v3_falsefreeze.pt'  # Đường dẫn đến model đã fine-tune
    frame_dir = "D:/code/projects/NCKH/video_frame/video_test_training_CLIP/segment_Video_1"  # Thư mục chứa frames
    text_query = "a group of teenagers were fighting, pulling each other's hair"  # Text query để tìm kiếm (nếu để trống sẽ nhận từ input)
    top_k = 10  # Số lượng frames kết quả trả về
    output_file = 'results_multimodal_v3.json'  # File để lưu kết quả (JSON), để trống nếu không lưu
    batch_size = 32  # Batch size khi xử lý frames
    cache_features = True  # Có lưu features đã tính toán hay không
    features_path = 'multimodal_v3_frame_features.npz'  # File để lưu/tải features
    display_results_flag = True  # Có hiển thị kết quả bằng matplotlib hay không
    violence_filter = None  # Lọc theo xác suất bạo lực (None: không lọc, float trong [0, 1])
    retrieval_mode = 'multimodal'  # 'multimodal' hoặc 'classification'
    
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
    
    # Lấy text query nếu cần
    if retrieval_mode == 'multimodal' and not text_query:
        text_query = input("Nhập text query để tìm kiếm frames: ")
    
    # Mã hóa frames hoặc tải từ cache
    encoded_frames = None
    if os.path.exists(features_path) and cache_features:
        print(f"Đang tải features từ {features_path}...")
        encoded_frames = {k: v for k, v in np.load(features_path, allow_pickle=True).items()}
    else:
        print(f"Đang mã hóa frames từ {frame_dir}...")
        start_time = time.time()
        encoded_frames = encode_frames(
            model, frame_dir, batch_size=batch_size, device=device, memory_efficient=memory_efficient
        )
        print(f"Thời gian mã hóa: {time.time() - start_time:.2f} giây")
        
        # Lưu features nếu cần
        if cache_features:
            print(f"Đang lưu features vào {features_path}...")
            np.savez_compressed(features_path, **encoded_frames)
    
    # Tìm kiếm frames dựa trên text query hoặc xác suất bạo lực
    if retrieval_mode == 'multimodal':
        print(f"Đang tìm kiếm frames phù hợp với query: '{text_query}'...")
    else:
        print(f"Đang tìm kiếm frames có xác suất bạo lực cao nhất...")
        
    start_time = time.time()
    results = search_frames(
        model, text_query, encoded_frames, top_k=top_k, device=device,
        violence_filter=violence_filter, retrieval_mode=retrieval_mode
    )
    print(f"Thời gian tìm kiếm: {time.time() - start_time:.2f} giây")
    
    if not results:
        print("Không tìm thấy kết quả nào thỏa mãn điều kiện.")
        return []
    
    # Hiển thị kết quả
    print("\nKết quả tìm kiếm:")
    for i, result in enumerate(results):
        filename = os.path.basename(result['path'])
        if retrieval_mode == 'multimodal':
            print(f"{i+1}. {filename} - Similarity: {result['similarity']:.4f}, Violence Prob: {result['violence_prob']:.4f}")
        else:
            print(f"{i+1}. {filename} - Violence Prob: {result['violence_prob']:.4f}")
    
    # Lưu kết quả nếu cần
    if output_file:
        save_results(results, output_file)
    
    # Hiển thị kết quả bằng matplotlib nếu cần
    if display_results_flag:
        display_results(
            results, 
            text_query=(text_query if retrieval_mode == 'multimodal' else None),
            show_violence_prob=True,
            retrieval_mode=retrieval_mode
        )
    
    return results

if __name__ == '__main__':
    main() 