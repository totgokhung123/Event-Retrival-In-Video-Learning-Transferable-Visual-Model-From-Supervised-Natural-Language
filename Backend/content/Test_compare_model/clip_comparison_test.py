import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch.nn as nn
from torchvision.utils import save_image
import json
import shutil

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clip_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CLIPFinetuner(nn.Module):
    def __init__(self, pretrained="ViT-B/32", freeze_layers=0):
        """Load mô hình CLIP fine-tuned với cấu trúc từ clip_retrieval_finetune.py"""
        super(CLIPFinetuner, self).__init__()
        self.clip_model, self.preprocess = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.clip_model.parameters()).device
        
        # Thêm logit_scale parameter có thể trainable (thay vì dùng temperature cố định)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image_batch):
        """Encode images và trả về normalized features"""
        # Đảm bảo dùng float32 để tránh vấn đề NaN
        if image_batch.dtype != torch.float32:
            image_batch = image_batch.to(dtype=torch.float32)
            
        # Encode image
        image_features = self.clip_model.encode_image(image_batch)
            
        # Safe normalization - thêm epsilon để tránh chia cho 0
        norms = image_features.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        image_features = image_features / norms
        
        return image_features
        
    def encode_text(self, text_batch):
        """Encode text và trả về normalized features"""
        # Tokenize text
        text_tokens = clip.tokenize(text_batch).to(self.device)
        
        # Encode text
        text_features = self.clip_model.encode_text(text_tokens)
        
        # Safe normalization
        norms = text_features.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        text_features = text_features / norms
        
        return text_features

    def compute_similarity(self, image_features, text_features):
        """Tính similarity với logit_scale"""
        # Đảm bảo sử dụng float32
        image_features = image_features.float()
        text_features = text_features.float()
        
        # Áp dụng logit_scale
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * torch.matmul(image_features, text_features.t())
        
        return similarity

def load_original_clip(model_name="ViT-B/32"):
    """Tải mô hình CLIP nguyên bản"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def load_fine_tuned_clip(checkpoint_path, model_name="ViT-B/32"):
    """Tải mô hình CLIP đã fine-tune"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Khởi tạo model với cấu trúc từ clip_retrieval_finetune.py
    model = CLIPFinetuner(pretrained=model_name)
    model = model.to(device)
    
    # Tải weights từ checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Kiểm tra xem checkpoint có dạng state_dict trực tiếp hay là dict với state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Case: checkpoint from training loop with full optimizer state
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Loaded model state from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        # Case: checkpoint is just the model state dict
        state_dict = checkpoint
        logger.info("Loaded model state directly from checkpoint")
    
    # Xử lý trường hợp không có logit_scale trong checkpoint
    missing_keys = []
    try:
        # Thử load với strict=True trước
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "Missing key(s) in state_dict: \"logit_scale\"" in str(e):
            # Nếu chỉ thiếu logit_scale, load không strict và giữ nguyên logit_scale mặc định
            logger.warning("Checkpoint không có tham số logit_scale. Sử dụng giá trị mặc định (temperature=0.07)")
            model.load_state_dict(state_dict, strict=False)
        else:
            # Nếu có lỗi khác, raise lại
            raise e
    
    # Log giá trị logit_scale hiện tại
    logger.info(f"logit_scale sau khi load checkpoint: {model.logit_scale.item()}")
    
    return model, model.preprocess

def process_image_directory(image_dir, original_model, fine_tuned_model, preprocess, query_texts):
    """Xử lý thư mục ảnh và tính similarity với cả hai mô hình"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model.to(device)
    fine_tuned_model.to(device)
    
    # Bật eval mode
    original_model.eval()
    fine_tuned_model.eval()
    
    # Xác định các định dạng ảnh hợp lệ
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    # Thu thập đường dẫn ảnh
    image_paths = []
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1]
        if ext in valid_extensions:
            image_paths.append(os.path.join(image_dir, file))
    
    logger.info(f"Found {len(image_paths)} images to process")
    if len(image_paths) == 0:
        logger.error("No images found in directory!")
        return None
    
    # Encode queries
    with torch.no_grad():
        # Encode text query với CLIP nguyên bản
        original_text_features = original_model.encode_text(clip.tokenize(query_texts).to(device))
        original_text_features /= original_text_features.norm(dim=-1, keepdim=True)
        
        # Encode text query với CLIP fine-tuned
        fine_tuned_text_features = fine_tuned_model.encode_text(query_texts)
    
    results = []
    batch_size = 32  # Xử lý theo batch để tối ưu GPU
    
    # Xử lý từng batch ảnh
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        # Load và tiền xử lý ảnh
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                processed_image = preprocess(image)
                batch_images.append(processed_image)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
        
        if not batch_images:
            continue
            
        # Stack batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            # Encode images với CLIP nguyên bản
            original_image_features = original_model.encode_image(batch_tensor)
            original_image_features /= original_image_features.norm(dim=-1, keepdim=True)
            
            # Encode images với CLIP fine-tuned
            fine_tuned_image_features = fine_tuned_model.encode_image(batch_tensor)
            
            # Tính similarity cho mỗi query
            for q_idx, query in enumerate(query_texts):
                for img_idx, img_path in enumerate(valid_paths):
                    # So sánh với mô hình gốc
                    original_sim = (100.0 * original_image_features[img_idx] @ original_text_features[q_idx]).item()
                    
                    # So sánh với mô hình fine-tuned
                    # Lấy ra features của 1 ảnh và 1 query text
                    img_feat = fine_tuned_image_features[img_idx].unsqueeze(0)
                    txt_feat = fine_tuned_text_features[q_idx].unsqueeze(0)
                    # Tính similarity với logit_scale
                    fine_tuned_sim = fine_tuned_model.compute_similarity(img_feat, txt_feat).item()
                    
                    results.append({
                        'image_path': img_path,
                        'query': query,
                        'original_similarity': original_sim,
                        'fine_tuned_similarity': fine_tuned_sim,
                        'similarity_diff': fine_tuned_sim - original_sim
                    })
    
    return results

def save_comparison_results(results, output_dir, top_k=5):
    """Lưu kết quả so sánh và tạo hình ảnh trực quan"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Tổ chức kết quả theo query
    results_by_query = {}
    for item in results:
        query = item['query']
        if query not in results_by_query:
            results_by_query[query] = []
        results_by_query[query].append(item)
    
    # Xử lý từng query
    for query, query_results in results_by_query.items():
        # Thư mục cho query này
        query_dir = os.path.join(output_dir, query.replace(" ", "_"))
        os.makedirs(query_dir, exist_ok=True)
        
        # Sắp xếp theo CLIP gốc
        original_sorted = sorted(query_results, key=lambda x: x['original_similarity'], reverse=True)
        # Sắp xếp theo CLIP fine-tuned
        fine_tuned_sorted = sorted(query_results, key=lambda x: x['fine_tuned_similarity'], reverse=True)
        # Sắp xếp theo độ chênh lệch
        diff_sorted = sorted(query_results, key=lambda x: x['similarity_diff'], reverse=True)
        
        # Lưu kết quả top-K
        for sort_type, sorted_results in [
            ("original", original_sorted), 
            ("fine_tuned", fine_tuned_sorted), 
            ("diff", diff_sorted)
        ]:
            type_dir = os.path.join(query_dir, f"top_{sort_type}")
            os.makedirs(type_dir, exist_ok=True)
            
            # Lưu top-K ảnh
            for i, result in enumerate(sorted_results[:top_k]):
                img_path = result['image_path']
                # Tạo tên file mới
                filename = f"{i+1:02d}_orig{result['original_similarity']:.2f}_ft{result['fine_tuned_similarity']:.2f}.jpg"
                dst_path = os.path.join(type_dir, filename)
                try:
                    shutil.copy(img_path, dst_path)
                except Exception as e:
                    logger.error(f"Error copying {img_path} to {dst_path}: {e}")
        
        # Lưu tất cả kết quả dạng JSON để phân tích sau
        with open(os.path.join(query_dir, "results.json"), "w") as f:
            json.dump(query_results, f, indent=2)
        
        # Tạo report tổng hợp
        with open(os.path.join(query_dir, "report.txt"), "w") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Total images: {len(query_results)}\n\n")
            
            # Top-K theo mô hình gốc
            f.write(f"Top {top_k} by Original CLIP:\n")
            for i, item in enumerate(original_sorted[:top_k]):
                f.write(f"{i+1:2d}. {os.path.basename(item['image_path'])}: {item['original_similarity']:.2f} (Fine-tuned: {item['fine_tuned_similarity']:.2f})\n")
            
            # Top-K theo mô hình fine-tuned
            f.write(f"\nTop {top_k} by Fine-tuned CLIP:\n")
            for i, item in enumerate(fine_tuned_sorted[:top_k]):
                f.write(f"{i+1:2d}. {os.path.basename(item['image_path'])}: {item['fine_tuned_similarity']:.2f} (Original: {item['original_similarity']:.2f})\n")
            
            # Phân tích accuracy
            common_in_top_k = set([item['image_path'] for item in original_sorted[:top_k]]) & set([item['image_path'] for item in fine_tuned_sorted[:top_k]])
            f.write(f"\nOverlap in top {top_k}: {len(common_in_top_k)}/{top_k} ({len(common_in_top_k)/top_k*100:.2f}%)\n")
            
            # Phân tích similarity distribution
            original_sims = [item['original_similarity'] for item in query_results]
            fine_tuned_sims = [item['fine_tuned_similarity'] for item in query_results]
            f.write(f"\nOriginal CLIP similarity: min={min(original_sims):.2f}, max={max(original_sims):.2f}, mean={np.mean(original_sims):.2f}\n")
            f.write(f"Fine-tuned CLIP similarity: min={min(fine_tuned_sims):.2f}, max={max(fine_tuned_sims):.2f}, mean={np.mean(fine_tuned_sims):.2f}\n")
    
    logger.info(f"Results saved to {output_dir}")
    return True

def plot_similarity_comparison(results, output_file):
    """Vẽ biểu đồ scatter plot so sánh similarity giữa hai mô hình"""
    plt.figure(figsize=(10, 8))
    
    # Chuẩn bị dữ liệu
    original_scores = [item['original_similarity'] for item in results]
    fine_tuned_scores = [item['fine_tuned_similarity'] for item in results]
    
    # Vẽ scatter plot
    plt.scatter(original_scores, fine_tuned_scores, alpha=0.5)
    
    # Vẽ đường y=x để so sánh
    max_val = max(max(original_scores), max(fine_tuned_scores))
    min_val = min(min(original_scores), min(fine_tuned_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    # Thêm thông tin cho biểu đồ
    plt.xlabel('Original CLIP Similarity')
    plt.ylabel('Fine-tuned CLIP Similarity')
    plt.title('Comparison of Similarity Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Similarity comparison plot saved to {output_file}")

def main():
    # Thay thế argparse bằng khai báo biến trực tiếp
    image_dir = "E:\\Đồ án chuyên ngành\\dataset\\testseg\\datasosanh\\NSFW_Violence\\cam1_2"
    fine_tuned_model = "E:\\Đồ án chuyên ngành\\testing\\CLIP_v3_NSFW_RLVSD_50Agu\\best_model_v4_3.pt"
    output_dir = "E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\clip_comparison_results"
    clip_model = "ViT-B/32"
    top_k = 5
    queries = "a group nude woman stack each other"
    
    # Chuyển đổi queries thành list nếu là string
    if isinstance(queries, str):
        queries = [queries]
    
    # Kiểm tra đường dẫn hợp lệ
    for path_name, path in [
        ("Image directory", image_dir),
        ("Fine-tuned model", fine_tuned_model)
    ]:
        if not os.path.exists(path):
            logger.error(f"{path_name} not found: {path}")
            return
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    logger.info(f"Loading original CLIP model ({clip_model})...")
    original_model, preprocess = load_original_clip(clip_model)
    
    logger.info(f"Loading fine-tuned CLIP model from {fine_tuned_model}...")
    fine_tuned_model_loaded, _ = load_fine_tuned_clip(fine_tuned_model, clip_model)
    
    # Kiểm tra logit_scale của mô hình fine-tuned
    logit_scale_value = fine_tuned_model_loaded.logit_scale.item()
    temperature = np.exp(-logit_scale_value)
    logger.info(f"Fine-tuned model logit_scale: {logit_scale_value} (temperature ≈ {temperature:.4f})")
    
    logger.info(f"Using queries: {queries}")
    
    # Xử lý ảnh và tính toán similarity
    results = process_image_directory(image_dir, original_model, fine_tuned_model_loaded, preprocess, queries)
    
    if results:
        # Lưu kết quả
        save_comparison_results(results, output_dir, top_k)
        
        # Vẽ biểu đồ so sánh
        plot_output = os.path.join(output_dir, "similarity_comparison.png")
        plot_similarity_comparison(results, plot_output)
        
        logger.info("Comparison completed successfully!")
    else:
        logger.error("Failed to process images!")

if __name__ == "__main__":
    main() 