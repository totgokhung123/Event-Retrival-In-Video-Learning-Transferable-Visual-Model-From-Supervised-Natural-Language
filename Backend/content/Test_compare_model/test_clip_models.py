import os
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from tqdm import tqdm
import shutil
import sys

# Import mô hình CLIPWithClassifier từ file clip_finetune_correct.py
sys.path.append('.')
from clip_finetune_correct import CLIPWithClassifier

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_clip_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_original_clip(model_name="ViT-B/32", device=None):
    """Tải mô hình CLIP nguyên bản"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def load_finetuned_clip(checkpoint_path, model_name="ViT-B/32", device=None):
    """Tải mô hình CLIP đã fine-tune với CLIPWithClassifier"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tải mô hình CLIP gốc
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)
    
    # Khởi tạo CLIPWithClassifier từ clip_finetune_correct.py
    model = CLIPWithClassifier(clip_model, num_classes=3)
    model = model.to(device)
    model = model.float()
    
    # Tải checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Kiểm tra định dạng checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Đã tải model_state_dict từ checkpoint {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"Đã tải trực tiếp state_dict từ {checkpoint_path}")
    
    model.eval()  # Đặt model ở chế độ evaluation
    return model, preprocess

def run_comparison_test(original_model, finetuned_model, image_dir, queries, preprocess, device=None):
    """Chạy so sánh giữa hai mô hình CLIP trên cùng một tập dữ liệu"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tìm tất cả các file hình ảnh trong thư mục
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(image_dir, file))
    
    logger.info(f"Tìm thấy {len(image_paths)} hình ảnh trong thư mục {image_dir}")
    
    # Chuẩn bị truy vấn
    if isinstance(queries, str):
        queries = [queries]
    
    text_tokens = clip.tokenize(queries).to(device)
    
    # Kết quả so sánh
    results = []
    
    # Chạy so sánh
    with torch.no_grad():
        # Mã hóa text trước
        original_text_features = original_model.encode_text(text_tokens)
        original_text_features /= original_text_features.norm(dim=-1, keepdim=True)
        
        # Với model fine-tuned, cần truy cập clip_model bên trong
        finetuned_text_features = finetuned_model.clip_model.encode_text(text_tokens)
        finetuned_text_features /= finetuned_text_features.norm(dim=-1, keepdim=True)
        
        # Xử lý từng ảnh
        for image_path in tqdm(image_paths, desc="Đang phân tích ảnh"):
            try:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                
                # Mã hóa ảnh với cả hai mô hình
                original_image_features = original_model.encode_image(image)
                original_image_features /= original_image_features.norm(dim=-1, keepdim=True)
                
                # Với model fine-tuned, cần truy cập clip_model bên trong
                finetuned_image_features = finetuned_model.clip_model.encode_image(image)
                finetuned_image_features /= finetuned_image_features.norm(dim=-1, keepdim=True)
                
                # Thêm thông tin phân loại nếu có
                class_info = {}
                try:
                    # Dùng classifier head để phân loại
                    with torch.no_grad():
                        class_logits = finetuned_model.classifier(finetuned_image_features)
                        class_probs = torch.softmax(class_logits, dim=-1)[0].cpu().numpy()
                        class_info = {
                            "sensitive_prob": float(class_probs[0]),
                            "violence_prob": float(class_probs[1]),
                            "normal_prob": float(class_probs[2])
                        }
                except Exception as e:
                    logger.warning(f"Không thể lấy thông tin phân loại: {str(e)}")
                
                # Tính toán độ tương đồng cho từng truy vấn
                for i, query in enumerate(queries):
                    original_similarity = (100.0 * original_image_features @ original_text_features[i].unsqueeze(0).T).item()
                    finetuned_similarity = (100.0 * finetuned_image_features @ finetuned_text_features[i].unsqueeze(0).T).item()
                    
                    result_item = {
                        "image": os.path.basename(image_path),
                        "query": query,
                        "original_similarity": original_similarity,
                        "finetuned_similarity": finetuned_similarity,
                        "difference": finetuned_similarity - original_similarity
                    }
                    
                    # Thêm thông tin phân loại nếu có
                    if class_info:
                        result_item.update(class_info)
                    
                    results.append(result_item)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
    
    return results

def visualize_results(results, output_dir, top_k=5):
    """Trực quan hóa kết quả so sánh"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Nhóm kết quả theo truy vấn
    results_by_query = {}
    for item in results:
        query = item["query"]
        if query not in results_by_query:
            results_by_query[query] = []
        results_by_query[query].append(item)
    
    # Tạo báo cáo tổng hợp
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("SO SÁNH GIỮA MÔ HÌNH CLIP GỐC VÀ MÔ HÌNH FINE-TUNED\n")
        f.write("=" * 80 + "\n\n")
        
        for query, query_results in results_by_query.items():
            f.write(f"TRUY VẤN: \"{query}\"\n")
            f.write("-" * 50 + "\n")
            
            # Sắp xếp theo độ tương đồng của mô hình fine-tuned
            sorted_results = sorted(query_results, key=lambda x: x["finetuned_similarity"], reverse=True)
            
            # Top-K của mô hình fine-tuned
            f.write(f"Top {top_k} ảnh có độ tương đồng cao nhất với mô hình fine-tuned:\n")
            for i, item in enumerate(sorted_results[:top_k]):
                info_str = f"{i+1}. {item['image']}: {item['finetuned_similarity']:.2f} " \
                           f"(CLIP gốc: {item['original_similarity']:.2f}, chênh lệch: {item['difference']:.2f})"
                
                # Thêm thông tin phân loại nếu có
                if "sensitive_prob" in item:
                    info_str += f" | Sensitive: {item['sensitive_prob']:.2f}, Violence: {item['violence_prob']:.2f}, Normal: {item['normal_prob']:.2f}"
                
                f.write(info_str + "\n")
            
            # Thống kê độ chênh lệch
            diffs = [item["difference"] for item in query_results]
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            min_diff = min(diffs)
            
            f.write(f"\nThống kê độ chênh lệch cho \"{query}\":\n")
            f.write(f"- Trung bình: {avg_diff:.2f}\n")
            f.write(f"- Cao nhất: {max_diff:.2f}\n")
            f.write(f"- Thấp nhất: {min_diff:.2f}\n")
            
            # Tỷ lệ cải thiện
            improved_count = sum(1 for d in diffs if d > 0)
            improvement_rate = improved_count / len(diffs) * 100
            f.write(f"- Tỷ lệ cải thiện: {improved_count}/{len(diffs)} ({improvement_rate:.2f}%)\n\n")
    
    # Tạo biểu đồ so sánh
    for query, query_results in results_by_query.items():
        plt.figure(figsize=(10, 6))
        
        original_scores = [item["original_similarity"] for item in query_results]
        finetuned_scores = [item["finetuned_similarity"] for item in query_results]
        
        plt.scatter(original_scores, finetuned_scores, alpha=0.6)
        
        # Vẽ đường chéo y=x
        max_val = max(max(original_scores), max(finetuned_scores))
        min_val = min(min(original_scores), min(finetuned_scores))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        
        plt.xlabel('Điểm tương đồng - CLIP gốc')
        plt.ylabel('Điểm tương đồng - CLIP fine-tuned')
        plt.title(f'So sánh độ tương đồng cho truy vấn "{query}"')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        chart_path = os.path.join(output_dir, f"chart_{query.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        
        # Tạo biểu đồ phân loại nếu có dữ liệu
        if "sensitive_prob" in query_results[0]:
            plt.figure(figsize=(10, 6))
            
            # Sắp xếp theo độ tương đồng với query
            sorted_results = sorted(query_results, key=lambda x: x["finetuned_similarity"], reverse=True)
            top_results = sorted_results[:min(20, len(sorted_results))]  # Lấy top 20 kết quả
            
            sensitive_probs = [item["sensitive_prob"] for item in top_results]
            violence_probs = [item["violence_prob"] for item in top_results]
            normal_probs = [item["normal_prob"] for item in top_results]
            image_names = [item["image"] for item in top_results]
            
            x = range(len(top_results))
            width = 0.25
            
            plt.bar([i-width for i in x], sensitive_probs, width=width, label='Sensitive')
            plt.bar(x, violence_probs, width=width, label='Violence')
            plt.bar([i+width for i in x], normal_probs, width=width, label='Normal')
            
            plt.xlabel('Ảnh')
            plt.ylabel('Xác suất phân lớp')
            plt.title(f'Kết quả phân lớp cho top ảnh tương đồng với "{query}"')
            plt.xticks(x, [f"{i+1}" for i in range(len(top_results))], rotation=45)
            plt.legend()
            plt.tight_layout()
            
            chart_class_path = os.path.join(output_dir, f"class_chart_{query.replace(' ', '_')}.png")
            plt.savefig(chart_class_path)
            plt.close()
    
    # Lưu kết quả dạng JSON để phân tích sau
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Đã lưu báo cáo tổng hợp tại {summary_path}")
    logger.info(f"Đã lưu kết quả chi tiết tại {json_path}")

def main():
    # Cấu hình
    CONFIG = {
        'image_dir':"E:\\Đồ án chuyên ngành\\testing\\segment_Video_1", #"E:\\Đồ án chuyên ngành\\dataset\\testseg\\datasosanh\\NSFW_Violence\\cam1_2",
        'original_clip': "ViT-B/32",
        'finetuned_clip': "E:\\Đồ án chuyên ngành\\testing\\CLIP_v3_NSFW_RLVSD_50Agu\\final_checkpoint_hoanthanh.pt",
        'output_dir': "E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\clip_model_comparison_final",
        'queries': "a violence scene, many people fighting, pulling each other hair",
        'top_k': 5
    }
    
    # Thiết lập device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Sử dụng device: {device}")
    
    # Tải mô hình CLIP gốc
    logger.info(f"Đang tải mô hình CLIP gốc ({CONFIG['original_clip']})...")
    original_model, preprocess = load_original_clip(CONFIG['original_clip'], device)
    
    # Tải mô hình CLIP fine-tuned
    logger.info(f"Đang tải mô hình CLIP fine-tuned từ {CONFIG['finetuned_clip']}...")
    finetuned_model, _ = load_finetuned_clip(CONFIG['finetuned_clip'], CONFIG['original_clip'], device)
    
    # Chạy so sánh
    logger.info("Đang so sánh hiệu suất giữa hai mô hình...")
    results = run_comparison_test(
        original_model,
        finetuned_model,
        CONFIG['image_dir'],
        CONFIG['queries'],
        preprocess,
        device
    )
    
    # Trực quan hóa kết quả
    logger.info("Đang tạo báo cáo và trực quan hóa kết quả...")
    visualize_results(results, CONFIG['output_dir'], CONFIG['top_k'])
    
    logger.info("Quá trình so sánh đã hoàn tất!")

if __name__ == "__main__":
    main() 