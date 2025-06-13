import os
import json
import torch
import torch.nn as nn
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
import torchvision.transforms as transforms

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clip_diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiagnosticDataset(Dataset):
    """Dataset lấy mẫu từ dataset gốc để chẩn đoán"""
    def __init__(self, json_path, base_dir, transform=None, max_samples=100):
        self.base_dir = base_dir
        self.transform = transform
        
        # Đọc dữ liệu từ JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Giới hạn số lượng mẫu cho chẩn đoán
        self.items = []
        self.categories = set()
        invalid_images = 0
        invalid_captions = 0
        
        for path, meta in self.data.items():
            # Kiểm tra đường dẫn ảnh
            full_path = os.path.join(base_dir, path)
            is_valid_image = os.path.exists(full_path)
            
            # Kiểm tra caption
            caption = meta.get('caption', '').strip()
            is_valid_caption = bool(caption)
            
            category = meta.get('category', 'Unknown')
            self.categories.add(category)
            
            # Thêm trạng thái hợp lệ vào metadata
            meta['is_valid_image'] = is_valid_image
            meta['is_valid_caption'] = is_valid_caption
            
            if not is_valid_image:
                invalid_images += 1
            if not is_valid_caption:
                invalid_captions += 1
            
            self.items.append((path, meta))
            if len(self.items) >= max_samples:
                break
        
        self.valid_count = len(self.items) - invalid_images - invalid_captions
        self.invalid_image_count = invalid_images
        self.invalid_caption_count = invalid_captions
        
        logger.info(f"Đã tải {len(self.items)} mẫu từ {json_path}")
        logger.info(f"Ảnh không hợp lệ: {invalid_images}, Caption không hợp lệ: {invalid_captions}")
        logger.info(f"Danh sách category: {self.categories}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, meta = self.items[idx]
        caption = meta.get('caption', '')
        category = meta.get('category', 'Unknown')
        is_valid_image = meta.get('is_valid_image', False)
        is_valid_caption = meta.get('is_valid_caption', False)
        
        # Xây dựng đường dẫn đầy đủ
        full_path = os.path.join(self.base_dir, path)
        
        # Đọc và xử lý ảnh
        try:
            if is_valid_image:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image_tensor = self.transform(image)
                else:
                    image_tensor = transforms.ToTensor()(image)
            else:
                image = Image.new('RGB', (224, 224))
                image_tensor = transforms.ToTensor()(image)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {full_path}: {str(e)}")
            image = Image.new('RGB', (224, 224))
            image_tensor = transforms.ToTensor()(image)
        
        return {
            'image': image_tensor,
            'caption': caption,
            'category': category,
            'path': path,
            'is_valid_image': is_valid_image,
            'is_valid_caption': is_valid_caption
        }

def check_model_parameters(model):
    """Kiểm tra các tham số của mô hình"""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    logger.info(f"Tổng số tham số: {total_params:,}")
    logger.info(f"Tham số có thể học: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Tham số đã đóng băng: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    # Kiểm tra các tham số quan trọng
    for name, param in model.named_parameters():
        if 'logit_scale' in name:
            logger.info(f"Logit scale: {param.item()}, requires_grad: {param.requires_grad}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percent': trainable_params/total_params*100
    }

def check_optimizer(optimizer):
    """Kiểm tra optimizer và các tham số được tối ưu"""
    params_groups = len(optimizer.param_groups)
    total_params_opt = 0
    
    lr_groups = []
    
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        total_params_opt += num_params
        lr_groups.append((group['lr'], num_params))
        logger.info(f"Param group {i}: {num_params:,} tham số, lr={group['lr']}")
    
    logger.info(f"Tổng số tham số được tối ưu: {total_params_opt:,}")
    return {
        'param_groups': params_groups,
        'total_params_opt': total_params_opt,
        'lr_groups': lr_groups
    }

def visualize_batch(dataloader, dataset, num_samples=5):
    batch = next(iter(dataloader))
    images = batch['image']
    captions = batch['caption']
    categories = batch['category']
    is_valid_image = batch['is_valid_image']
    is_valid_caption = batch['is_valid_caption']
    paths = batch['path']

    logger.info(f"Batch size: {len(images)}")
    logger.info(f"Image tensor shape: {images.shape}")
    logger.info(f"Image dtype: {images.dtype}")
    logger.info(f"Image min: {images.min().item()}, max: {images.max().item()}, mean: {images.mean().item()}")
    caption_lengths = [len(cap) for cap in captions]
    logger.info(f"Caption length min: {min(caption_lengths)}, max: {max(caption_lengths)}, avg: {sum(caption_lengths)/len(caption_lengths):.1f}")

    fig, axes = plt.subplots(nrows=min(num_samples, len(images)), ncols=1, figsize=(10, 3*min(num_samples, len(images))))
    if num_samples == 1:
        axes = [axes]
    for i in range(min(num_samples, len(images))):
        # Lấy lại ảnh PIL từ dataset gốc
        idx = dataset.items.index((paths[i], next(meta for p, meta in dataset.items if p == paths[i])))
        pil_img = Image.open(os.path.join(dataset.base_dir, paths[i])).convert('RGB') if is_valid_image[i] else Image.new('RGB', (224, 224))
        axes[i].imshow(pil_img)
        status = []
        if not is_valid_image[i]:
            status.append("INVALID IMAGE")
        if not is_valid_caption[i]:
            status.append("INVALID CAPTION")
        status_str = " - ".join(status) if status else "VALID"
        axes[i].set_title(f"Caption: {captions[i][:50]}{'...' if len(captions[i])>50 else ''}\nCategory: {categories[i]}\nStatus: {status_str}")
    plt.tight_layout()
    plt.savefig('batch_samples.png')
    logger.info(f"Đã lưu hình ảnh mẫu batch tại 'batch_samples.png'")

def check_logit_scale(model):
    """Kiểm tra logit scale"""
    if hasattr(model, 'clip_model') and hasattr(model.clip_model, 'logit_scale'):
        logit_scale = model.clip_model.logit_scale
        logger.info(f"Logit scale value: {logit_scale.item()}")
        logger.info(f"Logit scale exp value: {logit_scale.exp().item()}")
        logger.info(f"Logit scale requires_grad: {logit_scale.requires_grad}")
        return {
            'value': logit_scale.item(),
            'exp_value': logit_scale.exp().item(),
            'requires_grad': logit_scale.requires_grad
        }
    elif hasattr(model, 'logit_scale'):
        logit_scale = model.logit_scale
        logger.info(f"Logit scale value: {logit_scale.item()}")
        logger.info(f"Logit scale exp value: {logit_scale.exp().item()}")
        logger.info(f"Logit scale requires_grad: {logit_scale.requires_grad}")
        return {
            'value': logit_scale.item(),
            'exp_value': logit_scale.exp().item(),
            'requires_grad': logit_scale.requires_grad
        }
    else:
        logger.warning("Không tìm thấy logit_scale trong model!")
        return None

def check_embeddings_normalization(model, dataloader, device):
    """Kiểm tra việc chuẩn hóa embeddings"""
    model.eval()
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    captions = batch['caption']
    
    with torch.no_grad():
        # Tokenize captions
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
        
        # Get embeddings
        if hasattr(model, 'clip_model'):
            # Using custom model
            image_features = model.clip_model.encode_image(images)
            text_features = model.clip_model.encode_text(text_tokens)
        else:
            # Using CLIP model directly
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
    
    # Check normalization
    img_norms = torch.norm(image_features, dim=1)
    txt_norms = torch.norm(text_features, dim=1)
    
    logger.info(f"Image embedding norms: min={img_norms.min().item():.4f}, max={img_norms.max().item():.4f}, mean={img_norms.mean().item():.4f}")
    logger.info(f"Text embedding norms: min={txt_norms.min().item():.4f}, max={txt_norms.max().item():.4f}, mean={txt_norms.mean().item():.4f}")
    
    # Check if close to 1 (normalized)
    img_normalized = (img_norms - 1.0).abs().mean().item()
    txt_normalized = (txt_norms - 1.0).abs().mean().item()
    
    logger.info(f"Image embeddings normalized? {img_normalized < 0.01} (diff from 1.0: {img_normalized:.4f})")
    logger.info(f"Text embeddings normalized? {txt_normalized < 0.01} (diff from 1.0: {txt_normalized:.4f})")
    
    return {
        'img_norms': {
            'min': img_norms.min().item(),
            'max': img_norms.max().item(),
            'mean': img_norms.mean().item(),
            'normalized': img_normalized < 0.01
        },
        'txt_norms': {
            'min': txt_norms.min().item(),
            'max': txt_norms.max().item(),
            'mean': txt_norms.mean().item(),
            'normalized': txt_normalized < 0.01
        }
    }

def compute_loss_statistics(model, dataloader, device):
    """Tính toán thống kê về loss"""
    model.eval()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Tokenize captions
            text_tokens = clip.tokenize(captions, truncate=True).to(device)
            
            # Get logits
            if hasattr(model, 'clip_model'):
                # Custom model with clip_model attribute
                image_features = model.clip_model.encode_image(images)
                text_features = model.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate logits
                logit_scale = model.clip_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
            else:
                # Standard CLIP model or custom model with different structure
                try:
                    # Try using the forward method
                    logits_per_image, logits_per_text = model(images, text_tokens)
                except:
                    logger.error("Không thể tính toán logits từ model. Kiểm tra cấu trúc model!")
                    return None
            
            # Ground truth should be diagonal
            batch_size = images.shape[0]
            ground_truth = torch.arange(batch_size, device=device)
            
            # Calculate loss
            loss_i = loss_img(logits_per_image, ground_truth)
            loss_t = loss_txt(logits_per_text, ground_truth)
            batch_loss = (loss_i + loss_t) / 2
            
            losses.append(batch_loss.item())
    
    if not losses:
        logger.warning("Không có batch nào để tính loss!")
        return None
    
    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)
    
    logger.info(f"Loss statistics: avg={avg_loss:.4f}, min={min_loss:.4f}, max={max_loss:.4f}")
    logger.info(f"Theoretical random loss with batch size {batch_size}: {np.log(batch_size):.4f}")
    
    return {
        'avg_loss': avg_loss,
        'min_loss': min_loss,
        'max_loss': max_loss,
        'theoretical_random': np.log(batch_size)
    }

def check_dtype_consistency(model):
    """Kiểm tra tính nhất quán của kiểu dữ liệu trong model"""
    dtypes = {}
    
    for name, param in model.named_parameters():
        dtype = param.dtype
        if dtype not in dtypes:
            dtypes[dtype] = []
        dtypes[dtype].append(name)
    
    for dtype, params in dtypes.items():
        logger.info(f"Dtype {dtype}: {len(params)} parameters")
        # Print some examples
        examples = params[:3] + ([] if len(params) <= 3 else ['...'])
        logger.info(f"Examples: {', '.join(examples)}")
    
    # Check for mixed precision
    has_mixed_precision = len(dtypes) > 1
    logger.info(f"Model has mixed precision: {has_mixed_precision}")
    
    return {
        'dtypes': {str(dtype): len(params) for dtype, params in dtypes.items()},
        'has_mixed_precision': has_mixed_precision
    }

def test_batch_size_compatibility(model, device, batch_sizes=[1, 8, 16, 32]):
    """Kiểm tra khả năng tương thích của các batch size khác nhau"""
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        # Create dummy inputs
        dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_texts = ["Test caption"] * batch_size
        dummy_tokens = clip.tokenize(dummy_texts, truncate=True).to(device)
        
        try:
            with torch.no_grad():
                if hasattr(model, 'clip_model'):
                    # Using custom model with clip_model
                    image_features = model.clip_model.encode_image(dummy_images)
                    text_features = model.clip_model.encode_text(dummy_tokens)
                    
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Logits
                    logit_scale = model.clip_model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    
                    # Ground truth
                    ground_truth = torch.arange(batch_size, device=device)
                    
                    # Loss
                    loss = nn.CrossEntropyLoss()(logits_per_image, ground_truth)
                else:
                    # Try direct forward
                    logits_per_image, logits_per_text = model(dummy_images, dummy_tokens)
                    ground_truth = torch.arange(batch_size, device=device)
                    loss = nn.CrossEntropyLoss()(logits_per_image, ground_truth)
            
            results[batch_size] = {
                'status': 'OK',
                'loss': loss.item()
            }
            logger.info(f"Batch size {batch_size}: OK, loss = {loss.item():.4f}")
            
        except Exception as e:
            results[batch_size] = {
                'status': 'ERROR',
                'error': str(e)
            }
            logger.error(f"Batch size {batch_size}: ERROR - {str(e)}")
    
    return results

def main():
    # Thiết lập các thông số
    CONFIG = {
        'clip_model_name': 'ViT-B/32',
        'batch_size': 32,
        'num_workers': 4,
        'data_dir': "E:\\Đồ án chuyên ngành\\dataset\\Data_Merge_2",
        'json_paths': [
            "E:\\Đồ án chuyên ngành\\dataset\\script_data_merge_2\\caption_Violence_train_dir.json",
            "E:\\Đồ án chuyên ngành\\dataset\\script_data_merge_2\\caption_Sensitive_train_dir.json"
        ],
        'save_dir': 'E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\clip_diagnostics',
    }
    
    # Tạo thư mục lưu
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Thiết lập device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Sử dụng device: {device}")
    
    # Tải model CLIP
    clip_model, preprocess = clip.load(CONFIG['clip_model_name'], device=device, jit=False)
    logger.info(f"Đã tải model CLIP {CONFIG['clip_model_name']}")
    
    # Kiểm tra loại dữ liệu đầu vào và model
    check_dtype_consistency(clip_model)
    
    # Kiểm tra logit scale
    check_logit_scale(clip_model)
    
    # Kiểm tra tham số model
    check_model_parameters(clip_model)
    
    # Tạo diagnostic dataset
    all_datasets = []
    for json_path in CONFIG['json_paths']:
        try:
            dataset = DiagnosticDataset(json_path, CONFIG['data_dir'], preprocess, max_samples=50)
            all_datasets.append(dataset)
            logger.info(f"Thống kê dataset từ {json_path}:")
            logger.info(f"- Mẫu hợp lệ: {dataset.valid_count}/{len(dataset)}")
            logger.info(f"- Tỷ lệ mẫu hợp lệ: {dataset.valid_count/len(dataset)*100:.1f}%")
        except Exception as e:
            logger.error(f"Lỗi khi tạo dataset từ {json_path}: {str(e)}")
    
    if not all_datasets:
        logger.error("Không thể tạo bất kỳ dataset nào! Kết thúc chẩn đoán.")
        return
    
    # Tạo dataloader
    dataloader = DataLoader(
        all_datasets[0],  # Chỉ dùng dataset đầu tiên để chẩn đoán
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        drop_last=True  # Kiểm tra với drop_last=True
    )
    
    # Hiển thị một số mẫu từ batch
    visualize_batch(dataloader, all_datasets[0], num_samples=5)
    
    # Kiểm tra chuẩn hóa embeddings
    embedding_check = check_embeddings_normalization(clip_model, dataloader, device)
    
    # Tính toán thống kê về loss
    loss_stats = compute_loss_statistics(clip_model, dataloader, device)
    
    # Kiểm tra khả năng tương thích của các batch size khác nhau
    batch_compatibility = test_batch_size_compatibility(clip_model, device)
    
    # Tạo báo cáo chẩn đoán
    diagnostics = {
        'model_info': {
            'name': CONFIG['clip_model_name'],
            'dtype_consistency': check_dtype_consistency(clip_model),
            'logit_scale': check_logit_scale(clip_model),
            'parameters': check_model_parameters(clip_model)
        },
        'data_info': {
            'valid_samples': dataset.valid_count,
            'total_samples': len(dataset),
            'valid_ratio': dataset.valid_count/len(dataset),
            'invalid_images': dataset.invalid_image_count,
            'invalid_captions': dataset.invalid_caption_count,
            'categories': list(dataset.categories)
        },
        'embedding_check': embedding_check,
        'loss_stats': loss_stats,
        'batch_compatibility': batch_compatibility
    }
    
    # Lưu kết quả chẩn đoán
    with open(os.path.join(CONFIG['save_dir'], 'diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    logger.info("Đã hoàn thành chẩn đoán pipeline CLIP")
    logger.info(f"Kết quả được lưu tại: {os.path.join(CONFIG['save_dir'], 'diagnostics.json')}")
    
    # Tóm tắt vấn đề tìm thấy
    issues = []
    
    # Kiểm tra vấn đề dữ liệu
    if dataset.invalid_image_count > 0 or dataset.invalid_caption_count > 0:
        issues.append(f"Dữ liệu có vấn đề: {dataset.invalid_image_count} ảnh không hợp lệ, {dataset.invalid_caption_count} caption không hợp lệ")
    
    # Kiểm tra vấn đề chuẩn hóa embedding
    if not embedding_check['img_norms']['normalized']:
        issues.append("Image embeddings không được chuẩn hóa đúng")
    if not embedding_check['txt_norms']['normalized']:
        issues.append("Text embeddings không được chuẩn hóa đúng")
    
    # Kiểm tra vấn đề logit scale
    logit_scale_info = check_logit_scale(clip_model)
    if logit_scale_info and not logit_scale_info['requires_grad']:
        issues.append("Logit scale không có requires_grad=True, không thể học được")
    
    # Kiểm tra vấn đề kiểu dữ liệu
    dtype_info = check_dtype_consistency(clip_model)
    if dtype_info['has_mixed_precision']:
        issues.append("Model có mixed precision, có thể gây vấn đề về dtype không đồng nhất")
    
    # Kiểm tra vấn đề loss
    if loss_stats and loss_stats['avg_loss'] >= loss_stats['theoretical_random'] * 0.95:
        issues.append(f"Loss ({loss_stats['avg_loss']:.4f}) gần với giá trị random ({loss_stats['theoretical_random']:.4f}), model có thể không học được gì")
    
    # In tóm tắt vấn đề
    if issues:
        logger.warning("Các vấn đề được phát hiện:")
        for i, issue in enumerate(issues, 1):
            logger.warning(f"{i}. {issue}")
    else:
        logger.info("Không phát hiện vấn đề nghiêm trọng nào trong pipeline.")

if __name__ == "__main__":
    main() 