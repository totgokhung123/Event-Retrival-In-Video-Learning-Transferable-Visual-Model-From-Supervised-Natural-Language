import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import clip
from tqdm import tqdm
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import time
import datetime

# Suppress some common warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*The PyTorch API of nested tensors is in prototype stage.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Đặt seed cho tất cả các thư viện ngẫu nhiên để đảm bảo tính lặp lại."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đảm bảo tính xác định cho CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Đã đặt seed = {seed} cho tính lặp lại cao.")

class FrameCaptionDataset(Dataset):
    def __init__(self, json_path, transform=None, label_map=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.items = list(self.data.items())
        self.transform = transform
        self.label_map = label_map or {"Violence": 1, "NonViolence": 0}
        
        # Kiểm tra tính hợp lệ của đường dẫn
        self.base_dir = os.path.dirname(json_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        frame_path, meta = self.items[idx]
        caption = meta['caption']
        category = meta['category']  # Giữ lại category cho việc đánh giá
        label = self.label_map.get(category, 0)
        
        # Kiểm tra đường dẫn tuyệt đối hoặc tương đối
        if not os.path.isabs(frame_path):
            frame_path = os.path.join(self.base_dir, frame_path)
            
        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Lỗi đọc ảnh {frame_path}: {str(e)}")
            image = torch.randn(3, 224, 224)  # fallback
            
        return {
            "image": image, 
            "caption": caption, 
            "category": category,
            "label": label,
            "frame_path": frame_path
        }

def patch_clip_for_fp32(clip_model):
    """
    Patch CLIP model to ensure all outputs are float32 instead of float16.
    This should solve the 'Attempting to unscale FP16 gradients' error.
    """
    original_encode_image = clip_model.encode_image
    original_encode_text = clip_model.encode_text
    
    def wrapped_encode_image(image):
        features = original_encode_image(image)
        if features.dtype == torch.float16:
            features = features.to(torch.float32)
        return features
    
    def wrapped_encode_text(text):
        features = original_encode_text(text)
        if features.dtype == torch.float16:
            features = features.to(torch.float32)
        return features
    
    clip_model.encode_image = wrapped_encode_image
    clip_model.encode_text = wrapped_encode_text
    
    logger.info("Patched CLIP model to ensure float32 outputs")
    return clip_model

class CLIPMultiModalModel(nn.Module):
    def __init__(self, pretrained="ViT-B/32", embed_dim=256, freeze_clip=True, use_gradient_checkpointing=True):
        super(CLIPMultiModalModel, self).__init__()
        self.clip_model, _ = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Patch CLIP model để đảm bảo đầu ra là float32
        self.clip_model = patch_clip_for_fp32(self.clip_model)
        
        self.embedding_dim = self.clip_model.visual.output_dim
        
        # Đóng băng hoặc fine-tune CLIP
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            logger.info("Fine-tuning CLIP backbone!")
            # Để cho phép fine-tune, đảm bảo requires_grad=True và dữ liệu ở định dạng float32
            for param in self.clip_model.parameters():
                param.requires_grad = True
                # Đảm bảo params dạng float32
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
            
            # Bật gradient checkpointing để tiết kiệm bộ nhớ (chỉ khi fine-tune)
            if use_gradient_checkpointing and hasattr(self.clip_model.visual, 'transformer') and hasattr(self.clip_model.visual.transformer, 'resblocks'):
                logger.info("Bật gradient checkpointing cho CLIP ViT model để tiết kiệm bộ nhớ")
                for r in self.clip_model.visual.transformer.resblocks:
                    if hasattr(r, 'enable_gradient_checkpointing'):
                        r.enable_gradient_checkpointing = True
                    elif hasattr(r, 'checkpoint'):
                        r.checkpoint = True
        
        # Projection layers (có thể fine-tune ngay cả khi CLIP bị đóng băng)
        if embed_dim > 0:
            self.image_projection = nn.Linear(self.embedding_dim, embed_dim, dtype=torch.float32)
            self.text_projection = nn.Linear(self.embedding_dim, embed_dim, dtype=torch.float32)
            
            # Khởi tạo các lớp projection để tránh vanishing gradient
            nn.init.xavier_uniform_(self.image_projection.weight)
            nn.init.xavier_uniform_(self.text_projection.weight)
            nn.init.constant_(self.image_projection.bias, 0)
            nn.init.constant_(self.text_projection.bias, 0)
            
            # Đảm bảo các lớp projection luôn được train
            self.image_projection.requires_grad_(True)
            self.text_projection.requires_grad_(True)
        else:
            self.image_projection = nn.Identity()
            self.text_projection = nn.Identity()
        
        # Temperature parameter (học được trong quá trình training)
        self.logit_scale = nn.Parameter(torch.ones([], dtype=torch.float32) * np.log(1 / 0.07))
        
        # Final embedding dimension
        self.final_dim = embed_dim if embed_dim > 0 else self.embedding_dim

    def encode_image(self, image_batch):
        with torch.no_grad() if all(p.requires_grad == False for p in self.clip_model.parameters()) else torch.enable_grad():
            image_features = self.clip_model.encode_image(image_batch)
            # Đảm bảo features là float32 để tránh vấn đề với gradient fp16
            if image_features.dtype == torch.float16:
                image_features = image_features.to(torch.float32)
        return image_features
    
    def encode_text(self, text_batch):
        with torch.no_grad() if all(p.requires_grad == False for p in self.clip_model.parameters()) else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_batch)
            # Đảm bảo features là float32 để tránh vấn đề với gradient fp16
            if text_features.dtype == torch.float16:
                text_features = text_features.to(torch.float32)
        return text_features
    
    def forward(self, image_batch=None, text_batch=None):
        results = {}
        
        # Encode image if provided
        if image_batch is not None:
            # Handle potential fp16/mixed precision
            image_batch_dtype = image_batch.dtype
            image_features = self.encode_image(image_batch)
            
            # Convert to model dtype
            if hasattr(self.image_projection, 'weight'):
                target_dtype = self.image_projection.weight.dtype
            else:
                target_dtype = torch.float32
            
            # Ensure features match projection layer dtype
            if image_features.dtype != target_dtype:
                image_features = image_features.to(target_dtype)
            
            image_embed = self.image_projection(image_features)
            # Normalize embeddings
            image_embed = image_embed / (image_embed.norm(dim=-1, keepdim=True) + 1e-8)
            results["image_features"] = image_features
            results["image_embed"] = image_embed
        
        # Encode text if provided
        if text_batch is not None:
            # Handle potential fp16/mixed precision
            text_features = self.encode_text(text_batch)
            
            # Convert to model dtype
            if hasattr(self.text_projection, 'weight'):
                target_dtype = self.text_projection.weight.dtype
            else:
                target_dtype = torch.float32
            
            # Ensure features match projection layer dtype
            if text_features.dtype != target_dtype:
                text_features = text_features.to(target_dtype)
            
            text_embed = self.text_projection(text_features)
            # Normalize embeddings
            text_embed = text_embed / (text_embed.norm(dim=-1, keepdim=True) + 1e-8)
            results["text_features"] = text_features
            results["text_embed"] = text_embed
        
        # Compute similarity if both image and text are provided
        if image_batch is not None and text_batch is not None:
            # Ensure both embeddings use the same dtype for dot product
            common_dtype = image_embed.dtype
            if text_embed.dtype != common_dtype:
                text_embed = text_embed.to(common_dtype)
            
            # Scale dot-product by learned temperature
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * torch.matmul(image_embed, text_embed.transpose(0, 1))
            logits_per_text = logits_per_image.t()
            
            results["logits_per_image"] = logits_per_image  # [batch_size, batch_size]
            results["logits_per_text"] = logits_per_text    # [batch_size, batch_size]
        
        return results

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Compute contrastive loss with improved numerical stability.
    This implementation handles potential numerical issues when using mixed precision.
    """
    # Check for invalid values
    if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
        # Return a valid but high loss to signal an issue
        return torch.tensor(10.0, device=logits_per_image.device, dtype=logits_per_image.dtype)
        
    # Ground-truth labels = diagonal (identity matrix)
    batch_size = logits_per_image.size(0)
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)
    
    # Apply temperature scaling if needed
    if logits_per_image.max() > 100:
        logger.warning(f"Numerical instability detected: max logit value is {logits_per_image.max().item()}")
        scaling_factor = 100.0 / logits_per_image.max().item()
        logits_per_image = logits_per_image * scaling_factor
        logits_per_text = logits_per_text * scaling_factor
    
    # Compute cross-entropy loss in both directions with improved stability
    loss_i2t = nn.functional.cross_entropy(logits_per_image, ground_truth)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, ground_truth)
    
    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss

def validate(model, dataloader, device, memory_efficient=True):
    model.eval()
    running_loss = 0.0
    
    # Metrics for retrieval
    img2txt_ranks = []
    txt2img_ranks = []
    
    # For category evaluation
    all_categories = []
    all_predictions = []
    
    # Số lượng batch để tính toán metrics
    max_batches = None if not memory_efficient else min(50, len(dataloader))
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), desc="Validation", total=len(dataloader) if max_batches is None else max_batches)
        for batch_idx, batch in pbar:
            # Nếu memory_efficient, chỉ đánh giá một số batch nhất định
            if memory_efficient and max_batches is not None and batch_idx >= max_batches:
                break
                
            # Giải phóng bộ nhớ cache định kỳ
            if memory_efficient and batch_idx > 0 and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
            images = batch["image"].to(device)
            categories = batch["category"]
            all_categories.extend(categories)
            
            # Tokenize captions
            text_tokens = clip.tokenize(batch["caption"]).to(device)
            
            # Đảm bảo dtype nhất quán ngay cả khi dùng amp
            with autocast(enabled=False):
                outputs = model(images, text_tokens)
                loss = contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
            
            # Update loss metric
            running_loss += loss.item()
            
            # Calculate retrieval ranks
            logits_per_image = outputs["logits_per_image"].cpu()
            logits_per_text = outputs["logits_per_text"].cpu()
            
            # Xóa outputs để giải phóng bộ nhớ
            del outputs
            
            # For each image, find rank of matching text
            img2txt_sorted = torch.argsort(logits_per_image, dim=1, descending=True)
            for i, sorted_indices in enumerate(img2txt_sorted):
                rank = torch.where(sorted_indices == i)[0].item()
                img2txt_ranks.append(rank)
            
            # For each text, find rank of matching image
            txt2img_sorted = torch.argsort(logits_per_text, dim=1, descending=True)
            for i, sorted_indices in enumerate(txt2img_sorted):
                rank = torch.where(sorted_indices == i)[0].item()
                txt2img_ranks.append(rank)
            
            # For category prediction
            for i, row in enumerate(logits_per_image):
                # Lấy các caption liên quan nhất cho mỗi ảnh
                top_caption_idx = torch.argmax(row).item()
                predicted_category = batch["category"][top_caption_idx]
                all_predictions.append(predicted_category)
            
            # Xóa các biến tạm thời để giải phóng bộ nhớ
            del img2txt_sorted, txt2img_sorted, logits_per_image, logits_per_text
            
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    # Calculate R@K metrics
    r1_img2txt = 100.0 * sum(r == 0 for r in img2txt_ranks) / len(img2txt_ranks)
    r5_img2txt = 100.0 * sum(r < 5 for r in img2txt_ranks) / len(img2txt_ranks)
    r10_img2txt = 100.0 * sum(r < 10 for r in img2txt_ranks) / len(img2txt_ranks)
    
    r1_txt2img = 100.0 * sum(r == 0 for r in txt2img_ranks) / len(txt2img_ranks)
    r5_txt2img = 100.0 * sum(r < 5 for r in txt2img_ranks) / len(txt2img_ranks)
    r10_txt2img = 100.0 * sum(r < 10 for r in txt2img_ranks) / len(txt2img_ranks)
    
    # Calculate mean rank
    mr_img2txt = sum(img2txt_ranks) / len(img2txt_ranks)
    mr_txt2img = sum(txt2img_ranks) / len(txt2img_ranks)
    
    # Calculate category accuracy
    category_acc = 100.0 * sum(p == t for p, t in zip(all_predictions, all_categories)) / len(all_categories)
    
    metrics = {
        'loss': running_loss / (len(dataloader) if max_batches is None else max_batches),
        'r1_img2txt': r1_img2txt,
        'r5_img2txt': r5_img2txt,
        'r10_img2txt': r10_img2txt,
        'r1_txt2img': r1_txt2img,
        'r5_txt2img': r5_txt2img,
        'r10_txt2img': r10_txt2img,
        'mr_img2txt': mr_img2txt,
        'mr_txt2img': mr_txt2img,
        'category_acc': category_acc,
        'eval_batches': len(dataloader) if max_batches is None else max_batches
    }
    
    # Giải phóng bộ nhớ sau khi validate
    if memory_efficient and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics

def train_one_epoch(model, dataloader, optimizer, scaler, device, grad_accumulation_steps=1, memory_efficient=True):
    model.train()
    running_loss = 0.0
    
    # Check if any parameters require gradients
    has_trainable_params = any(p.requires_grad for p in model.parameters())
    if not has_trainable_params:
        logger.warning("Không có tham số nào cần cập nhật gradients! Kiểm tra freeze_clip và các tham số mô hình.")
    
    # Biến để theo dõi xem có cần tắt AMP không
    should_disable_amp = False
    
    pbar = tqdm(enumerate(dataloader), desc="Training", total=len(dataloader))
    accumulated_batches = 0
    
    for batch_idx, batch in pbar:
        # Giải phóng bộ nhớ cache định kỳ
        if memory_efficient and batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            
        images = batch["image"].to(device)
        
        # Tokenize captions
        with torch.no_grad():
            text_tokens = clip.tokenize(batch["caption"]).to(device)
        
        # Chỉ xóa gradients khi bắt đầu chu kỳ tích lũy mới
        if accumulated_batches == 0:
            optimizer.zero_grad()
        
        # Kiểm tra xem có sử dụng mixed precision không
        use_mixed_precision = scaler._enabled
        
        # Nếu không sử dụng mixed precision, đảm bảo tất cả là float32
        if not use_mixed_precision:
            images = images.to(torch.float32)
            
        with autocast(enabled=use_mixed_precision):
            outputs = model(images, text_tokens)
            loss = contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
            
            # Xóa outputs để giải phóng bộ nhớ
            del outputs
            
            # Chia loss theo số bước tích lũy gradient
            loss = loss / grad_accumulation_steps
            
            # Kiểm tra nếu loss là NaN hoặc infinity
            if not torch.isfinite(loss):
                logger.warning(f"Loss không hợp lệ (NaN/infinity): {loss.item()}. Bỏ qua batch này.")
                continue
        
        # Scale loss và backprop
        scaler.scale(loss).backward()
        
        # Update metrics ngay cả khi chưa thực hiện optimizer step
        running_loss += loss.item() * grad_accumulation_steps
        
        # Tăng counter cho batches đã tích lũy
        accumulated_batches += 1
        
        # Chỉ optimizer step sau mỗi grad_accumulation_steps batch
        if accumulated_batches == grad_accumulation_steps or batch_idx == len(dataloader) - 1:
            # Đảm bảo tất cả các gradients đều là float32 trước khi unscale
            for param in model.parameters():
                if param.requires_grad and param.grad is not None and param.grad.dtype == torch.float16:
                    param.grad = param.grad.to(torch.float32)
            
            # Kiểm tra tất cả các gradients
            has_fp16_grads = False
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad and p.grad is not None:
                        if p.grad.dtype == torch.float16:
                            has_fp16_grads = True
                            logger.warning(f"Phát hiện gradient FP16 trong tham số shape={p.shape}")
            
            try:
                if has_fp16_grads:
                    logger.warning("Vẫn tồn tại gradients FP16. Bỏ qua unscale để tránh lỗi.")
                    # Bỏ qua unscale nhưng vẫn step (sẽ không có gradient clipping)
                    scaler.step(optimizer)
                    scaler.update()
                    # Flag để tắt AMP nếu gặp lỗi liên tục
                    should_disable_amp = True
                else:
                    # Unscale gradients để clip
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients để tránh exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Thực hiện optimizer step và update scaler
                    scaler.step(optimizer)
                    scaler.update()
            except ValueError as e:
                if "Attempting to unscale FP16 gradients" in str(e):
                    logger.warning("Bắt gặp lỗi 'Attempting to unscale FP16 gradients'. Đang xử lý...")
                    # Bỏ qua lỗi, xóa gradients và tắt AMP
                    optimizer.zero_grad()
                    should_disable_amp = True
                else:
                    raise e
            
            # Reset counter
            accumulated_batches = 0
            
            # Giải phóng bộ nhớ sau mỗi lần cập nhật
            if memory_efficient:
                torch.cuda.empty_cache()
        
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    # Return thêm trạng thái nên tắt AMP không
    return {'loss': running_loss / len(dataloader), 'disable_amp': should_disable_amp}

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """Lưu checkpoint model và optimizer."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Đã lưu checkpoint tại: {save_path}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
        torch.save(model.state_dict(), best_path)
        logger.info(f"Đã lưu best model với R@1: {metrics['r1_img2txt']:.2f}% tại: {best_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint để tiếp tục train."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu train từ đầu.")
        return 0, {}
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Đã tải checkpoint từ epoch {epoch} với R@1: {metrics.get('r1_img2txt', 0):.2f}%")
    return epoch + 1, metrics

def main():
    # Khai báo cứng các tham số
    json_path = "./Filtered_Frames_Script_sorted.json"  # Đường dẫn tới file JSON
    batch_size = 32  # Giảm batch size từ 128 xuống 32 để tiết kiệm bộ nhớ
    epochs = 30  # Train lâu hơn để học được embedding tốt
    lr = 2e-5  # Learning rate
    save_dir = 'clip_multimodal_retrieval'  # Thư mục lưu model
    freeze_clip = True  # Đóng băng CLIP backbone để tiết kiệm bộ nhớ
    embed_dim = 256  # Dimension của embedding space
    seed = 42  # Random seed
    resume = ''  # Đường dẫn checkpoint để tiếp tục train
    use_amp = True  # Sử dụng Automatic Mixed Precision (AMP)
    grad_accumulation_steps = 4  # Tăng từ 1 lên 4 để có hiệu quả batch size lớn mà vẫn tiết kiệm bộ nhớ
    
    # Tùy chọn gỡ lỗi
    debug_mode = False  # Bật để ghi log thêm thông tin
    torch_debug = False  # Bật để bắt lỗi trong pytorch anomaly detection
    force_fp32 = True  # Buộc sử dụng float32 cho mọi tính toán
    memory_efficient = True  # Bật các tối ưu hóa bộ nhớ
    
    # Bật các tính năng gỡ lỗi của PyTorch nếu cần
    if torch_debug:
        logger.info("Đã bật PyTorch anomaly detection - có thể làm chậm quá trình train")
        torch.autograd.set_detect_anomaly(True)
    
    # Nếu gặp lỗi liên tục với mixed precision hoặc buộc dùng fp32, tắt AMP
    if debug_mode or force_fp32:
        use_amp = False
        logger.info("Debug mode/Force FP32: Đã tắt mixed precision để tránh lỗi unscale gradients")

    # Set seed
    set_seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Kiểm tra cuda
    if torch.cuda.is_available():
        logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        if force_fp32:
            logger.info("Đã tắt AMP vì force_fp32=True")
            use_amp = False
        else:
            # Kiểm tra xem GPU có hỗ trợ mixed precision hay không
            if torch.cuda.get_device_capability()[0] >= 7:
                logger.info("GPU hỗ trợ mixed precision training (Tensor Cores)")
            else:
                logger.warning("GPU có thể không hỗ trợ tối ưu Tensor Cores cho mixed precision")
                use_amp = False
                logger.warning("Đã tắt AMP do GPU không hỗ trợ tối ưu")
    else:
        logger.warning("CUDA không khả dụng, sử dụng CPU")
        use_amp = False
    
    # Kiểm tra xem PyTorch được biên dịch với CUDA hay không
    if torch.cuda.is_available() and not torch.cuda.has_half:
        logger.warning("PyTorch không hỗ trợ half precision (FP16) trên GPU này")
        use_amp = False
    
    # Logging mixed precision settings
    logger.info(f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    logger.info(f"Force FP32 mode: {'Enabled' if force_fp32 else 'Disabled'}")
    
    # Đặt default tensor type nếu cần
    if force_fp32:
        torch.set_default_tensor_type(torch.FloatTensor)
        logger.info("Đã đặt default tensor type là FloatTensor")
    
    # Tối ưu hóa bộ nhớ CUDA
    if memory_efficient and torch.cuda.is_available():
        # Giải phóng bộ nhớ cache
        torch.cuda.empty_cache()
        
        # Thiết lập các thông số phân bổ CUDA để tránh phân mảnh
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Đặt max_split_size để tránh phân mảnh
        torch.cuda.set_per_process_memory_fraction(0.8)  # Chỉ sử dụng 80% bộ nhớ GPU
        logger.info("Đã áp dụng các tối ưu hóa bộ nhớ CUDA")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    full_dataset = FrameCaptionDataset(json_path, transform=train_transform)
    logger.info(f"Loaded dataset with {len(full_dataset)} samples")
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    val_dataset.dataset.transform = val_transform

    # Worker init function
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Data loaders - giảm num_workers nếu thiếu bộ nhớ
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Giảm từ 4 xuống 2
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # Giảm từ 4 xuống 2
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    # Model
    model = CLIPMultiModalModel(
        pretrained="ViT-B/32", 
        embed_dim=embed_dim,
        freeze_clip=freeze_clip,
        use_gradient_checkpointing=memory_efficient and not freeze_clip
    ).to(device)
    
    # Kiểm tra xem có tham số nào được train không
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning("Mô hình không có tham số nào được train! Hãy đặt freeze_clip=False hoặc kiểm tra lại.")
    else:
        logger.info(f"Tổng số tham số được train: {sum(p.numel() for p in trainable_params):,}")
    
    # Optimizer - Weight decay cho regularization
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training
    scaler = GradScaler(enabled=use_amp)

    # Resume training if needed
    start_epoch = 0
    best_metrics = {'r1_img2txt': 0.0}
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if resume:
        start_epoch, loaded_metrics = load_checkpoint(model, optimizer, resume, device)
        if 'r1_img2txt' in loaded_metrics:
            best_metrics['r1_img2txt'] = loaded_metrics['r1_img2txt']

    # Save training config
    config = {
        'json_path': json_path,
        'batch_size': batch_size,
        'grad_accumulation_steps': grad_accumulation_steps,
        'epochs': epochs,
        'lr': lr,
        'save_dir': save_dir,
        'freeze_clip': freeze_clip,
        'embed_dim': embed_dim,
        'seed': seed,
        'resume': resume,
        'start_epoch': start_epoch,
        'use_amp': use_amp,
        'debug_mode': debug_mode,
        'weight_decay': 0.01,
        'device': device.type,
        'date_time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Training loop
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Giải phóng bộ nhớ trước khi bắt đầu epoch mới
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, grad_accumulation_steps, memory_efficient)
        
        # Kiểm tra xem có cần tắt AMP không
        if train_metrics.get('disable_amp', False) and scaler._enabled:
            logger.warning("Đang tắt Automatic Mixed Precision (AMP) do phát hiện lỗi FP16 gradients")
            # Tắt AMP
            scaler = GradScaler(enabled=False)
            # Đảm bảo mô hình là float32
            for param in model.parameters():
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
            logger.info("Đã chuyển đổi model sang float32 và tắt AMP")
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        val_metrics = validate(model, val_loader, device, memory_efficient)
        
        # Log metrics
        logger.info(f"Train loss: {train_metrics['loss']:.4f}")
        logger.info(
            f"Val loss: {val_metrics['loss']:.4f}, "
            f"Image→Text: R@1={val_metrics['r1_img2txt']:.2f}%, "
            f"R@5={val_metrics['r5_img2txt']:.2f}%, "
            f"R@10={val_metrics['r10_img2txt']:.2f}%, "
            f"MeanRank={val_metrics['mr_img2txt']:.2f}"
        )
        logger.info(
            f"Text→Image: R@1={val_metrics['r1_txt2img']:.2f}%, "
            f"R@5={val_metrics['r5_txt2img']:.2f}%, "
            f"R@10={val_metrics['r10_txt2img']:.2f}%, "
            f"MeanRank={val_metrics['mr_txt2img']:.2f}, "
            f"Category Acc={val_metrics['category_acc']:.2f}%"
        )

        # Kiểm tra xem có phải best model không
        is_best = val_metrics['r1_img2txt'] > best_metrics.get('r1_img2txt', 0)
        if is_best:
            best_metrics = val_metrics
            # Lưu best model
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"Đã lưu best model với R@1: {val_metrics['r1_img2txt']:.2f}% tại: {best_path}")
        
        # Luôn lưu checkpoint vào cùng một file sau mỗi epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'best_metrics': best_metrics,
            'lr_scheduler': scheduler.state_dict()
        }, checkpoint_path)
        logger.info(f"Đã lưu checkpoint cho epoch {epoch+1} tại: {checkpoint_path}")

    # Kết quả cuối cùng
    logger.info(f"Training completed. Best R@1: {best_metrics['r1_img2txt']:.2f}%")

    # Lưu embedding của tất cả frame vào file để triển khai retrieval
    if os.path.exists(os.path.join(save_dir, "best_model.pt")):
        logger.info("Generating embeddings for all frames...")
        model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
        
        # Dataset cho tất cả frames
        all_dataset = FrameCaptionDataset(json_path, transform=val_transform)
        
        # Sử dụng batch size nhỏ hơn để tránh OOM
        embed_batch_size = 16
        logger.info(f"Trích xuất embedding với batch size: {embed_batch_size}")
        
        all_loader = DataLoader(all_dataset, batch_size=embed_batch_size, shuffle=False, num_workers=2)
        
        # Xử lý theo phần để tiết kiệm bộ nhớ
        all_frame_paths = []
        all_captions = []
        all_categories = []
        
        # File để lưu embedding theo phần
        embedding_dir = os.path.join(save_dir, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            chunk_idx = 0
            chunk_size = 1000
            current_embeddings = []
            current_frames = []
            current_captions = []
            current_categories = []
            
            for batch_idx, batch in enumerate(tqdm(all_loader, desc="Extracting embeddings")):
                # Giải phóng bộ nhớ cache định kỳ
                if memory_efficient and batch_idx > 0 and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
                images = batch["image"].to(device)
                
                # Đảm bảo độ chính xác float32 cho inference
                with autocast(enabled=False):
                    results = model(images)
                
                if "image_embed" in results:
                    embeddings = results["image_embed"].cpu().numpy()
                    frame_paths = batch["frame_path"]
                    captions = batch["caption"]
                    categories = batch["category"]
                    
                    # Thêm vào danh sách hiện tại
                    current_embeddings.append(embeddings)
                    current_frames.extend(frame_paths)
                    current_captions.extend(captions)
                    current_categories.extend(categories)
                    
                    # Khi đủ chunk_size hoặc đã xử lý hết, lưu vào file
                    if len(current_frames) >= chunk_size or batch_idx == len(all_loader) - 1:
                        combined_embeddings = np.vstack(current_embeddings)
                        
                        # Lưu phần này
                        chunk_file = os.path.join(embedding_dir, f"embeddings_chunk_{chunk_idx:03d}.npy")
                        np.save(chunk_file, combined_embeddings)
                        
                        # Lưu metadata
                        with open(os.path.join(embedding_dir, f"metadata_chunk_{chunk_idx:03d}.json"), "w") as f:
                            json.dump({
                                "frame_paths": current_frames,
                                "captions": current_captions,
                                "categories": current_categories,
                                "size": len(current_frames)
                            }, f)
                        
                        # Thêm vào danh sách tổng
                        all_frame_paths.extend(current_frames)
                        all_captions.extend(current_captions)
                        all_categories.extend(current_categories)
                        
                        # Reset danh sách hiện tại
                        current_embeddings = []
                        current_frames = []
                        current_captions = []
                        current_categories = []
                        chunk_idx += 1
                        
                        # Giải phóng bộ nhớ
                        torch.cuda.empty_cache()
                else:
                    logger.error("Không tìm thấy image_embed trong kết quả model!")
        
        # Lưu metadata tổng hợp
        with open(os.path.join(save_dir, "all_frame_metadata.json"), "w") as f:
            json.dump({
                "frame_paths": all_frame_paths,
                "captions": all_captions,
                "categories": all_categories,
                "num_chunks": chunk_idx,
                "embedding_dir": embedding_dir
            }, f)
        
        logger.info(f"Đã lưu embeddings cho {len(all_frame_paths)} frames trong {chunk_idx} phần tại {embedding_dir}")

if __name__ == '__main__':
    main() 