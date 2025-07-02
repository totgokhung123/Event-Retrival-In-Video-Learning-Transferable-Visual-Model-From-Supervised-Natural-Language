import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import clip
from PIL import Image
from tqdm import tqdm
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clip_finetune_correct.log'),
        logging.StreamHandler()
    ]
)
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

class ContentAwareDataset(Dataset):
    """
    Dataset cho ảnh và caption, tập trung vào việc nhận biết nội dung nhạy cảm
    """
    def __init__(self, json_path, base_dir, transform=None, token_length=77):
        """
        Args:
            json_path: Đường dẫn đến file JSON chứa dữ liệu
            base_dir: Thư mục gốc chứa hình ảnh
            transform: Transform cho ảnh
            token_length: Độ dài tối đa của token
        """
        self.base_dir = base_dir
        self.transform = transform
        self.token_length = token_length
        
        # Đọc dữ liệu từ JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.valid_items = []
        for path, meta in self.data.items():
            # Kiểm tra đường dẫn ảnh
            full_path = os.path.join(base_dir, path)
            if not os.path.exists(full_path):
                continue
            
            # Kiểm tra caption
            caption = meta.get('caption', '').strip()
            if not caption:
                continue
            
            self.valid_items.append((path, meta))
        
        logger.info(f"Đã tải dataset từ {json_path}")
        logger.info(f"Tổng số mẫu: {len(self.data)}, Mẫu hợp lệ: {len(self.valid_items)}")
        
        # Log phân bố nhãn nếu có
        category_counts = {}
        for _, meta in self.valid_items:
            cat = meta.get('category', 'NonViolence')
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1
        logger.info(f"Phân bố loại nội dung: {category_counts}")
        
        # Tạo mapping từ category name sang id
        self.category_mapping = {
            "Sensitive content": 0,
            "Violence": 1,
            "NonViolence": 2  # Thay thế Unknown thành NonViolence
        }

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        """Trả về (image, caption, category_id)"""
        path, meta = self.valid_items[idx]
        caption = meta.get('caption', '')
        category = meta.get('category', 'NonViolence')
        category_id = self.category_mapping.get(category, 2)  # Default to NonViolence (2) if not found
        
        # Xây dựng đường dẫn đầy đủ
        full_path = os.path.join(self.base_dir, path)
        
        # Đọc và xử lý ảnh
        try:
            image = Image.open(full_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {full_path}: {str(e)}")
            # Fallback: Tạo ảnh trống nếu có lỗi
            image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
        
        return image, caption, category_id

# Mô hình CLIP với classification head
class CLIPWithClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=3, freeze_layers=7):
        super(CLIPWithClassifier, self).__init__()
        self.clip_model = clip_model
        
        # Convert model to float32 để đảm bảo nhất quán
        self.clip_model.float()
        
        # Đóng băng các lớp đầu của visual encoder
        if freeze_layers > 0:
            # Đóng băng các lớp đầu của visual encoder
            for i, param in enumerate(self.clip_model.visual.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
            
            logger.info(f"Đã đóng băng {freeze_layers} lớp đầu tiên của visual encoder")
            
            # Đóng băng một phần của text encoder
            text_layers = list(self.clip_model.transformer.parameters())
            for i, param in enumerate(text_layers):
                if i < freeze_layers:
                    param.requires_grad = False
            
            logger.info(f"Đã đóng băng {freeze_layers} lớp đầu tiên của text encoder")
        
        # Thêm classification head
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Logit scale parameter
        self.logit_scale = self.clip_model.logit_scale
    
    def forward(self, images, texts, get_embeddings=False):
        # Đảm bảo đầu vào có cùng kiểu dữ liệu với model
        images = images.type(torch.float32)
        
        # QUAN TRỌNG: texts phải giữ nguyên kiểu Int/Long cho embedding layer
        # Chỉ convert kiểu float cho tensor hình ảnh
        
        # Lấy embeddings từ CLIP model
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)
        
        # Chuyển đổi về float32 để xử lý tiếp
        image_features = image_features.float()
        text_features = text_features.float()
        
        # Chuẩn hoá embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
       
        # Tính logits cho contrastive learning
        logit_scale = self.logit_scale.exp().float()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Classification logits từ image features
        class_logits = self.classifier(image_features)
        
        if get_embeddings:
            return image_features, text_features, logits_per_image, logits_per_text, class_logits
        
        return logits_per_image, logits_per_text, class_logits

# Hàm tạo kết hợp datasets từ nhiều nguồn
def create_combined_dataset(json_paths, data_dir, transform=None):
    """
    Tạo dataset kết hợp từ nhiều file JSON
    
    Args:
        json_paths: Danh sách đường dẫn các file JSON
        data_dir: Thư mục gốc chứa dữ liệu
        transform: Transform cho ảnh
    
    Returns:
        ConcatDataset: Dataset kết hợp
    """
    datasets = []
    for json_path in json_paths:
        if os.path.exists(json_path):
            dataset = ContentAwareDataset(json_path, data_dir, transform)
            datasets.append(dataset)
        else:
            logger.warning(f"Không tìm thấy file JSON: {json_path}")
    
    if not datasets:
        raise ValueError("Không có dataset nào được tạo! Kiểm tra lại đường dẫn các file JSON.")
    
    combined_dataset = ConcatDataset(datasets)
    logger.info(f"Đã tạo dataset kết hợp với tổng số {len(combined_dataset)} mẫu từ {len(datasets)} nguồn")
    return combined_dataset

# Hàm lưu checkpoint
def save_checkpoint(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Đã lưu checkpoint sau epoch {epoch+1} tại {save_path}")

# Hàm đánh giá model
def validate_model(model, validation_loader, device):
    model.eval()
    total_loss = 0
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_cls = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, texts, labels) in enumerate(validation_loader):
            images = images.to(device).float()  # Đảm bảo floating point
            texts = clip.tokenize(texts, truncate=True).to(device)  # Giữ nguyên kiểu Int/Long
            labels = labels.to(device)
            
            # Forward pass
            logits_per_image, logits_per_text, class_logits = model(images, texts)
            
            # Ground truth cho contrastive loss là diagonal
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            # Tính contrastive loss
            contrast_loss = (loss_img(logits_per_image, ground_truth) + 
                          loss_txt(logits_per_text, ground_truth)) / 2
            
            # Tính classification loss
            cls_loss = loss_cls(class_logits, labels)
            
            # Kết hợp loss
            batch_loss = contrast_loss + cls_loss
            
            # Tracking accuracy
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += batch_loss.item()
    
    avg_loss = total_loss / len(validation_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# Hàm vẽ biểu đồ loss
def plot_losses(train_losses, val_losses, train_acc, val_acc, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Đã lưu biểu đồ tại {save_path}")

def main():
    # Cấu hình training
    CONFIG = {
        'seed': 42,
        'batch_size': 32,  # Tăng batch size cho contrastive learning tốt hơn
        'num_workers': 4,
        'epochs': 10,
        'lr': 1e-5,  # Giảm learning rate
        'weight_decay': 0.01,  # Giảm weight decay
        'betas': (0.9, 0.98),
        'eps': 1e-6,
        'grad_clip': 1.0,
        'early_stopping': 5,  # Tăng patience
        'save_dir': '/kaggle/working/clip_finetune_checkpoints',
        'clip_model': 'ViT-B/32',
        'data_dir': "/kaggle/input/data-merge-2-themnonviolence2/Data_Merge_2",
        'train_json': [
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_Violence_train.json",
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_Sensitive_train.json",
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_NonViolence_train.json"
        ],
        'val_json': [
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_Violence_val.json",
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_Sensitive_val.json",
            "/kaggle/input/script-data-merge2/script_data_merge_2/caption_NonViolence_val.json"
        ],
        'freeze_layers': 8,  # Đóng băng 7 layer đầu tiên
        'contrastive_weight': 1.0,  # Trọng số cho contrastive loss
        'classification_weight': 0.2,  # Trọng số cho classification loss
        'temperature': 0.07,  # Temperature cho contrastive loss
    }
    
    # Đặt seed và chuẩn bị thư mục lưu
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Lưu config để tái hiện
    with open(os.path.join(CONFIG['save_dir'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Thiết lập device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Sử dụng device: {device}")
    
    # Tải model CLIP - PHẢI set jit=False cho training
    clip_model, preprocess = clip.load(CONFIG['clip_model'], device=device, jit=False)
    
    # Khởi tạo mô hình với classifier và chuyển sang float32
    model = CLIPWithClassifier(clip_model, num_classes=3, freeze_layers=CONFIG['freeze_layers'])
    model = model.to(device)
    model = model.float()  # Đảm bảo model ở precision float32
    
    # Đặt temperature cho CLIP
    with torch.no_grad():
        model.logit_scale.copy_(torch.tensor(np.log(1.0 / CONFIG['temperature']), dtype=torch.float32))
    
    # Tạo dataset kết hợp cho training và validation
    train_dataset = create_combined_dataset(CONFIG['train_json'], CONFIG['data_dir'], transform=preprocess)
    val_dataset = create_combined_dataset(CONFIG['val_json'], CONFIG['data_dir'], transform=preprocess)
    
    # Tạo dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True  # Quan trọng: bỏ batch cuối nếu không đủ
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Log thông tin về số lượng batch
    logger.info(f"Số batch trong train_loader: {len(train_loader)}")
    logger.info(f"Số batch trong val_loader: {len(val_loader)}")
    
    # Loss functions
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_cls = nn.CrossEntropyLoss()
    
    # Sử dụng các tham số khác nhau cho các phần khác nhau của mô hình
    # Tạo các tham số không trùng lặp cho từng nhóm
    visual_params = []
    text_params = []
    classifier_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Bỏ qua các tham số đã freeze
            
        if 'visual' in name:
            visual_params.append(param)
        elif 'transformer' in name:
            text_params.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
        else:
            other_params.append(param)
    
    # Log số lượng tham số trong mỗi nhóm
    logger.info(f"Số lượng tham số Visual: {len(visual_params)}")
    logger.info(f"Số lượng tham số Text: {len(text_params)}")
    logger.info(f"Số lượng tham số Classifier: {len(classifier_params)}")
    logger.info(f"Số lượng tham số khác: {len(other_params)}")
    
    # Tạo nhóm tham số cho optimizer - đảm bảo không trùng lặp
    params = [
        {'params': visual_params, 'lr': CONFIG['lr']},
        {'params': text_params, 'lr': CONFIG['lr'] * 0.5},  # Học chậm hơn cho text encoder
        {'params': classifier_params, 'lr': CONFIG['lr'] * 5},  # Học nhanh hơn cho classifier
        {'params': other_params, 'lr': CONFIG['lr']}  # Các tham số còn lại
    ]
    
    # Optimizer
    optimizer = optim.AdamW(
        params,
        betas=CONFIG['betas'],
        eps=CONFIG['eps'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['lr']/10)
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_idx, (images, texts, labels) in pbar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Chuyển dữ liệu sang device và đảm bảo đúng kiểu dữ liệu
            images = images.to(device).float()  # Đảm bảo floating point
            text_tokens = clip.tokenize(texts, truncate=True).to(device)  # Giữ nguyên kiểu Int/Long
            labels = labels.to(device)
            
            # Forward pass
            logits_per_image, logits_per_text, class_logits = model(images, text_tokens)
            
            # Ground truth: diagonal matching
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            # Calculate contrastive loss
            contrast_loss = (loss_img(logits_per_image, ground_truth) + 
                          loss_txt(logits_per_text, ground_truth)) / 2
            
            # Calculate classification loss
            cls_loss = loss_cls(class_logits, labels)
            
            # Combined loss
            total_loss = CONFIG['contrastive_weight'] * contrast_loss + CONFIG['classification_weight'] * cls_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping để đảm bảo ổn định
            if CONFIG['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            
            # Optimizer step
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update running loss
            running_loss += total_loss.item()
            
            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch+1}/{CONFIG['epochs']}, "
                f"Loss: {total_loss.item():.4f}, "
                f"Cont: {contrast_loss.item():.4f}, "
                f"Cls: {cls_loss.item():.4f}"
            )
        
        # Step the scheduler
        scheduler.step()
        
        # Tính loss và accuracy trung bình
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.2f}%")
        
        # Validation
        val_loss, val_acc = validate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Chỉ lưu best model, không lưu từng checkpoint để tiết kiệm bộ nhớ
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Lưu best model
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(CONFIG['save_dir'], "best_model.pt")
            )
            logger.info(f"Đã lưu best model với validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss không cải thiện. Patience: {patience_counter}/{CONFIG['early_stopping']}")
            
            if patience_counter >= CONFIG['early_stopping']:
                logger.info(f"Early stopping sau {epoch+1} epochs")
                break
    
    # Lưu checkpoint cuối cùng
    save_checkpoint(
        model,
        optimizer,
        CONFIG['epochs'] - 1,
        val_loss,
        os.path.join(CONFIG['save_dir'], "final_checkpoint.pt")
    )
    logger.info(f"Đã lưu checkpoint cuối cùng sau khi hoàn thành training")
    
    # Vẽ biểu đồ loss và accuracy
    plot_losses(
        train_losses, 
        val_losses, 
        train_accs,
        val_accs,
        os.path.join(CONFIG['save_dir'], 'metrics_plot.png')
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 