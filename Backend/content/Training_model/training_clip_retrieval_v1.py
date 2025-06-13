import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from PIL import Image
import clip
from tqdm import tqdm
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
import logging

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
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.items = list(self.data.items())
        self.transform = transform
        self.label_map = label_map or {"Violence": 1, "Sensitive content": 0}
        
        # Đường dẫn gốc cho dataset
        self.base_dir = "/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge"
        
        # Log loại dataset và số lượng
        category = list(self.label_map.keys())[0] if len(self.items) > 0 and "category" in self.items[0][1] else "Unknown"
        logger.info(f"Loaded dataset from {os.path.basename(json_path)} with {len(self.items)} samples, category: {category}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        relative_path, meta = self.items[idx]
        caption = meta['caption']
        label = self.label_map.get(meta['category'], 0)
        
        # Xây dựng đường dẫn đầy đủ từ đường dẫn tương đối
        full_path = os.path.join(self.base_dir, relative_path)
            
        try:
            image = Image.open(full_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Lỗi đọc ảnh {full_path}: {str(e)}")
            image = torch.zeros(3, 224, 224)  # fallback
            
        return image, caption, label

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

    def forward(self, image_batch, caption_batch):
        """Forward pass với cả image và text."""
        # Đảm bảo tắt autocast cho encode CLIP để tránh lỗi dtype mismatch
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad() if all(p.requires_grad == False for p in self.clip_model.parameters()) else torch.enable_grad():
                # Encode image
                image_features = self.clip_model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                # Encode text (captions)
                text_tokens = clip.tokenize(caption_batch).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Ép kiểu về float32 để đảm bảo tính toán chính xác
            image_features = image_features.float()
            text_features = text_features.float()
            
            # Individual predictions (for auxiliary losses)
            image_logits = self.image_classifier(image_features)
            text_logits = self.text_classifier(text_features)
            
            # Fusion của image và text features
            combined_features = torch.cat([image_features, text_features], dim=1)
            fused_features = self.fusion(combined_features)
            fused_logits = self.classifier(fused_features)
        
        return {
            'fused_logits': fused_logits,  # Logits từ đặc trưng kết hợp
            'image_logits': image_logits,  # Logits từ image
            'text_logits': text_logits,    # Logits từ text
            'image_features': image_features,  # Đặc trưng ảnh (để tính contrastive loss nếu cần)
            'text_features': text_features     # Đặc trưng text (để tính contrastive loss nếu cần)
        }

class MultiModalLoss(nn.Module):
    """Loss function kết hợp các thành phần loss của multimodal."""
    def __init__(self, alpha=0.7, beta=0.15, gamma=0.15, temp=0.07, use_contrastive=True, contrastive_weight=1.0):
        super(MultiModalLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # Trọng số cho fusion loss
        self.beta = beta    # Trọng số cho image loss 
        self.gamma = gamma  # Trọng số cho text loss
        self.temp = temp    # Temperature cho contrastive loss
        self.use_contrastive = use_contrastive  # Flag để bật/tắt contrastive loss
        self.contrastive_weight = contrastive_weight  # Hệ số điều chỉnh contrastive loss
    
    def forward(self, outputs, labels):
        # Cross entropy losses cho các logits
        fusion_loss = self.ce_loss(outputs['fused_logits'], labels)
        image_loss = self.ce_loss(outputs['image_logits'], labels)
        text_loss = self.ce_loss(outputs['text_logits'], labels)
        
        # Contrastive loss giữa image-text features (tính cosine similarity)
        image_features = outputs['image_features'].float()
        text_features = outputs['text_features'].float()
        
        # Khởi tạo contrastive loss = 0
        contrastive_loss = torch.tensor(0.0, device=image_features.device)
        
        # Chỉ tính contrastive loss nếu được bật
        if self.use_contrastive and self.temp > 0:
            # Tính similarity matrix
            logits = torch.matmul(image_features, text_features.t()) / self.temp
            
            # Ground truth là matrix đơn vị (mỗi image pair với text tương ứng)
            targets = torch.arange(len(image_features), device=image_features.device)
            
            # Tính contrastive loss (từ image sang text và ngược lại)
            i2t_loss = self.ce_loss(logits, targets)
            t2i_loss = self.ce_loss(logits.t(), targets)
            contrastive_loss = (i2t_loss + t2i_loss) / 2.0
            
            # Áp dụng trọng số cho contrastive loss
            contrastive_loss = contrastive_loss * self.contrastive_weight
        
        # Tổng hợp các thành phần loss
        total_loss = self.alpha * fusion_loss + self.beta * image_loss + self.gamma * text_loss
        if self.use_contrastive:
            total_loss += contrastive_loss
        
        # Trả về dict chứa tất cả các thành phần loss
        return {
            'total': total_loss,
            'fusion': fusion_loss,
            'image': image_loss, 
            'text': text_loss,
            'contrastive': contrastive_loss
        }

def analyze_batch(outputs, labels, device, step=0):
    """Phân tích batch để tìm vấn đề tiềm ẩn."""
    # Lấy đặc trưng
    fused_logits = outputs['fused_logits'].float().detach()
    image_logits = outputs['image_logits'].float().detach()
    text_logits = outputs['text_logits'].float().detach()
    img_features = outputs['image_features'].float().detach()
    txt_features = outputs['text_features'].float().detach()
    
    # Phân tích các logits
    fused_min, fused_max = fused_logits.min().item(), fused_logits.max().item()
    fused_mean, fused_std = fused_logits.mean().item(), fused_logits.std().item()
    
    # Phân tích các features
    img_feat_norm = torch.norm(img_features, dim=1)
    txt_feat_norm = torch.norm(txt_features, dim=1)
    img_feat_mean_norm = img_feat_norm.mean().item()
    txt_feat_mean_norm = txt_feat_norm.mean().item()
    
    # Tính cosine similarity giữa image và text features
    sim_i2t = torch.matmul(img_features, txt_features.t())
    sim_diag = torch.diagonal(sim_i2t).mean().item()  # similarity của cặp image-text đúng
    sim_off_diag = (sim_i2t.sum() - torch.diagonal(sim_i2t).sum()) / (sim_i2t.numel() - len(sim_i2t))  # trung bình similarity của cặp khác nhau
    
    # Phân tích vấn đề label
    label_counts = {}
    for l in labels.cpu().numpy():
        if l not in label_counts:
            label_counts[l] = 0
        label_counts[l] += 1
    
    # Tính dự đoán chính xác
    _, preds = fused_logits.max(1)
    correct = (preds == labels).sum().item()
    
    stats = {
        'fused_logits': {'min': fused_min, 'max': fused_max, 'mean': fused_mean, 'std': fused_std},
        'image_feat_norm': img_feat_mean_norm,
        'text_feat_norm': txt_feat_mean_norm,
        'similarity': {'diagonal': sim_diag, 'off_diagonal': sim_off_diag.item()},
        'labels': label_counts,
        'accuracy': correct / len(labels)
    }
    
    # Chỉ log chi tiết cho batch đầu tiên
    if step == 0:
        logger.info(f"Debug Analysis - Batch Stats: {stats}")
    
    return stats

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    # Khởi tạo các biến theo dõi các thành phần loss riêng
    running_fusion_loss, running_image_loss = 0.0, 0.0
    running_text_loss, running_contrastive_loss = 0.0, 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for step, (images, captions, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Tách forward và loss ra khỏi autocast để tránh lỗi dtype
        # Phần forward không cần autocast nữa vì đã xử lý trong model
        outputs = model(images, captions)
        
        # Debug: Phân tích batch nếu là epoch đầu hoặc batch đầu
        if step == 0:
            analyze_batch(outputs, labels, device, step)
        
        # Tính loss với float32
        with torch.cuda.amp.autocast(enabled=False):
            losses = criterion(outputs, labels)
            loss = losses['total']  # Lấy tổng loss từ dictionary
        
        # Scale và backprop với scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Cập nhật các biến theo dõi loss
        running_loss += loss.item()
        running_fusion_loss += losses['fusion'].item()
        running_image_loss += losses['image'].item()
        running_text_loss += losses['text'].item()
        running_contrastive_loss += losses['contrastive'].item()
        
        _, predicted = outputs['fused_logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Cập nhật postfix với các thành phần loss chi tiết
        pbar.set_postfix({
            'loss': running_loss / (step + 1),
            'fusion': running_fusion_loss / (step + 1),
            'img': running_image_loss / (step + 1),
            'txt': running_text_loss / (step + 1),
            'cont': running_contrastive_loss / (step + 1),
            'acc': 100. * correct / total
        })

    # Tính trung bình các loss
    avg_loss = running_loss / len(dataloader)
    avg_fusion_loss = running_fusion_loss / len(dataloader)
    avg_image_loss = running_image_loss / len(dataloader)
    avg_text_loss = running_text_loss / len(dataloader)
    avg_contrastive_loss = running_contrastive_loss / len(dataloader)
    avg_acc = 100. * correct / total
    
    # Log chi tiết các thành phần loss sau mỗi epoch
    logger.info(f"Train - Total loss: {avg_loss:.4f}, Fusion: {avg_fusion_loss:.4f}, "
               f"Image: {avg_image_loss:.4f}, Text: {avg_text_loss:.4f}, "
               f"Contrastive: {avg_contrastive_loss:.4f}, Acc: {avg_acc:.2f}%")

    return avg_loss, avg_acc, {
        'fusion': avg_fusion_loss,
        'image': avg_image_loss,
        'text': avg_text_loss,
        'contrastive': avg_contrastive_loss
    }

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    # Khởi tạo các biến theo dõi các thành phần loss riêng
    running_fusion_loss, running_image_loss = 0.0, 0.0
    running_text_loss, running_contrastive_loss = 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for step, (images, captions, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, captions)
            
            # Debug: Phân tích batch đầu tiên
            if step == 0:
                analyze_batch(outputs, labels, device, step)
            
            losses = criterion(outputs, labels)
            loss = losses['total']
            
            # Cập nhật các biến theo dõi loss
            running_loss += loss.item()
            running_fusion_loss += losses['fusion'].item()
            running_image_loss += losses['image'].item()
            running_text_loss += losses['text'].item()
            running_contrastive_loss += losses['contrastive'].item()
            
            _, predicted = outputs['fused_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Cập nhật postfix với các thành phần loss chi tiết
            pbar.set_postfix({
                'loss': running_loss / (step + 1),
                'fusion': running_fusion_loss / (step + 1),
                'img': running_image_loss / (step + 1),
                'txt': running_text_loss / (step + 1),
                'cont': running_contrastive_loss / (step + 1),
                'acc': 100. * correct / total
            })
    
    # Tính trung bình các loss
    avg_loss = running_loss / len(dataloader)
    avg_fusion_loss = running_fusion_loss / len(dataloader)
    avg_image_loss = running_image_loss / len(dataloader)
    avg_text_loss = running_text_loss / len(dataloader)
    avg_contrastive_loss = running_contrastive_loss / len(dataloader)
    avg_acc = 100. * correct / total
    
    # Log chi tiết các thành phần loss
    logger.info(f"Val - Total loss: {avg_loss:.4f}, Fusion: {avg_fusion_loss:.4f}, "
               f"Image: {avg_image_loss:.4f}, Text: {avg_text_loss:.4f}, "
               f"Contrastive: {avg_contrastive_loss:.4f}, Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc, {
        'fusion': avg_fusion_loss,
        'image': avg_image_loss,
        'text': avg_text_loss,
        'contrastive': avg_contrastive_loss
    }

def save_checkpoint(model, optimizer, epoch, val_acc, save_path, is_best=False):
    """Lưu checkpoint model và optimizer."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    
    # Lưu checkpoint cho epoch hiện tại
    torch.save(checkpoint, save_path)
    logger.info(f"Đã lưu checkpoint tại: {save_path}")
    
    # Nếu là model tốt nhất, lưu riêng
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
        torch.save(model.state_dict(), best_path)
        logger.info(f"Đã lưu best model với val_acc: {val_acc:.2f}% tại: {best_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint để tiếp tục train."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu train từ đầu.")
        return 0, 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint.get('val_acc', 0.0)
    
    logger.info(f"Đã tải checkpoint từ epoch {epoch} với val_acc: {val_acc:.2f}%")
    return epoch + 1, val_acc

def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    # Thư mục chứa 4 file JSON
    script_dir = "/kaggle/input/script-data-merge/Script_Data_Merge_dir"
    
    # Đường dẫn tới các file JSON
    violence_train_json = os.path.join(script_dir, "caption_Violence_train.json")
    violence_val_json = os.path.join(script_dir, "caption_Violence_val.json")
    sensitive_train_json = os.path.join(script_dir, "caption_Sensitive_train.json")
    sensitive_val_json = os.path.join(script_dir, "caption_Sensitive_val.json")
    
    # Thư mục chứa ảnh
    data_dir = "/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge"
    
    # Các tham số huấn luyện
    batch_size = 64
    epochs = 10
    lr = 2e-4
    save_dir = "/kaggle/working/clip_finetune"
    freeze_clip = True
    seed = 42
    resume = ''  # Để trống nếu train từ đầu
    
    # Chọn một trong các cấu hình contrastive loss
    configuration_choice = 2  # Đổi số này để chọn cấu hình (1-4)
    
    # Các hệ số mặc định cho loss
    alpha = 0.7  # Trọng số cho fusion loss (chính)
    beta = 0.15  # Trọng số cho image loss
    gamma = 0.15  # Trọng số cho text loss
    temp = 0.07  # Temperature cho contrastive loss
    use_contrastive = True  # Có sử dụng contrastive loss hay không
    contrastive_weight = 1.0  # Hệ số điều chỉnh contrastive loss
    
    # Các cấu hình contrastive loss
    configurations = {
        1: {  # Cấu hình với contrastive loss đầy đủ
            'name': 'Full_Contrastive',
            'use_contrastive': True,
            'contrastive_weight': 1.0,
            'temp': 0.07
        },
        2: {  # Giảm mạnh contrastive loss (giảm 80%)
            'name': 'Reduced_Contrastive',
            'use_contrastive': True,
            'contrastive_weight': 0.2,
            'temp': 0.07
        },
        3: {  # Tắt contrastive loss hoàn toàn
            'name': 'No_Contrastive',
            'use_contrastive': False,
            'contrastive_weight': 0.0,
            'temp': 0.07
        },
        4: {  # Tăng temperature
            'name': 'Higher_Temperature',
            'use_contrastive': True,
            'contrastive_weight': 1.0,
            'temp': 0.2
        }
    }
    
    # Áp dụng cấu hình đã chọn
    selected_config = configurations.get(configuration_choice, configurations[1])
    use_contrastive = selected_config['use_contrastive']
    contrastive_weight = selected_config['contrastive_weight']
    temp = selected_config['temp']
    
    # Cập nhật tên thư mục lưu dựa trên cấu hình
    save_dir = os.path.join(save_dir, selected_config["name"])
    
    logger.info(f"Selected configuration: {selected_config['name']}")
    logger.info(f"use_contrastive={use_contrastive}, contrastive_weight={contrastive_weight}, temp={temp}")

    # Set seed cho tính lặp lại cao
    set_seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Log thông số hiện tại
    logger.info(f"Batch size: {batch_size}, Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")
    
    # Label map mới với "Violence": 1, "Sensitive content": 0
    label_map = {"Violence": 1, "Sensitive content": 0}
    
    # Khởi tạo transform cho ảnh (chỉ cần normalize cho CLIP, không cần tăng cường vì đã được tăng cường trước đó)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Tạo dataset
    violence_train_dataset = FrameCaptionDataset(violence_train_json, transform=train_transform, label_map=label_map)
    sensitive_train_dataset = FrameCaptionDataset(sensitive_train_json, transform=train_transform, label_map=label_map)
    violence_val_dataset = FrameCaptionDataset(violence_val_json, transform=val_transform, label_map=label_map)
    sensitive_val_dataset = FrameCaptionDataset(sensitive_val_json, transform=val_transform, label_map=label_map)
    
    # Kết hợp các dataset
    train_dataset = ConcatDataset([violence_train_dataset, sensitive_train_dataset])
    val_dataset = ConcatDataset([violence_val_dataset, sensitive_val_dataset])
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Sử dụng worker_init_fn để đảm bảo workers có seed khác nhau nhưng deterministic
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=False
    )

    model = CLIPFineTuner(pretrained="ViT-B/32", freeze_clip=freeze_clip).to(device)
    criterion = MultiModalLoss(alpha=alpha, beta=beta, gamma=gamma, temp=temp, use_contrastive=use_contrastive, contrastive_weight=contrastive_weight)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()

    # Tải checkpoint nếu có
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if resume:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, resume, device)

    # Lưu thông tin cấu hình training để tái tạo sau này
    config = {
        'violence_train_json': violence_train_json,
        'violence_val_json': violence_val_json,
        'sensitive_train_json': sensitive_train_json,
        'sensitive_val_json': sensitive_val_json,
        'data_dir': data_dir,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'save_dir': save_dir,
        'freeze_clip': freeze_clip,
        'seed': seed,
        'resume': resume,
        'start_epoch': start_epoch,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'temp': temp,
        'use_contrastive': use_contrastive,
        'contrastive_weight': contrastive_weight,
        'configuration_name': selected_config['name']
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc, train_loss_components = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_loss_components = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Kiểm tra xem có phải best model không
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # Lưu best model
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"Đã lưu best model với val_acc: {val_acc:.2f}% tại: {best_path}")
        
        # Luôn lưu checkpoint vào cùng một file sau mỗi epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'train_loss_components': train_loss_components,
            'val_loss_components': val_loss_components
        }, checkpoint_path)
        logger.info(f"Đã lưu checkpoint cho epoch {epoch+1} tại: {checkpoint_path}")

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main() 