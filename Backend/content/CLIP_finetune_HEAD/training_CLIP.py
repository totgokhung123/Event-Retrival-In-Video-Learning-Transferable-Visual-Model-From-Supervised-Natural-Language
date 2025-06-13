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
        label = self.label_map.get(meta['category'], 0)
        
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
    def __init__(self, alpha=0.7, beta=0.15, gamma=0.15, temp=0.07):
        super(MultiModalLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # Trọng số cho fusion loss
        self.beta = beta    # Trọng số cho image loss 
        self.gamma = gamma  # Trọng số cho text loss
        self.temp = temp    # Temperature cho contrastive loss
    
    def forward(self, outputs, labels):
        # Cross entropy losses cho các logits
        fusion_loss = self.ce_loss(outputs['fused_logits'], labels)
        image_loss = self.ce_loss(outputs['image_logits'], labels)
        text_loss = self.ce_loss(outputs['text_logits'], labels)
        
        # Contrastive loss giữa image-text features (tính cosine similarity)
        # Đảm bảo ép kiểu thành float32 để tính toán chính xác
        image_features = outputs['image_features'].float()
        text_features = outputs['text_features'].float()
        
        # Tính similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temp
        
        # Ground truth là matrix đơn vị (mỗi image pair với text tương ứng)
        targets = torch.arange(len(image_features), device=image_features.device)
        
        # Tính contrastive loss (từ image sang text và ngược lại)
        i2t_loss = self.ce_loss(logits, targets)
        t2i_loss = self.ce_loss(logits.t(), targets)
        contrastive_loss = (i2t_loss + t2i_loss) / 2.0
        
        # Tổng hợp các thành phần loss
        total_loss = self.alpha * fusion_loss + self.beta * image_loss + self.gamma * text_loss + contrastive_loss
        
        return total_loss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Training")
    for images, captions, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Tách forward và loss ra khỏi autocast để tránh lỗi dtype
        # Phần forward không cần autocast nữa vì đã xử lý trong model
        outputs = model(images, captions)
        
        # Tính loss với float32
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(outputs, labels)
        
        # Scale và backprop với scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs['fused_logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, captions, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Không cần autocast trong validate
            outputs = model(images, captions)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs['fused_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    return running_loss / len(dataloader), 100. * correct / total

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
    # Thay đổi các giá trị này trực tiếp trong code nếu cần
    json_path = "./Filtered_Frames_Script_sorted.json"  # Đường dẫn tới file JSON
    batch_size = 32  # Kích thước batch
    epochs = 10  # Số epochs
    lr = 1e-4  # Learning rate
    save_dir = 'clip_frame_finetune'  # Thư mục lưu model
    freeze_clip = True  # Có đóng băng tham số CLIP hay không
    seed = 42  # Seed cho tính lặp lại
    resume = ''  # Đường dẫn checkpoint để tiếp tục train (để trống nếu train từ đầu)

    # Set seed cho tính lặp lại cao
    set_seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo transform cho ảnh
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    
    # Phân chia train/val theo seed đã đặt
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Cập nhật transform cho validation set
    val_dataset.dataset.transform = val_transform

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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    model = CLIPFineTuner(pretrained="ViT-B/32", freeze_clip=freeze_clip).to(device)
    criterion = MultiModalLoss(alpha=0.7, beta=0.15, gamma=0.15)
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
        'json_path': json_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'save_dir': save_dir,
        'freeze_clip': freeze_clip,
        'seed': seed,
        'resume': resume,
        'start_epoch': start_epoch
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

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
            'best_val_acc': best_val_acc
        }, checkpoint_path)
        logger.info(f"Đã lưu checkpoint cho epoch {epoch+1} tại: {checkpoint_path}")

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
