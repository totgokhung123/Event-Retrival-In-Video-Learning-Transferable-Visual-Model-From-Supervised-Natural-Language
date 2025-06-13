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

class CLIPContrastiveFineTuner(nn.Module):
    def __init__(self, num_classes=2, pretrained="ViT-B/32", freeze_clip=True, embed_dim=512):
        super(CLIPContrastiveFineTuner, self).__init__()
        self.clip_model, _ = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = self.clip_model.visual.output_dim
        
        # Đóng băng hoặc fine-tune CLIP
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Projection layer để giảm chiều (nếu cần)
        if embed_dim > 0:
            self.image_projection = nn.Linear(self.embedding_dim, embed_dim)
            self.text_projection = nn.Linear(self.embedding_dim, embed_dim)
            self.final_embedding_dim = embed_dim
        else:
            self.image_projection = nn.Identity()
            self.text_projection = nn.Identity()
            self.final_embedding_dim = self.embedding_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Đặt temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, image_batch):
        with torch.no_grad() if all(p.requires_grad == False for p in self.clip_model.parameters()) else torch.enable_grad():
            image_features = self.clip_model.encode_image(image_batch)
        return image_features
    
    def encode_text(self, text_batch):
        with torch.no_grad() if all(p.requires_grad == False for p in self.clip_model.parameters()) else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_batch)
        return text_features
    
    def project_features(self, image_features, text_features):
        # Đảm bảo sử dụng cùng dtype với projection layers
        if hasattr(self.image_projection, 'weight'):
            image_features = image_features.to(self.image_projection.weight.dtype)
        if hasattr(self.text_projection, 'weight'):
            text_features = text_features.to(self.text_projection.weight.dtype)
            
        # Project to common embedding space
        image_embed = self.image_projection(image_features)
        text_embed = self.text_projection(text_features)
        
        # Normalize
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        return image_embed, text_embed
    
    def forward(self, image_batch, text_batch=None):
        image_features = self.encode_image(image_batch)
        
        # Nếu chỉ cần classification
        if text_batch is None:
            # Đảm bảo classifier và image_features có cùng dtype
            logits = self.classifier(image_features.to(self.classifier[0].weight.dtype))
            return {"logits": logits}
        
        # Encode text
        text_features = self.encode_text(text_batch)
        
        # Project và normalize features
        image_embed, text_embed = self.project_features(image_features, text_features)
        
        # Đảm bảo image_embed và text_embed cùng dtype trước khi tính dot product
        if image_embed.dtype != text_embed.dtype:
            text_embed = text_embed.to(image_embed.dtype)
        
        # Tính similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embed @ text_embed.t()
        logits_per_text = logits_per_image.t()
        
        # Classification logits từ image features
        class_logits = self.classifier(image_features.to(self.classifier[0].weight.dtype))
        
        return {
            "logits": class_logits,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_features": image_features,
            "text_features": text_features,
            "image_embed": image_embed,
            "text_embed": text_embed
        }

def contrastive_loss(logits_per_image, logits_per_text, labels=None):
    # Ground-truth là diagonal matrix (ảnh và text cùng index match với nhau)
    batch_size = logits_per_image.size(0)
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)
    
    # Tính cross entropy loss từ logits
    loss_i2t = nn.functional.cross_entropy(logits_per_image, ground_truth)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, ground_truth)
    
    # Average của image-to-text và text-to-image loss
    contrastive_loss = (loss_i2t + loss_t2i) / 2
    
    return contrastive_loss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, lambda_contrastive=0.5):
    model.train()
    running_class_loss, running_contrastive_loss = 0.0, 0.0
    running_total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, captions, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Tokenize captions
        with torch.no_grad():
            text_tokens = clip.tokenize(captions).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, text_tokens)
            
            # Classification loss
            class_loss = criterion(outputs["logits"], labels)
            
            # Contrastive loss
            contr_loss = contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
            
            # Combine losses
            loss = (1 - lambda_contrastive) * class_loss + lambda_contrastive * contr_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        running_class_loss += class_loss.item()
        running_contrastive_loss += contr_loss.item()
        running_total_loss += loss.item()
        
        _, predicted = outputs["logits"].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_total_loss / (pbar.n + 1), 
            'cls_loss': running_class_loss / (pbar.n + 1),
            'con_loss': running_contrastive_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return {
        'total_loss': running_total_loss / len(dataloader),
        'class_loss': running_class_loss / len(dataloader),
        'contrastive_loss': running_contrastive_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }

def validate(model, dataloader, criterion, device, lambda_contrastive=0.5):
    model.eval()
    running_class_loss, running_contrastive_loss = 0.0, 0.0
    running_total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, captions, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Tokenize captions
            text_tokens = clip.tokenize(captions).to(device)
            
            outputs = model(images, text_tokens)
            
            # Classification loss
            class_loss = criterion(outputs["logits"], labels)
            
            # Contrastive loss
            contr_loss = contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
            
            # Combine losses
            loss = (1 - lambda_contrastive) * class_loss + lambda_contrastive * contr_loss
            
            # Update metrics
            running_class_loss += class_loss.item()
            running_contrastive_loss += contr_loss.item()
            running_total_loss += loss.item()
            
            _, predicted = outputs["logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_total_loss / (pbar.n + 1), 
                'cls_loss': running_class_loss / (pbar.n + 1),
                'con_loss': running_contrastive_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    return {
        'total_loss': running_total_loss / len(dataloader),
        'class_loss': running_class_loss / len(dataloader),
        'contrastive_loss': running_contrastive_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """Lưu checkpoint model và optimizer."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Lưu checkpoint cho epoch hiện tại
    torch.save(checkpoint, save_path)
    logger.info(f"Đã lưu checkpoint tại: {save_path}")
    
    # Nếu là model tốt nhất, lưu riêng
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
        torch.save(model.state_dict(), best_path)
        logger.info(f"Đã lưu best model với val_acc: {metrics['accuracy']:.2f}% tại: {best_path}")

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
    
    logger.info(f"Đã tải checkpoint từ epoch {epoch} với val_acc: {metrics.get('accuracy', 0):.2f}%")
    return epoch + 1, metrics

def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    # Thay đổi các giá trị này trực tiếp trong code nếu cần
    json_path = "./Filtered_Frames_Script_sorted.json"  # Đường dẫn tới file JSON
    batch_size = 64  # Tăng batch size để có đủ negative samples cho contrastive learning
    epochs = 20  # Có thể cần train lâu hơn để học cả contrastive
    lr = 5e-5  # Learning rate thấp hơn để fine-tune ổn định hơn
    save_dir = 'clip_contrastive_finetune'  # Thư mục lưu model
    freeze_clip = True  # Có đóng băng tham số CLIP hay không
    embed_dim = 256  # Chiều của embedding space sau khi project
    lambda_contrastive = 0.5  # Trọng số của contrastive loss (0-1)
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

    model = CLIPContrastiveFineTuner(
        pretrained="ViT-B/32", 
        freeze_clip=freeze_clip, 
        embed_dim=embed_dim
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()

    # Tải checkpoint nếu có
    start_epoch = 0
    best_metrics = {'accuracy': 0.0}
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if resume:
        start_epoch, best_metrics = load_checkpoint(model, optimizer, resume, device)

    # Lưu thông tin cấu hình training để tái tạo sau này
    config = {
        'json_path': json_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'save_dir': save_dir,
        'freeze_clip': freeze_clip,
        'embed_dim': embed_dim,
        'lambda_contrastive': lambda_contrastive,
        'seed': seed,
        'resume': resume,
        'start_epoch': start_epoch
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, lambda_contrastive
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, lambda_contrastive)
        
        # Log metrics
        logger.info(
            f"Train: loss={train_metrics['total_loss']:.4f}, "
            f"class_loss={train_metrics['class_loss']:.4f}, "
            f"contr_loss={train_metrics['contrastive_loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Val: loss={val_metrics['total_loss']:.4f}, "
            f"class_loss={val_metrics['class_loss']:.4f}, "
            f"contr_loss={val_metrics['contrastive_loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.2f}%"
        )

        # Kiểm tra xem có phải best model không
        is_best = val_metrics['accuracy'] > best_metrics.get('accuracy', 0)
        if is_best:
            best_metrics = val_metrics
            # Lưu best model
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"Đã lưu best model với val_acc: {val_metrics['accuracy']:.2f}% tại: {best_path}")
        
        # Luôn lưu checkpoint vào cùng một file sau mỗi epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'best_metrics': best_metrics
        }, checkpoint_path)
        logger.info(f"Đã lưu checkpoint cho epoch {epoch+1} tại: {checkpoint_path}")

    logger.info(f"Training completed. Best validation accuracy: {best_metrics['accuracy']:.2f}%")

if __name__ == '__main__':
    main() 