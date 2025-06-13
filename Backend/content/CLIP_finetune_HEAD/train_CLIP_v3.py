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
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import copy
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def __init__(self, num_classes=2, pretrained="ViT-B/32", training_phase=1):
        """
        CLIP Fine-tuner với huấn luyện nhiều giai đoạn và cải tiến cho fine-tune.
        
        Args:
            num_classes: Số lớp để phân loại.
            pretrained: Loại mô hình CLIP pretrained.
            training_phase: Giai đoạn huấn luyện (1: chỉ lớp mới, 2: mở đóng băng từng phần, 3: toàn bộ).
        """
        super(CLIPFineTuner, self).__init__()
        self.clip_model, self.preprocess = clip.load(pretrained, device="cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.clip_model.visual.output_dim
        self.device = next(self.clip_model.parameters()).device
        self.training_phase = training_phase
        
        # Xác định các phần của mô hình CLIP để fine-tune theo từng giai đoạn
        if hasattr(self.clip_model, 'visual') and hasattr(self.clip_model.visual, 'transformer'):
            # Đối với ViT
            self.visual_blocks = self.clip_model.visual.transformer.resblocks
            self.text_blocks = self.clip_model.transformer.resblocks
            self._setup_gradients_for_phase()
        else:
            logger.warning("Không nhận dạng được kiến trúc CLIP. Sẽ đóng băng toàn bộ trong phase 1.")
            if training_phase == 1:
                for param in self.clip_model.parameters():
                    param.requires_grad = False
        
        # Layer normalization và dropout ở các lớp mới tăng cường
        self.image_norm = nn.LayerNorm(embedding_dim)
        self.text_norm = nn.LayerNorm(embedding_dim)
        
        # Fusion layer cải tiến với normalization và dropout cao hơn
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
        
        # Classifier với label smoothing tích hợp
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
        
        # Hệ số cho các thành phần loss
        self.alpha = 0.7  # Trọng số cho fusion loss
        self.beta = 0.15  # Trọng số cho image loss
        self.gamma = 0.15  # Trọng số cho text loss
        
        logger.info(f"Khởi tạo CLIPFineTuner với training_phase={training_phase}")
        
    def _setup_gradients_for_phase(self):
        """Thiết lập gradients dựa trên giai đoạn huấn luyện"""
        # Mặc định đóng băng tất cả
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Giai đoạn 1: Chỉ train lớp mới, đóng băng CLIP
        if self.training_phase == 1:
            logger.info("Phase 1: Đóng băng toàn bộ CLIP, chỉ train các lớp fusion và classifier.")
            # Đóng băng toàn bộ CLIP, train lớp fusion + classifier
            return
            
        # Giai đoạn 2: Mở đóng băng từng phần cuối của CLIP
        elif self.training_phase == 2:
            logger.info("Phase 2: Mở đóng băng các lớp cuối của CLIP.")
            
            # Mở khóa 3 khối cuối của visual encoder
            num_visual_blocks = len(self.visual_blocks)
            for i in range(num_visual_blocks - 3, num_visual_blocks):
                for param in self.visual_blocks[i].parameters():
                    param.requires_grad = True
                    
            # Mở khóa 3 khối cuối của text encoder
            num_text_blocks = len(self.text_blocks)
            for i in range(num_text_blocks - 3, num_text_blocks):
                for param in self.text_blocks[i].parameters():
                    param.requires_grad = True
                    
            # Mở khóa các lớp projection
            for param in self.clip_model.visual.ln_post.parameters():
                param.requires_grad = True
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = True
            self.clip_model.visual.proj.requires_grad = True
            self.clip_model.text_projection.requires_grad = True
                
        # Giai đoạn 3: Fine-tune toàn bộ CLIP
        elif self.training_phase == 3:
            logger.info("Phase 3: Fine-tune toàn bộ CLIP.")
            for param in self.clip_model.parameters():
                param.requires_grad = True
                
        # Log số lượng tham số được train
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Tổng số tham số: {total_params:,}")
        logger.info(f"Số tham số được train: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, image_batch, caption_batch):
        """Forward pass với cả image và text."""
        # Xử lý khác nhau cho các phase
        with torch.cuda.amp.autocast(enabled=False):  # Đảm bảo chạy ở độ chính xác FP32
            # Encode image
            if self.training and self.training_phase > 1:
                # Chế độ train + phase > 1: Sử dụng encode image với gradients
                image_features = self.clip_model.encode_image(image_batch.float())
            else:
                # Phase 1 hoặc eval mode: Có thể sử dụng no_grad để tối ưu
                with torch.no_grad() if not self.training or self.training_phase == 1 else torch.enable_grad():
                    image_features = self.clip_model.encode_image(image_batch.float())
            
            # Normalize image features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Encode text
            text_tokens = clip.tokenize(caption_batch).to(self.device)
            if self.training and self.training_phase > 1:
                # Chế độ train + phase > 1: Sử dụng encode text với gradients
                text_features = self.clip_model.encode_text(text_tokens)
            else:
                # Phase 1 hoặc eval mode: Có thể sử dụng no_grad để tối ưu
                with torch.no_grad() if not self.training or self.training_phase == 1 else torch.enable_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize text features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Đảm bảo tất cả đều ở FP32
        image_features = image_features.float()
        text_features = text_features.float()
        
        # Áp dụng normalization
        image_features = self.image_norm(image_features)
        text_features = self.text_norm(text_features)
        
        # Individual predictions (for auxiliary losses)
        image_logits = self.image_classifier(image_features)
        text_logits = self.text_classifier(text_features)
        
        # Fusion của image và text features
        combined_features = torch.cat([image_features, text_features], dim=1)
        fused_features = self.fusion(combined_features)
        fused_logits = self.classifier(fused_features)
        
        return {
            'fused_logits': fused_logits,      # Logits từ đặc trưng kết hợp
            'image_logits': image_logits,      # Logits từ image
            'text_logits': text_logits,        # Logits từ text
            'image_features': image_features,  # Đặc trưng ảnh
            'text_features': text_features     # Đặc trưng text
        }

class MultiModalLossV2(nn.Module):
    """Loss function cải tiến với label smoothing và entropy regularization."""
    def __init__(self, alpha=0.7, beta=0.15, gamma=0.15, temp=0.07, label_smoothing=0.1, 
                 weight_decay=1e-5, entropy_weight=0.01):
        super(MultiModalLossV2, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.alpha = alpha  # Trọng số cho fusion loss
        self.beta = beta    # Trọng số cho image loss 
        self.gamma = gamma  # Trọng số cho text loss
        self.temp = temp    # Temperature cho contrastive loss
        self.weight_decay = weight_decay  # L2 regularization
        self.entropy_weight = entropy_weight  # Trọng số cho entropy regularization
    
    def forward(self, outputs, labels, model=None):
        # Đảm bảo tất cả đều ở kiểu dữ liệu float32
        for key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].float()
        
        # Cross entropy losses cho các logits
        fusion_loss = self.ce_loss(outputs['fused_logits'], labels)
        image_loss = self.ce_loss(outputs['image_logits'], labels)
        text_loss = self.ce_loss(outputs['text_logits'], labels)
        
        # Contrastive loss giữa image-text features (tính cosine similarity)
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        
        # Tính similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temp
        
        # Ground truth là matrix đơn vị (mỗi image pair với text tương ứng)
        targets = torch.arange(len(image_features), device=image_features.device)
        
        # Tính contrastive loss (từ image sang text và ngược lại)
        i2t_loss = self.ce_loss(logits, targets)
        t2i_loss = self.ce_loss(logits.t(), targets)
        contrastive_loss = (i2t_loss + t2i_loss) / 2.0
        
        # Entropy regularization để tránh overfitting (khuyến khích dự đoán đa dạng)
        entropy_loss = 0.0
        if self.entropy_weight > 0:
            fusion_probs = torch.softmax(outputs['fused_logits'], dim=1)
            fusion_entropy = -(fusion_probs * torch.log(fusion_probs + 1e-6)).sum(dim=1).mean()
            entropy_loss = -self.entropy_weight * fusion_entropy  # Maximize entropy = minimize -entropy
        
        # L2 regularization
        l2_reg = 0.0
        if model is not None and self.weight_decay > 0:
            # Chỉ áp dụng L2 reg cho các tham số được train
            for name, param in model.named_parameters():
                if param.requires_grad and 'weight' in name:
                    l2_reg += torch.norm(param, p=2)
            l2_reg *= self.weight_decay
        
        # Tổng hợp các thành phần loss
        total_loss = (self.alpha * fusion_loss + 
                     self.beta * image_loss + 
                     self.gamma * text_loss + 
                     contrastive_loss + 
                     entropy_loss + 
                     l2_reg)
        
        return total_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad_norm=1.0):
    """Huấn luyện một epoch với gradient clipping và logging chi tiết."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Training")
    
    for images, captions, labels in pbar:
        # Đảm bảo dữ liệu ở kiểu float32 để tránh lỗi dtype mismatch
        images = images.float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass (đã xử lý float32 trong model.forward)
        outputs = model(images, captions)
        
        # Tính loss với mô hình hiện tại
        loss = criterion(outputs, labels, model)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs['fused_logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Đánh giá model trên tập validation với logging chi tiết."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, captions, labels in pbar:
            # Đảm bảo dữ liệu ở kiểu float32 để tránh lỗi dtype mismatch
            images = images.float().to(device)
            labels = labels.to(device)
            
            # Forward (đã xử lý float32 trong model.forward)
            outputs = model(images, captions)
            
            # Tính loss
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs['fused_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Thu thập dự đoán và nhãn cho các chỉ số metric chi tiết hơn
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    # Tính precision, recall, f1 (nếu là binary classification)
    if len(torch.unique(torch.cat(all_labels))) == 2:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return running_loss / len(dataloader), 100. * correct / total

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, save_path, training_phase, is_best=False):
    """Lưu checkpoint với nhiều thông tin hơn."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'training_phase': training_phase
    }
    
    # Lưu checkpoint cho epoch hiện tại
    torch.save(checkpoint, save_path)
    logger.info(f"Đã lưu checkpoint phase {training_phase} tại: {save_path}")
    
    # Nếu là model tốt nhất, lưu riêng
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), f"best_model_phase{training_phase}.pt")
        torch.save(model.state_dict(), best_path)
        logger.info(f"Đã lưu best model phase {training_phase} với val_acc: {val_acc:.2f}% tại: {best_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint với nhiều thông tin hơn."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu train từ đầu.")
        return 0, 0.0, 1  # epoch, val_acc, training_phase
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint['epoch']
    val_acc = checkpoint.get('val_acc', 0.0)
    training_phase = checkpoint.get('training_phase', 1)
    
    logger.info(f"Đã tải checkpoint từ phase {training_phase}, epoch {epoch} với val_acc: {val_acc:.2f}%")
    return epoch + 1, val_acc, training_phase

def get_optimizer_and_scheduler(model, training_phase, batch_size, max_epochs, train_loader_len):
    """Tạo optimizer và scheduler phù hợp với từng phase."""
    # Learning rate và weight decay phù hợp với từng phase
    if training_phase == 1:
        # Giai đoạn 1: LR cao hơn cho train lớp mới
        base_lr = 1e-4
        weight_decay = 1e-4
    elif training_phase == 2:
        # Giai đoạn 2: LR thấp hơn cho fine-tune một phần
        base_lr = 5e-5
        weight_decay = 5e-5
    else:
        # Giai đoạn 3: LR rất thấp cho fine-tune toàn bộ
        base_lr = 1e-5
        weight_decay = 1e-5
        
    # Sử dụng discriminative learning rates nếu ở phase 2 hoặc 3
    params = []
    if training_phase > 1:
        # Nhóm tham số CLIP với LR thấp
        clip_params = []
        for name, param in model.named_parameters():
            if 'clip_model' in name and param.requires_grad:
                clip_params.append(param)
        
        # Nhóm các tham số của lớp mới với LR cao hơn
        new_params = []
        for name, param in model.named_parameters():
            if 'clip_model' not in name and param.requires_grad:
                new_params.append(param)
                
        # Đối với phase 3, tiếp tục chia CLIP thành 2 nhóm: đầu và cuối
        if training_phase == 3:
            early_clip_params = []
            late_clip_params = []
            
            for name, param in model.named_parameters():
                if 'clip_model' in name and param.requires_grad:
                    # Các khối đầu với LR rất thấp
                    if any(x in name for x in ['visual.conv1', 'visual.transformer.resblocks.0', 
                                               'visual.transformer.resblocks.1', 'visual.transformer.resblocks.2',
                                               'transformer.resblocks.0', 'transformer.resblocks.1', 'transformer.resblocks.2']):
                        early_clip_params.append(param)
                    # Các khối cuối với LR cao hơn một chút
                    else:
                        late_clip_params.append(param)
            
            params = [
                {'params': early_clip_params, 'lr': base_lr / 10, 'weight_decay': weight_decay},
                {'params': late_clip_params, 'lr': base_lr / 3, 'weight_decay': weight_decay},
                {'params': new_params, 'lr': base_lr, 'weight_decay': weight_decay}
            ]
        else:
            # Phase 2: Chỉ chia làm 2 nhóm
            params = [
                {'params': clip_params, 'lr': base_lr / 3, 'weight_decay': weight_decay},
                {'params': new_params, 'lr': base_lr, 'weight_decay': weight_decay}
            ]
    else:
        # Phase 1: 1 nhóm duy nhất
        params = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 
                   'lr': base_lr, 'weight_decay': weight_decay}]
    
    # Sử dụng AdamW thay vì Adam
    optimizer = optim.AdamW(params)
    
    # Tính tổng số steps trong training
    total_steps = max_epochs * train_loader_len
    
    # OneCycleLR với warm-up và decay
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[param_group['lr'] for param_group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=0.1,  # warm-up 10% đầu
        div_factor=25,  # lr_initial = max_lr/25
        final_div_factor=1000,  # lr_final = lr_initial/1000
        anneal_strategy='cos'  # cosine annealing
    )
    
    return optimizer, scheduler

def train_phase(training_phase, model, train_loader, val_loader, criterion, device, 
                save_dir, resume='', max_epochs=10, patience=3):
    """Train mô hình cho một giai đoạn cụ thể, với early stopping."""
    logger.info(f"========== BẮT ĐẦU HUẤN LUYỆN PHASE {training_phase} ==========")
    
    # Thiết lập optimizer và scheduler phù hợp với phase
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, training_phase, train_loader.batch_size, max_epochs, len(train_loader)
    )
    
    # Thiết lập checkpoint path cho phase này
    checkpoint_path = os.path.join(save_dir, f"phase{training_phase}_checkpoint.pt")
    
    # Tải checkpoint nếu cần
    start_epoch, best_val_acc, loaded_phase = 0, 0.0, training_phase
    if resume:
        start_epoch, best_val_acc, loaded_phase = load_checkpoint(
            model, optimizer, scheduler, resume, device
        )
        # Nếu loaded_phase khác training_phase, bắt đầu lại
        if loaded_phase != training_phase:
            logger.warning(f"Loaded phase {loaded_phase} không khớp với training phase {training_phase}. Bắt đầu lại.")
            start_epoch, best_val_acc = 0, 0.0
    
    # Early stopping
    no_improve_epochs = 0
    
    # Gradient clipping theo phase
    clip_grad_norm = 1.0 if training_phase < 3 else 0.5
    
    # Training loop
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs} (Phase {training_phase})")
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad_norm
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Kiểm tra early stopping và save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # Lưu checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, checkpoint_path, 
            training_phase, is_best=is_best
        )
        
        # Early stopping
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered sau {no_improve_epochs} epochs không cải thiện.")
            break
    
    logger.info(f"========== KẾT THÚC HUẤN LUYỆN PHASE {training_phase} ==========")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc

def main():
    # Khai báo tham số
    json_path = "./Filtered_Frames_Script_sorted.json"  # Đường dẫn tới file JSON
    batch_size = 32  # Kích thước batch
    save_dir = 'clip_frame_finetune_v3'  # Thư mục lưu model
    seed = 42  # Seed cho tính lặp lại
    resume = ''  # Đường dẫn checkpoint để tiếp tục train (để trống nếu train từ đầu)
    phase1_epochs = 5  # Số epochs cho giai đoạn 1
    phase2_epochs = 5  # Số epochs cho giai đoạn 2
    phase3_epochs = 3  # Số epochs cho giai đoạn 3
    
    # Thiết lập cho mô hình PyTorch
    torch.set_float32_matmul_precision('high')  # Cải thiện chính xác phép nhân ma trận FP32
    # Buộc sử dụng FP32 cho các phép toán nhất định, giải quyết vấn đề Half/Float mismatch
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # Set seed cho tính lặp lại cao
    set_seed(seed)
    
    # Tạo thư mục lưu model
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ghi log thông tin GPU & PyTorch để debug
    if torch.cuda.is_available():
        logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
    
    # Khởi tạo transformers mạnh hơn cho data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((248, 248)),  # Resize lớn hơn 224 để random crop
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Khởi tạo dataset
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
    
    # DataLoaders
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

    # Lưu thông tin cấu hình training để tái tạo sau này
    config = {
        'json_path': json_path,
        'batch_size': batch_size,
        'save_dir': save_dir,
        'seed': seed,
        'resume': resume,
        'phase1_epochs': phase1_epochs,
        'phase2_epochs': phase2_epochs,
        'phase3_epochs': phase3_epochs
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Loss function cải tiến
    criterion = MultiModalLossV2(
        label_smoothing=0.1,
        weight_decay=1e-5,
        entropy_weight=0.01
    )
    
    # Nếu có resume từ checkpoint
    start_phase = 1
    if resume:
        # Load model và optimizer từ checkpoint để xác định giai đoạn bắt đầu
        temp_model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=1).to(device)
        temp_optimizer = optim.AdamW(temp_model.parameters())
        _, _, start_phase = load_checkpoint(temp_model, temp_optimizer, None, resume, device)
        
        # Xoá mô hình tạm, sẽ tạo mô hình thích hợp cho giai đoạn bên dưới
        del temp_model, temp_optimizer
        torch.cuda.empty_cache()
    
    # Huấn luyện theo 3 giai đoạn
    # Giai đoạn 1: Train chỉ các lớp mới, đóng băng CLIP
    if start_phase <= 1:
        model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=1).to(device)
        phase1_resume = resume if start_phase == 1 else ''
        best_val_acc = train_phase(
            1, model, train_loader, val_loader, criterion, device, 
            save_dir, phase1_resume, max_epochs=phase1_epochs
        )
        
        # Lưu best model cho phase 1
        phase1_best = os.path.join(save_dir, "best_model_phase1.pt")
        resume = os.path.join(save_dir, "phase1_checkpoint.pt")
    else:
        logger.info(f"Bỏ qua phase 1 vì đã hoàn thành")

    # Giai đoạn 2: Mở đóng băng các lớp cuối của CLIP
    if start_phase <= 2:
        # Tạo model mới hoặc load từ phase 1
        if start_phase == 2:
            model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=2).to(device)
            phase2_resume = resume
        else:
            # Load weights từ phase 1 và cập nhật training_phase
            model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=2).to(device)
            model.load_state_dict(torch.load(phase1_best))
            phase2_resume = ''
            
        best_val_acc = train_phase(
            2, model, train_loader, val_loader, criterion, device, 
            save_dir, phase2_resume, max_epochs=phase2_epochs, patience=2
        )
        
        # Lưu best model cho phase 2
        phase2_best = os.path.join(save_dir, "best_model_phase2.pt")
        resume = os.path.join(save_dir, "phase2_checkpoint.pt")
    else:
        logger.info(f"Bỏ qua phase 2 vì đã hoàn thành")
    
    # Giai đoạn 3: Fine-tune toàn bộ CLIP
    if start_phase <= 3:
        # Tạo model mới hoặc load từ phase 2
        if start_phase == 3:
            model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=3).to(device)
            phase3_resume = resume
        else:
            # Load weights từ phase 2 và cập nhật training_phase
            model = CLIPFineTuner(pretrained="ViT-B/32", training_phase=3).to(device)
            model.load_state_dict(torch.load(phase2_best))
            phase3_resume = ''
            
        best_val_acc = train_phase(
            3, model, train_loader, val_loader, criterion, device, 
            save_dir, phase3_resume, max_epochs=phase3_epochs, patience=1
        )
        
        # Lưu best model cho phase 3
        phase3_best = os.path.join(save_dir, "best_model_phase3.pt")
    else:
        logger.info(f"Bỏ qua phase 3 vì đã hoàn thành")
    
    logger.info("Huấn luyện hoàn tất.")
    logger.info(f"Model tốt nhất được lưu tại: {save_dir}")

if __name__ == '__main__':
    main() 