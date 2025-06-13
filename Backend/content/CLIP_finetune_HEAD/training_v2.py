import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from sklearn.model_selection import train_test_split

import clip
from clip.model import CLIP


def set_seed(seed=42):
    """Cài đặt seed cho tính tái hiện"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPDataset(Dataset):
    """Dataset cho CLIP training."""
    def __init__(self, image_paths, captions, categories, preprocess, tokenize_fn):
        self.image_paths = image_paths
        self.captions = captions
        self.categories = categories
        self.preprocess = preprocess
        self.tokenize_fn = tokenize_fn
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        category = self.categories[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {img_path}: {e}")
            # Tạo một ảnh trống nếu không đọc được
            img_tensor = torch.zeros(3, 224, 224)
        
        text_tensor = self.tokenize_fn([caption])[0]
        
        return img_tensor, text_tensor, category


class CLIPFineTuner:
    """Lớp quản lý quá trình fine-tune CLIP"""
    def __init__(self, 
                 data_path,
                 model_name="ViT-B/32", 
                 batch_size=32, 
                 lr=2e-5,
                 weight_decay=0.01,
                 num_epochs=10, 
                 save_dir="checkpoints",
                 random_seed=42):
        
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Cài đặt seed và device
        set_seed(random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Tạo thư mục lưu checkpoint
        os.makedirs(save_dir, exist_ok=True)
        
        # Các biến theo dõi trong quá trình huấn luyện
        self.best_val_acc = 0
        self.current_epoch = 0
        
    def prepare_data(self):
        """Chuẩn bị dữ liệu từ file JSON"""
        print("Đang đọc dữ liệu từ file JSON...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_paths = []
        captions = []
        categories = []
        
        for img_path, content in data.items():
            # Kiểm tra tệp tồn tại
            if os.path.exists(img_path):
                image_paths.append(img_path)
                captions.append(content["caption"])
                categories.append(content["category"])
            else:
                print(f"Không tìm thấy tệp: {img_path}")
        
        print(f"Đã đọc {len(image_paths)} ảnh hợp lệ")
        
        # Chia tập dữ liệu
        train_imgs, val_imgs, train_captions, val_captions, train_cats, val_cats = train_test_split(
            image_paths, captions, categories, test_size=0.2, stratify=categories, random_state=42
        )
        
        # Thống kê dữ liệu
        print(f"Dữ liệu huấn luyện: {len(train_imgs)} mẫu")
        print(f"Dữ liệu kiểm định: {len(val_imgs)} mẫu")
        
        # Phân phối các loại
        train_cat_counts = pd.Series(train_cats).value_counts()
        print("Phân phối loại trong tập huấn luyện:")
        for cat, count in train_cat_counts.items():
            print(f"  - {cat}: {count} ảnh ({count/len(train_cats)*100:.1f}%)")
        
        return (train_imgs, train_captions, train_cats), (val_imgs, val_captions, val_cats)
    
    def load_model(self):
        """Tải model CLIP"""
        print(f"Đang tải model CLIP {self.model_name}...")
        model, preprocess = clip.load(self.model_name, device=self.device)
        tokenize_fn = clip.tokenize
        
        return model, preprocess, tokenize_fn
    
    def create_dataloaders(self, train_data, val_data, preprocess, tokenize_fn):
        """Tạo các DataLoader"""
        train_imgs, train_captions, train_cats = train_data
        val_imgs, val_captions, val_cats = val_data
        
        train_dataset = CLIPDataset(
            train_imgs, train_captions, train_cats, preprocess, tokenize_fn
        )
        val_dataset = CLIPDataset(
            val_imgs, val_captions, val_cats, preprocess, tokenize_fn
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, epoch):
        """Huấn luyện một epoch"""
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for images, texts, _ in pbar:
            images = images.to(self.device)
            texts = texts.to(self.device)
            
            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)
            
            # Đây là ground truth identity matrix: text[i] matches với image[i]
            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
            
            # Contrastive loss (2 chiều: image->text và text->image)
            loss_i = F.cross_entropy(logits_per_image, ground_truth)
            loss_t = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_i + loss_t) / 2
            
            # Backward và optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Cập nhật loss
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, model, val_loader, epoch):
        """Đánh giá trên tập validation"""
        model.eval()
        val_loss = 0
        correct_img2txt = 0
        correct_txt2img = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            for images, texts, _ in pbar:
                images = images.to(self.device)
                texts = texts.to(self.device)
                batch_size = images.shape[0]
                total += batch_size
                
                # Forward pass
                logits_per_image, logits_per_text = model(images, texts)
                
                # Đây là ground truth identity matrix: text[i] matches với image[i]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)
                
                # Contrastive loss
                loss_i = F.cross_entropy(logits_per_image, ground_truth)
                loss_t = F.cross_entropy(logits_per_text, ground_truth)
                loss = (loss_i + loss_t) / 2
                val_loss += loss.item()
                
                # Tính accuracy
                pred_img2txt = logits_per_image.argmax(dim=1)
                pred_txt2img = logits_per_text.argmax(dim=1)
                
                correct_img2txt += (pred_img2txt == ground_truth).sum().item()
                correct_txt2img += (pred_txt2img == ground_truth).sum().item()
                
                # Hiển thị metrics
                img2txt_acc = correct_img2txt / total
                txt2img_acc = correct_txt2img / total
                avg_acc = (img2txt_acc + txt2img_acc) / 2
                
                pbar.set_postfix({
                    "loss": loss.item(),
                    "img2txt_acc": f"{img2txt_acc:.4f}",
                    "txt2img_acc": f"{txt2img_acc:.4f}",
                    "avg_acc": f"{avg_acc:.4f}"
                })
        
        avg_loss = val_loss / len(val_loader)
        img2txt_acc = correct_img2txt / total
        txt2img_acc = correct_txt2img / total
        avg_acc = (img2txt_acc + txt2img_acc) / 2
        
        print(f"Kết quả validation epoch {epoch+1}:")
        print(f"  - Val loss: {avg_loss:.4f}")
        print(f"  - Image->Text accuracy: {img2txt_acc:.4f}")
        print(f"  - Text->Image accuracy: {txt2img_acc:.4f}")
        print(f"  - Average accuracy: {avg_acc:.4f}")
        
        return avg_loss, img2txt_acc, txt2img_acc, avg_acc
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, avg_acc):
        """Lưu checkpoint"""
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "avg_acc": avg_acc,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Lưu checkpoint định kỳ
        checkpoint_path = os.path.join(self.save_dir, f"clip_ft_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Đã lưu checkpoint tại: {checkpoint_path}")
        
        # Lưu model tốt nhất (nếu có)
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            best_model_path = os.path.join(self.save_dir, "clip_ft_best.pt")
            torch.save(checkpoint, best_model_path)
            print(f"Đã lưu model tốt nhất tại: {best_model_path} (accuracy: {avg_acc:.4f})")
    
    def train(self):
        """Quy trình huấn luyện chính"""
        # Chuẩn bị dữ liệu
        train_data, val_data = self.prepare_data()
        
        # Tải model và tạo tiền xử lý dữ liệu
        model, preprocess, tokenize_fn = self.load_model()
        
        # Tạo dataloaders
        train_loader, val_loader = self.create_dataloaders(
            train_data, val_data, preprocess, tokenize_fn
        )
        
        # Cài đặt optimizer và scheduler
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # Bắt đầu training loop
        print(f"Bắt đầu huấn luyện trong {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validation
            val_loss, img2txt_acc, txt2img_acc, avg_acc = self.validate(model, val_loader, epoch)
            
            # Lưu checkpoint
            self.save_checkpoint(model, optimizer, scheduler, epoch, avg_acc)
            
            # Cập nhật scheduler
            scheduler.step()
            
            # Log thông tin epoch
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Avg acc: {avg_acc:.4f}")
        
        print(f"Huấn luyện hoàn tất! Model tốt nhất có accuracy: {self.best_val_acc:.4f}")
        return self.best_val_acc
    
    @staticmethod
    def infer_image(model, preprocess, image_path, text_queries, device):
        """Truy vấn hình ảnh với văn bản"""
        # Chuẩn bị ảnh
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Chuẩn bị văn bản
        text_tokens = clip.tokenize(text_queries).to(device)
        
        # Đưa qua model
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Tính similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        # Trả về kết quả
        results = []
        for i, query in enumerate(text_queries):
            score = similarity[0, i].item()
            results.append((query, score))
        
        # Sắp xếp kết quả theo điểm giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    json_path = "source_training/Filtered_Frames_Script_sorted.json"  # Đường dẫn tới file JSON
    model_name = "ViT-B/32"  # Model CLIP muốn fine-tune
    batch_size = 64  # Tăng batch size để có đủ negative samples cho contrastive learning
    lr = 2e-5  # Learning rate thấp hơn để fine-tune ổn định hơn
    epochs = 20  # Có thể cần train lâu hơn để học cả contrastive
    save_dir = "checkpoints/clip_contrastive_finetune"  # Thư mục lưu model
    seed = 42  # Random seed
    
    # Khởi tạo và chạy fine-tuner
    fine_tuner = CLIPFineTuner(
        data_path=json_path,
        model_name=model_name,
        batch_size=batch_size,
        lr=lr,
        num_epochs=epochs,
        save_dir=save_dir,
        random_seed=seed
    )
    
    # Bắt đầu huấn luyện
    fine_tuner.train()
    
    # Ví dụ về cách sử dụng model đã fine-tune để truy vấn
    print("\nVí dụ truy vấn với model đã fine-tune:")
    print("Bạn có thể load model tốt nhất và sử dụng nó để truy vấn như sau:")
    print(f"""
    # Load model đã fine-tune
    checkpoint = torch.load("{save_dir}/clip_ft_best.pt")
    model, preprocess = clip.load("{model_name}", device=device)
    model.load_state_dict(checkpoint["model"])
    
    # Truy vấn hình ảnh
    image_path = "path/to/image.jpg"
    text_queries = [
        "Cảnh bạo lực giữa hai người đàn ông",
        "Người đàn ông đấm người khác",
        "Hai người đang nói chuyện bình thường",
        "Cảnh sinh hoạt hằng ngày"
    ]
    
    results = CLIPFineTuner.infer_image(model, preprocess, image_path, text_queries, device)
    for query, score in results:
        print(f"{{query}}: {{score:.2%}}")
    """)


if __name__ == "__main__":
    main()
