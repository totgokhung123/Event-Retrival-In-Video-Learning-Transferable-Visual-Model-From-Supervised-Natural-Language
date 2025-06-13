import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

import open_clip
from open_clip import tokenize

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed()

class NSFWImageTextDataset(Dataset):
    def __init__(self, data_dir, caption_file, category, split="train", transform=None):
        """
        Args:
            data_dir: Root directory of data
            caption_file: JSON file containing captions
            category: "Violence" or "Sensitive"
            split: "train" or "val"
            transform: Transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.category = category
        self.split = split
        self.transform = transform
        
        # Load captions
        with open(caption_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)
            
        # Lọc ra các ảnh thuộc category và split cần thiết
        self.image_files = []
        self.image_captions = {}
        
        # Tạo pattern để tìm ảnh trong category
        category_lower = category.lower()
        
        for img_path, meta in self.captions.items():
            # Kiểm tra xem ảnh có thuộc category và split cần thiết không
            if (f"/{split}/{category}/" in img_path or f"/{split}/{category_lower}/" in img_path) and \
               meta["category"] == category:
                self.image_files.append(img_path)
                self.image_captions[img_path] = meta["caption"]
        
        print(f"Found {len(self.image_files)} valid image-caption pairs for {category} {split}")
        
        # Debug prints
        print(f"Data directory: {data_dir}")
        print(f"Category: {category}")
        print(f"Split: {split}")
        print(f"Number of total entries in JSON: {len(self.captions)}")
        print(f"First few image paths: {list(self.captions.keys())[:3]}")
        print(f"First few captions: {list(self.captions.values())[:3]}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Tạo một ảnh trống nếu không đọc được
            image = torch.zeros(3, 224, 224)
        
        text = self.image_captions[img_path]
        
        # Add category prefix to text for better context
        text = f"{self.category}: {text}"
        
        return image, text

class NSFWDataModule:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        script_dir = "/kaggle/input/script-data-merge/Script_Data_Merge_dir"
        
        # Caption files
        self.caption_files = {
            "Violence": {
                "train": os.path.join(script_dir, "caption_Violence_train.json"),
                "val": os.path.join(script_dir, "caption_Violence_val.json")
            },
            "Sensitive": {
                "train": os.path.join(script_dir, "caption_Sensitive_train.json"),
                "val": os.path.join(script_dir, "caption_Sensitive_val.json")
            }
        }
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def setup(self):
        # Create datasets
        violence_train = NSFWImageTextDataset(
            self.data_dir,
            self.caption_files["Violence"]["train"],
            "Violence",
            "train",
            self.train_transform
        )
        
        violence_val = NSFWImageTextDataset(
            self.data_dir,
            self.caption_files["Violence"]["val"],
            "Violence",
            "val",
            self.val_transform
        )
        
        sensitive_train = NSFWImageTextDataset(
            self.data_dir,
            self.caption_files["Sensitive"]["train"],
            "Sensitive",
            "train",
            self.train_transform
        )
        
        sensitive_val = NSFWImageTextDataset(
            self.data_dir,
            self.caption_files["Sensitive"]["val"],
            "Sensitive",
            "val",
            self.val_transform
        )
        
        # Combine datasets
        self.train_dataset = torch.utils.data.ConcatDataset([violence_train, sensitive_train])
        self.val_dataset = torch.utils.data.ConcatDataset([violence_val, sensitive_val])
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Hard Negative Mining
def get_hard_negatives(image_features, text_features, num_hard=3):
    with torch.no_grad():
        # Calculate similarity matrix
        sim_matrix = image_features @ text_features.T
        
        # Get positive pairs (diagonal)
        pos_indices = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        
        # For each row, get top-k most similar but not positive
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask[pos_indices, pos_indices] = False
        
        # Get top-k indices for each row (these are the hard negatives)
        hard_neg_values, hard_neg_indices = torch.topk(
            sim_matrix * mask.float(), k=num_hard, dim=1
        )
        
    return hard_neg_indices

# Trainer class
class CLIPFineTuner:
    def __init__(self, args):
        self.args = args
        
        # Setup data
        self.data_module = NSFWDataModule(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        self.data_module.setup()
        
        # Create model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.pretrained
        )
        self.model = self.model.to(args.device)
        
        # Set up tokenizer
        self.tokenizer = open_clip.get_tokenizer(args.model_name)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            args.num_epochs
        )
        
        # Mixed precision
        self.scaler = GradScaler() if args.use_amp else None
        
        # Logging
        self.global_step = 0
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Best validation loss
        self.best_val_loss = float('inf')
    
    def train(self):
        for epoch in range(self.args.num_epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Save model if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch}.pt")
            
            # Always save latest model
            self.save_model(f"last_model.pt")
            
            # Update learning rate
            self.scheduler.step()
    
    def save_model(self, filename):
        save_path = os.path.join(self.args.output_dir, filename)
        torch.save({
            'epoch': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(enumerate(self.data_module.train_dataloader()), desc=f"Epoch {epoch}")
        
        for step, batch in pbar:
            images, texts = batch
            images = images.to(self.args.device)
            
            # Tokenize texts
            texts = [text for text in texts]
            text_tokens = self.tokenizer(texts).to(self.args.device)
            
            # Forward pass with mixed precision
            if self.args.use_amp:
                with autocast():
                    # Get image and text features
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(text_tokens)
                    
                    # Normalize features
                    image_features = F.normalize(image_features, dim=1)
                    text_features = F.normalize(text_features, dim=1)
                    
                    # Hard negative mining if enabled
                    if self.args.hard_negatives:
                        hard_neg_indices = get_hard_negatives(image_features, text_features)
                    
                    # Compute similarity (logits)
                    logits_per_image = self.args.temperature * image_features @ text_features.T
                    logits_per_text = logits_per_image.T
                    
                    # Compute cross-entropy loss
                    labels = torch.arange(len(images)).to(self.args.device)
                    loss_img = F.cross_entropy(logits_per_image, labels)
                    loss_txt = F.cross_entropy(logits_per_text, labels)
                    loss = (loss_img + loss_txt) / 2
                
                # Backward pass with scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass without mixed precision
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)
                
                image_features = F.normalize(image_features, dim=1)
                text_features = F.normalize(text_features, dim=1)
                
                # Hard negative mining if enabled
                if self.args.hard_negatives:
                    hard_neg_indices = get_hard_negatives(image_features, text_features)
                
                # Compute similarity (logits)
                logits_per_image = self.args.temperature * image_features @ text_features.T
                logits_per_text = logits_per_image.T
                
                # Compute cross-entropy loss
                labels = torch.arange(len(images)).to(self.args.device)
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_txt = F.cross_entropy(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            self.global_step += 1
        
        # Log epoch loss
        avg_train_loss = train_loss / len(self.data_module.train_dataloader())
        print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for step, batch in enumerate(self.data_module.val_dataloader()):
                images, texts = batch
                images = images.to(self.args.device)
                
                # Tokenize texts
                texts = [text for text in texts]
                text_tokens = self.tokenizer(texts).to(self.args.device)
                
                # Forward pass
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)
                
                image_features = F.normalize(image_features, dim=1)
                text_features = F.normalize(text_features, dim=1)
                
                # Compute similarity (logits)
                logits_per_image = self.args.temperature * image_features @ text_features.T
                logits_per_text = logits_per_image.T
                
                # Compute cross-entropy loss
                labels = torch.arange(len(images)).to(self.args.device)
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_txt = F.cross_entropy(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
                
                # Update loss
                val_loss += loss.item()
        
        # Log epoch loss
        avg_val_loss = val_loss / len(self.data_module.val_dataloader())
        print(f"Epoch {epoch} val loss: {avg_val_loss:.4f}")
        
        # Evaluate retrieval metrics
        self.eval_retrieval(epoch)
        
        return avg_val_loss
    
    def eval_retrieval(self, epoch):
        """Evaluate retrieval metrics (recall@k)"""
        self.model.eval()
        
        # Get validation dataset and limit to at most 1000 samples for efficiency
        val_loader = DataLoader(
            self.data_module.val_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc="Computing embeddings"):
                images = images.to(self.args.device)
                
                # Get image embeddings
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=1)
                all_image_features.append(image_features)
                
                # Get text embeddings
                texts = [text for text in texts]
                text_tokens = self.tokenizer(texts).to(self.args.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=1)
                all_text_features.append(text_features)
        
        # Concatenate all features
        all_image_features = torch.cat(all_image_features)
        all_text_features = torch.cat(all_text_features)
        
        # Calculate similarity matrix
        similarity = all_image_features @ all_text_features.T
        
        # Calculate metrics for image-to-text retrieval
        i2t_retrieval = compute_retrieval_metrics(similarity)
        
        # Calculate metrics for text-to-image retrieval
        t2i_retrieval = compute_retrieval_metrics(similarity.T)
        
        print(f"Epoch {epoch} metrics:")
        print(f"  Image-to-Text retrieval: R@1: {i2t_retrieval['R1']:.2f}, R@5: {i2t_retrieval['R5']:.2f}, R@10: {i2t_retrieval['R10']:.2f}")
        print(f"  Text-to-Image retrieval: R@1: {t2i_retrieval['R1']:.2f}, R@5: {t2i_retrieval['R5']:.2f}, R@10: {t2i_retrieval['R10']:.2f}")

def compute_retrieval_metrics(similarity):
    """
    Compute recall@k for k=1,5,10
    """
    indices = torch.argsort(similarity, dim=1, descending=True)
    
    # Get position of ground truth
    gt_indices = torch.arange(similarity.shape[0]).to(similarity.device)
    ranks = torch.nonzero(indices == gt_indices.view(-1, 1), as_tuple=False)[:, 1] + 1
    
    # Calculate recall@k
    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    
    return {"R1": r1, "R5": r5, "R10": r10}

def main():
    # Thay thế argparse bằng class đơn giản để giữ cấu trúc dữ liệu
    class Args:
        def __init__(self):
            pass
    
    # Khai báo hardcode các tham số
    args = Args()
    args.data_dir = "/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge"
    args.model_name = "ViT-B-32"
    args.pretrained = "openai"
    args.batch_size = 64
    args.learning_rate = 2e-4
    args.weight_decay = 0.2
    args.num_epochs = 10
    args.temperature = 0.07
    args.use_amp = True
    args.hard_negatives = True
    args.num_workers = 4
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir = "./output"
    
    # In thông số ra màn hình
    print(f"Training with the following parameters:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model: {args.model_name} (pretrained from {args.pretrained})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Use AMP: {args.use_amp}")
    print(f"  Use hard negatives: {args.hard_negatives}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    print("-" * 40)
    
    # Start training
    trainer = CLIPFineTuner(args)
    trainer.train()

if __name__ == "__main__":
    main() 