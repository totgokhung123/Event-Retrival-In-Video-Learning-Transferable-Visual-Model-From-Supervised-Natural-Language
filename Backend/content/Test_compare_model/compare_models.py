import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import clip
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import json
import time
from pathlib import Path
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel, AutoTokenizer, FlavaProcessor, FlavaModel
from torch.utils.data import Dataset, DataLoader
import open_clip
import sys
from collections import defaultdict

# Import CLIPWithClassifier từ file clip_finetune_correct.py
sys.path.append('.')
try:
    from clip_finetune_correct import CLIPWithClassifier
except ImportError:
    print("WARNING: Không thể import CLIPWithClassifier từ clip_finetune_correct.py")
    print("Đảm bảo file clip_finetune_correct.py nằm trong cùng thư mục hoặc trong PYTHONPATH")

# Define encode_frames function for compatibility
def encode_frames_with_classifier(model, folder, batch_size=16, device=None):
    """Mã hóa frames bằng CLIPWithClassifier"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chuẩn bị transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Tạo dataset và dataloader
    dataset = TestFrameDataset(folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    frame_features = []
    frame_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đang mã hóa frames với CLIP fine-tuned"):
            images = batch['image'].to(device)
            
            # Trích xuất image features từ CLIP model bên trong CLIPWithClassifier
            features = model.clip_model.encode_image(images.float())
            features = features / features.norm(dim=1, keepdim=True)
            
            frame_features.append(features.cpu().numpy())
            frame_paths.extend(batch['path'])
    
    frame_features = np.vstack(frame_features)
    return frame_features, frame_paths

class TestFrameDataset(Dataset):
    """Dataset cho các frame trong một thư mục cụ thể"""
    def __init__(self, frame_dir, transform=None):
        self.frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.frame_paths.extend(glob.glob(os.path.join(frame_dir, ext)))
        self.frame_paths.sort()  # Sắp xếp để đảm bảo thứ tự ổn định
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Lỗi đọc ảnh {frame_path}: {str(e)}")
            image = torch.zeros(3, 224, 224)  # fallback
            
        return {
            'image': image,
            'path': frame_path
        }

class Flickr30kDataset:
    """Dataset cho Flickr30k để đánh giá image-text retrieval"""
    def __init__(self, images_dir, captions_path, split='test', test_size=1000):
        self.images_dir = images_dir
        
        # Parse caption file
        self.captions = defaultdict(list)
        self.image_ids = []
        self.all_captions = []
        
        print(f"Đang đọc captions từ {captions_path}...")
        import csv
        
        with open(captions_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            # Skip header if exists
            first_line = next(csv_reader)
            if not first_line[0].endswith('.jpg'):  # Kiểm tra nếu dòng đầu tiên là header
                header = first_line
                print(f"Đọc header: {header}")
            else:
                # Nếu dòng đầu không phải header, xử lý nó như dữ liệu
                image_name, _, caption = first_line
                image_id = image_name.replace('.jpg', '')
                
                if image_id not in self.image_ids:
                    self.image_ids.append(image_id)
                
                self.captions[image_id].append(caption)
                self.all_captions.append({
                    'image_id': image_id,
                    'caption': caption
                })
            
            # Đọc các dòng còn lại
            for row in csv_reader:
                if len(row) == 3:  # Đảm bảo đúng format: image_name,comment_number,comment
                    image_name, _, caption = row
                    image_id = image_name.replace('.jpg', '')
                    
                    if image_id not in self.image_ids:
                        self.image_ids.append(image_id)
                    
                    self.captions[image_id].append(caption)
                    self.all_captions.append({
                        'image_id': image_id,
                        'caption': caption
                    })
        
        # For test split, typically use first 1000 images
        if split == 'test':
            self.image_ids = self.image_ids[:test_size] if len(self.image_ids) > test_size else self.image_ids
            # Filter captions to only include test images
            self.all_captions = [item for item in self.all_captions if item['image_id'] in self.image_ids]
        
        print(f"Đã tải {len(self.image_ids)} ảnh và {len(self.all_captions)} captions cho {split} split")
        print(f"Số lượng caption trung bình cho mỗi ảnh: {len(self.all_captions)/len(self.image_ids) if len(self.image_ids) > 0 else 0:.1f}")
    
    def get_image_path(self, image_id):
        """Get full path to image from image_id"""
        return os.path.join(self.images_dir, f"{image_id}.jpg")

def load_clip_with_classifier(checkpoint_path, device=None):
    """Tải mô hình CLIPWithClassifier đã fine-tune"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tải mô hình CLIP gốc
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Tải checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Kiểm tra định dạng checkpoint và lấy config nếu có
        freeze_layers = 8  # Mặc định theo cấu hình trong clip_finetune_correct.py
        num_classes = 3  # Mặc định là 3 lớp: "Sensitive content", "Violence", "NonViolence"
        
        # Kiểm tra xem checkpoint có chứa config không
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                if 'freeze_layers' in config:
                    freeze_layers = config['freeze_layers']
                    print(f"Đã tìm thấy freeze_layers={freeze_layers} trong config của checkpoint")
                if 'num_classes' in config:
                    num_classes = config['num_classes']
                    print(f"Đã tìm thấy num_classes={num_classes} trong config của checkpoint")
            # Nếu không có config trong checkpoint, thử tìm thông tin từ file config.json cùng thư mục
            else:
                config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if 'freeze_layers' in config:
                                freeze_layers = config['freeze_layers']
                                print(f"Đã tìm thấy freeze_layers={freeze_layers} từ file config.json")
                            if 'num_classes' in config:
                                num_classes = config['num_classes']
                                print(f"Đã tìm thấy num_classes={num_classes} từ file config.json")
                    except:
                        pass
        
        # Khởi tạo CLIPWithClassifier với freeze_layers và num_classes phù hợp
        print(f"Khởi tạo CLIPWithClassifier với freeze_layers={freeze_layers}, num_classes={num_classes}")
        model = CLIPWithClassifier(clip_model, num_classes=num_classes, freeze_layers=freeze_layers)
        model = model.to(device)
        model = model.float()
        
        # Tải trọng số
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Đã tải model_state_dict từ checkpoint {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Đã tải trực tiếp state_dict từ {checkpoint_path}")
        
        model.eval()  # Đặt model ở chế độ evaluation
        return model, preprocess
    except Exception as e:
        print(f"Lỗi khi tải mô hình từ {checkpoint_path}: {str(e)}")
        # In traceback để dễ debug
        import traceback
        traceback.print_exc()
        raise

def encode_frames_with_classifier(model, frame_dir, batch_size=16, device=None):
    """Mã hóa frames bằng CLIPWithClassifier"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chuẩn bị transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Tạo dataset và dataloader
    dataset = TestFrameDataset(frame_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    frame_features = []
    frame_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đang mã hóa frames với CLIP fine-tuned"):
            images = batch['image'].to(device)
            
            # Trích xuất image features từ CLIP model bên trong CLIPWithClassifier
            features = model.clip_model.encode_image(images.float())
            features = features / features.norm(dim=1, keepdim=True)
            
            frame_features.append(features.cpu().numpy())
            frame_paths.extend(batch['path'])
    
    frame_features = np.vstack(frame_features)
    return frame_features, frame_paths

class ModelComparison:
    """Class chính để so sánh các model khác nhau trên tập test"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {self.device}")
        
        # Tải dataset Flickr30k thay vì đọc file Excel test
        print(f"Đang tải Flickr30k dataset từ {args.flickr_images_dir} và {args.flickr_captions_path}...")
        self.dataset = Flickr30kDataset(
            images_dir=args.flickr_images_dir,
            captions_path=args.flickr_captions_path,
            split='test',
            test_size=args.test_size if hasattr(args, 'test_size') else 1000
        )
        
        # Khởi tạo các model
        self.models = {}
        
        # Chỉ tải tất cả các mô hình ngay lập tức nếu không sử dụng memory-efficient mode
        if not hasattr(args, 'memory_efficient') or not args.memory_efficient:
            self.load_models()
            
            # Chuẩn bị transforms
            self.transforms = {}
            self.setup_transforms()
            
            # Thêm projection layer cho ViT -> CLIP compatibility
            self.setup_projection_layers()
        
        # Chuẩn bị thư mục kết quả
        os.makedirs(args.output_dir, exist_ok=True)

    def load_single_model(self, model_name):
        """Tải một model cụ thể"""
        print(f"Đang tải model {model_name}...")
        
        if model_name == 'CLIP-Finetune' and self.args.use_finetuned:
            try:
                # Tải model từ checkpoint với cấu trúc CLIPWithClassifier
                print(f"Đang tải model fine-tuned từ {self.args.finetuned_model_path}...")
                model, preprocess = load_clip_with_classifier(
                    self.args.finetuned_model_path, 
                    self.device
                )
                self.models['CLIP-Finetune'] = model
                print("Đã tải CLIPWithClassifier thành công.")
                
                # In thông tin cấu trúc model để debug
                clip_model = model.clip_model
                print(f"CLIP model: {clip_model.__class__.__name__}")
                print(f"Visual model: {clip_model.visual.__class__.__name__}")
                
                # Kiểm tra số lượng tham số và trạng thái requires_grad
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Tổng số tham số: {total_params:,}, Tham số trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")
            
            except Exception as e:
                print(f"ERROR: Không thể tải model fine-tuned từ {self.args.finetuned_model_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        elif model_name == 'CLIP-ViT-B/32' and self.args.use_clip:
            print("Đang tải CLIP gốc...")
            model, _ = clip.load("ViT-B/32", device=self.device)
            self.models['CLIP-ViT-B/32'] = model
            
        elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K' and self.args.use_openclip:
            print("Đang tải OpenCLIP (LAION)...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', 
                pretrained='laion2b_s32b_b79k',
                device=self.device
            )
            tokenizer = open_clip.get_tokenizer('ViT-H-14')
            self.models['CLIP-ViT-H-14-laion2B-s32B-b79K'] = {
                'model': model,
                'tokenizer': tokenizer,
                'preprocess': preprocess
            }
            
        elif model_name == 'Flava' and self.args.use_flava:
            print("Đang tải Flava cho image-text retrieval...")
            # Sử dụng mô hình Flava cho retrieval
            processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
            
            # Để tương thích với cấu trúc code hiện tại
            self.models['Flava'] = {
                'model': model,
                'processor': processor
            }
            print("Đã tải Flava thành công")

        elif model_name == 'ViT-B/16-224' and self.args.use_vit:
            print("Đang tải ViT...")
            model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.models['ViT-B/16-224'] = model
            
        # Chuẩn bị transforms cho model đã tải
        self.setup_transforms_for_model(model_name)
        
        # Chuẩn bị projection layers nếu cần
        if model_name == 'ViT-B/16-224' and self.args.use_vit:
            self.setup_projection_layers()
            
        return model_name in self.models

    def load_models(self):
        """Tải tất cả các model cần thiết"""
        print("Đang tải các model...")
        
        # 1. Model CLIP fine-tune
        if self.args.use_finetuned:
            self.load_single_model('CLIP-Finetune')
            
        # 2. CLIP gốc
        if self.args.use_clip:
            self.load_single_model('CLIP-ViT-B/32')
            
        # 3. OpenCLIP LAION (thay thế BLIP-2)
        if self.args.use_openclip:
            self.load_single_model('CLIP-ViT-H-14-laion2B-s32B-b79K')
            
        # 4. Flava
        if self.args.use_flava:
            self.load_single_model('Flava')

        # 5. ViT
        if self.args.use_vit:
            self.load_single_model('ViT-B/16-224')

    def setup_transforms(self):
        """Chuẩn bị transform cho từng model"""
        # CLIP và CLIP fine-tuned
        self.transforms['clip'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # OpenCLIP sử dụng preprocess riêng (đã tạo trong load_models)
        
        # Flava sử dụng processor riêng
        
        # ViT
        self.transforms['ViT-B/16-224'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def setup_transforms_for_model(self, model_name):
        """Chuẩn bị transform cho một model cụ thể"""
        if not hasattr(self, 'transforms'):
            self.transforms = {}
            
        if model_name in ['CLIP-Finetune', 'CLIP-ViT-B/32']:
            self.transforms['clip'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
        elif model_name == 'ViT-B/16-224':
            self.transforms['ViT-B/16-224'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
    def setup_projection_layers(self):
        """Khởi tạo các lớp projection để chuyển đổi kích thước embedding giữa các model"""
        # ViT có embedding là 768 chiều, CLIP có embedding là 512 chiều
        # Cần chuyển đổi từ 768 -> 512 để có thể so sánh
        self.vit_projection = nn.Linear(768, 512).to(self.device)  # từ ViT -> CLIP dim
        
        # Khởi tạo projection layer với phương pháp tốt hơn để tương thích với CLIP
        if self.args.use_clip and 'CLIP-ViT-B/32' in self.models:
            print("Khởi tạo projection layer cho ViT tương thích với CLIP...")
            # Tạo batch ảnh ngẫu nhiên để lấy mẫu embeddings từ cả hai model
            random_input = torch.randn(16, 3, 224, 224).to(self.device)
            
            # Lấy embeddings từ ViT
            if self.args.use_vit and 'ViT-B/16-224' in self.models:
                with torch.no_grad():
                    vit_outputs = self.models['ViT-B/16-224'](random_input)
                    vit_embeddings = vit_outputs.last_hidden_state[:, 0]  # CLS token embeddings
                
                # Lấy embeddings từ CLIP
                with torch.no_grad():
                    clip_embeddings = self.models['CLIP-ViT-B/32'].encode_image(random_input)
                    clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=1, keepdim=True)
                
                # Khởi tạo trọng số dựa trên mối tương quan giữa hai embedding spaces
                # Sử dụng pseudoinverse để tìm ma trận chuyển đổi tuyến tính tốt từ vit_embeddings -> clip_embeddings
                vit_flat = vit_embeddings.cpu().numpy()
                clip_flat = clip_embeddings.cpu().numpy()
                
                # Projection matrix using least squares solution
                proj_weights, residuals, rank, s = np.linalg.lstsq(vit_flat, clip_flat, rcond=None)
                
                # Áp dụng trọng số đã tìm được vào linear layer
                with torch.no_grad():
                    self.vit_projection.weight.copy_(torch.tensor(proj_weights.T, dtype=torch.float32))
                    # Bias term can be estimated as mean difference after projection
                    projected = np.dot(vit_flat, proj_weights)
                    bias = np.mean(clip_flat - projected, axis=0)
                    self.vit_projection.bias.copy_(torch.tensor(bias, dtype=torch.float32))
                
                print("Đã khởi tạo projection layer cho ViT với phương pháp least squares")
            else:
                # Fallback - Khởi tạo ngẫu nhiên nếu không có cả hai model
                with torch.no_grad():
                    nn.init.normal_(self.vit_projection.weight, std=0.02)
                    nn.init.zeros_(self.vit_projection.bias)
        else:
            # Khởi tạo trọng số ngẫu nhiên nếu không có model CLIP gốc
            with torch.no_grad():
                nn.init.normal_(self.vit_projection.weight, std=0.02)
                nn.init.zeros_(self.vit_projection.bias)

    def encode_frames_original_clip(self, frame_dir, batch_size=16):
        """Mã hóa frames bằng CLIP gốc"""
        dataset = TestFrameDataset(frame_dir, self.transforms['clip'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        frame_features = []
        frame_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Đang mã hóa frames với CLIP gốc"):
                images = batch['image'].to(self.device)
                features = self.models['CLIP-ViT-B/32'].encode_image(images)
                features = features / features.norm(dim=1, keepdim=True)
                
                frame_features.append(features.cpu().numpy())
                frame_paths.extend(batch['path'])
        
        frame_features = np.vstack(frame_features)
        return frame_features, frame_paths

    def encode_frames_openclip(self, frame_dir, batch_size=8):
        """Mã hóa frames bằng OpenCLIP (LAION)"""
        model = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model']
        preprocess = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['preprocess']
        
        frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_paths.extend(glob.glob(os.path.join(frame_dir, ext)))
        frame_paths.sort()
        
        frame_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(frame_paths), batch_size), desc="Đang mã hóa frames với OpenCLIP"):
                batch_paths = frame_paths[i:i+batch_size]
                batch_images = []
                for path in batch_paths:
                    image = Image.open(path).convert('RGB')
                    image = preprocess(image)
                    batch_images.append(image)
                
                # Chuyển batch thành tensor
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Lấy features từ model
                image_features = model.encode_image(batch_tensor)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                frame_features.append(image_features.cpu().numpy())
        
        frame_features = np.vstack(frame_features)
        return frame_features, frame_paths

    def encode_frames_flava(self, frame_dir, batch_size=8):
        """Mã hóa frames bằng Flava"""
        processor = self.models['Flava']['processor']
        model = self.models['Flava']['model']
        
        frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_paths.extend(glob.glob(os.path.join(frame_dir, ext)))
        frame_paths.sort()
        
        frame_features = []
        
        with torch.no_grad():
            for path in tqdm(frame_paths, desc="Đang mã hóa frames với Flava"):
                try:
                    # Xử lý từng ảnh một để tránh lỗi kích thước tensor
                    image = Image.open(path).convert('RGB')
                    
                    # Sử dụng processor của Flava để xử lý ảnh
                    inputs = processor(images=[image], return_tensors="pt").to(self.device)
                    
                    # Lấy features sử dụng mô hình Flava
                    outputs = model(
                        pixel_values=inputs["pixel_values"],
                        return_dict=True
                    )
                    
                    # Lấy image embeddings từ Flava
                    image_features = outputs.image_embeddings
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    
                    frame_features.append(image_features.cpu().numpy())
                except Exception as e:
                    print(f"CẢNH BÁO: Không thể xử lý ảnh {path}: {str(e)}")
                    feature_dim = 768  # Flava image encoder output dimension
                    frame_features.append(np.zeros((1, feature_dim)))
        
        frame_features = np.vstack(frame_features)
        return frame_features, frame_paths

    def encode_text_flava(self, caption):
        """Mã hóa text bằng Flava"""
        processor = self.models['Flava']['processor']
        model = self.models['Flava']['model']
        
        try:
            with torch.no_grad():
                inputs = processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
                
                # Sử dụng mô hình Flava để lấy text embeddings
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                
                # Lấy text embeddings từ Flava
                text_features = outputs.text_embeddings
                
                # Normalize features
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"CẢNH BÁO: Không thể xử lý caption '{caption[:30]}...': {str(e)}")
            feature_dim = 768  # Flava text encoder output dimension
            return np.zeros((1, feature_dim))

    def create_image_dataset(self, image_paths, model_name):
        """Tạo dataset từ danh sách đường dẫn ảnh phù hợp với model"""
        class SimpleImageDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
                
            def __len__(self):
                return len(self.paths)
                
            def __getitem__(self, idx):
                path = self.paths[idx]
                try:
                    image = Image.open(path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                except Exception as e:
                    print(f"Lỗi đọc ảnh {path}: {str(e)}")
                    image = torch.zeros(3, 224, 224)  # fallback
                return image
        
        # Chọn transform phù hợp với model
        if model_name == 'CLIP-Finetune' or model_name == 'CLIP-ViT-B/32':
            transform = self.transforms['clip']
        elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K':
            transform = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['preprocess']
        elif model_name == 'Flava':
            # Flava sử dụng processor riêng, sẽ xử lý trong process_image_batch
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        elif model_name == 'ViT-B/16-224':
            transform = self.transforms['ViT-B/16-224']
        else:
            transform = self.transforms['clip']  # fallback
            
        return SimpleImageDataset(image_paths, transform)

    def encode_frames_vit(self, frame_dir, batch_size=16):
        """Mã hóa frames bằng ViT"""
        dataset = TestFrameDataset(frame_dir, self.transforms['ViT-B/16-224'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        frame_features = []
        frame_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Đang mã hóa frames với ViT"):
                images = batch['image'].to(self.device)
                outputs = self.models['ViT-B/16-224'](images)
                
                # Lấy đặc trưng CLS token
                features = outputs.last_hidden_state[:, 0]
                
                # Chuyển đổi kích thước từ 768 -> 512 để tương thích với CLIP text embedding
                features = self.vit_projection(features)
                
                # Thêm chuẩn hóa scale và shift để tương thích tốt hơn với không gian nhúng của CLIP
                # CLIP features thường có scale khoảng 0.025-0.03 sau khi normalize
                # Sử dụng phương pháp chuẩn hóa thống kê để đảm bảo feature space tương đồng
                features = self.apply_statistical_normalization(features)
                
                # Normalize feature vector
                features = features / features.norm(dim=1, keepdim=True)
                
                frame_features.append(features.cpu().numpy())
                frame_paths.extend(batch['path'])
        
        frame_features = np.vstack(frame_features)
        return frame_features, frame_paths

    def apply_statistical_normalization(self, features):
        """Áp dụng chuẩn hóa thống kê để đảm bảo tương thích tốt hơn giữa các embedding spaces"""
        # Nếu chưa tính toán thông số chuẩn hóa, tính từ CLIP model để đảm bảo tương thích
        if not hasattr(self, 'clip_mean') or not hasattr(self, 'clip_std'):
            if self.args.use_clip and 'CLIP-ViT-B/32' in self.models:
                print("Tính toán thông số chuẩn hóa thống kê từ CLIP...")
                # Tạo batch ngẫu nhiên để tính thông số
                random_inputs = torch.randn(64, 3, 224, 224).to(self.device)
                
                with torch.no_grad():
                    # Lấy CLIP embeddings
                    clip_features = self.models['CLIP-ViT-B/32'].encode_image(random_inputs)
                    clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                    
                    # Tính thông số thống kê
                    self.clip_mean = torch.mean(clip_features, dim=0, keepdim=True)
                    self.clip_std = torch.std(clip_features, dim=0, keepdim=True)
                    
                    # Đưa về CPU để giảm sử dụng VRAM
                    self.clip_mean = self.clip_mean.cpu()
                    self.clip_std = self.clip_std.cpu()
                    
                    print("Đã tính xong thông số chuẩn hóa từ CLIP")
            else:
                # Nếu không có CLIP, sử dụng giá trị mặc định hợp lý cho đặc trưng đã normalized
                self.clip_mean = torch.zeros(1, 512)
                self.clip_std = torch.ones(1, 512) * 0.03  # Approximate scale for CLIP features
        
        # Apply normalization: z-score transform and then scale to match CLIP distribution
        features_mean = torch.mean(features, dim=0, keepdim=True)
        features_std = torch.std(features, dim=0, keepdim=True) + 1e-8  # Avoid division by zero
        
        # Z-score normalize
        features = (features - features_mean) / features_std
        
        # Scale and shift to match CLIP statistics
        features = features * self.clip_std.to(self.device) + self.clip_mean.to(self.device)
        
        return features

    def encode_text_original_clip(self, caption):
        """Mã hóa text bằng CLIP gốc"""
        with torch.no_grad():
            text_tokens = clip.tokenize([caption]).to(self.device)
            text_features = self.models['CLIP-ViT-B/32'].encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()

    def encode_text_openclip(self, caption):
        """Mã hóa text bằng OpenCLIP (LAION)"""
        model = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model']
        tokenizer = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['tokenizer']
        
        with torch.no_grad():
            text_tokens = tokenizer([caption]).to(self.device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        return text_features.cpu().numpy()

    def encode_text_flava(self, caption):
        """Mã hóa text bằng Flava"""
        processor = self.models['Flava']['processor']
        model = self.models['Flava']['model']
        
        try:
            with torch.no_grad():
                inputs = processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
                
                # Sử dụng mô hình Flava để lấy text embeddings
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                
                # Lấy text embeddings từ Flava
                text_features = outputs.text_embeddings
                
                # Normalize features
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"CẢNH BÁO: Không thể xử lý caption '{caption[:30]}...': {str(e)}")
            feature_dim = 768  # Flava text encoder output dimension
            return np.zeros((1, feature_dim))

    def calculate_metrics(self, similarities, ground_truth_indices):
        """Tính toán các metrics từ similarity scores và ground-truth indices"""
        ranks = []
        for gt_indices in ground_truth_indices:
            gt_ranks = []
            for gt_idx in gt_indices:
                rank = np.where(np.argsort(-similarities) == gt_idx)[0][0] + 1
                gt_ranks.append(rank)
            ranks.append(min(gt_ranks))  # Lấy rank tốt nhất trong các ground-truth
        
        ranks = np.array(ranks)
        
        metrics = {
            'R@1': (ranks <= 1).mean(),
            'R@5': (ranks <= 5).mean(),
            'R@10': (ranks <= 10).mean(),
            'MRR': (1.0 / ranks).mean(),
            'Median_Rank': np.median(ranks),
            'Mean_Rank': np.mean(ranks)
        }
        
        # Tính Precision@K
        for k in [1, 5, 10]:
            precision_sum = 0
            for gt_indices in ground_truth_indices:
                top_k_indices = np.argsort(-similarities)[:k]
                hits = sum(1 for idx in top_k_indices if idx in gt_indices)
                precision_sum += hits / k
            metrics[f'P@{k}'] = precision_sum / len(ground_truth_indices)
        
        return metrics, ranks

    def format_metrics_table(self, metrics, title=None):
        """Format metrics as a nicely aligned table with borders"""
        if title:
            print(f"\n{title}")
        
        # Define metrics order and column widths
        metrics_order = ['R@1', 'R@5', 'R@10', 'MRR', 'Median_Rank', 'Mean_Rank', 'rsum']
        col_width_metric = 12
        col_width_value = 10
        
        # Create border line
        border = "+" + "-" * (col_width_metric + 2) + "+" + "-" * (col_width_value + 2) + "+"
        
        # Print top border
        print(border)
        
        # Print header
        print(f"| {'Metric'.ljust(col_width_metric)} | {'Value'.ljust(col_width_value)} |")
        
        # Print header-data separator
        print(border)
        
        # Print metrics in specified order if present, then any remaining metrics
        for metric in metrics_order:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric == 'rsum':
                        value_str = f"{value:.2f}"
                    elif metric.startswith(('R@', 'MRR')):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                print(f"| {metric.ljust(col_width_metric)} | {value_str.ljust(col_width_value)} |")
        
        # Print any remaining metrics not in the predefined order
        for metric, value in metrics.items():
            if metric not in metrics_order:
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                print(f"| {metric.ljust(col_width_metric)} | {value_str.ljust(col_width_value)} |")
        
        # Print bottom border
        print(border)

    def format_combined_metrics_table(self, t2i_metrics, i2t_metrics, mean_metrics, processing_time=None):
        """Format a combined table with all metrics side by side"""
        print("\n========== SUMMARY OF ALL METRICS ==========")
        
        # Define metrics order and column widths
        metrics_order = ['R@1', 'R@5', 'R@10', 'MRR', 'Median_Rank', 'Mean_Rank']
        col_width_metric = 12
        col_width_value = 10
        
        # Create border line
        border = "+" + "-" * (col_width_metric + 2) + "+" + "-" * (col_width_value + 2) + "+" + "-" * (col_width_value + 2) + "+" + "-" * (col_width_value + 2) + "+"
        
        # Print top border
        print(border)
        
        # Print header
        print(f"| {'Metric'.ljust(col_width_metric)} | {'Text->Image'.ljust(col_width_value)} | {'Image->Text'.ljust(col_width_value)} | {'Mean'.ljust(col_width_value)} |")
        
        # Print header-data separator
        print(border)
        
        # Print metrics rows
        for metric in metrics_order:
            t2i_value = t2i_metrics.get(metric, "N/A")
            i2t_value = i2t_metrics.get(metric, "N/A")
            mean_value = mean_metrics.get(metric, "N/A")
            
            # Format values
            if isinstance(t2i_value, float):
                if metric.startswith(('R@', 'MRR')):
                    t2i_str = f"{t2i_value:.4f}"
                else:
                    t2i_str = f"{t2i_value:.2f}"
            else:
                t2i_str = str(t2i_value)
                
            if isinstance(i2t_value, float):
                if metric.startswith(('R@', 'MRR')):
                    i2t_str = f"{i2t_value:.4f}"
                else:
                    i2t_str = f"{i2t_value:.2f}"
            else:
                i2t_str = str(i2t_value)
                
            if isinstance(mean_value, float):
                if metric.startswith(('R@', 'MRR')):
                    mean_str = f"{mean_value:.4f}"
                else:
                    mean_str = f"{mean_value:.2f}"
            else:
                mean_str = str(mean_value)
            
            print(f"| {metric.ljust(col_width_metric)} | {t2i_str.ljust(col_width_value)} | {i2t_str.ljust(col_width_value)} | {mean_str.ljust(col_width_value)} |")
        
        # Add rsum as the last row if it exists in mean_metrics
        if 'rsum' in mean_metrics:
            rsum_value = mean_metrics['rsum']
            rsum_str = f"{rsum_value:.2f}" if isinstance(rsum_value, float) else str(rsum_value)
            print(border)
            print(f"| {'rsum'.ljust(col_width_metric)} | {''.ljust(col_width_value)} | {''.ljust(col_width_value)} | {rsum_str.ljust(col_width_value)} |")
        
        # Print bottom border
        print(border)
        
        # Print processing time if provided
        if processing_time is not None:
            print(f"Total processing time: {processing_time:.2f} seconds")

    def evaluate_model(self, model_name):
        """Đánh giá một model cụ thể trên Flickr30k dataset"""
        print(f"\n--- Đánh giá model: {model_name} trên Flickr30k dataset ---")
        
        results = {}
        processing_times = []
        
        # Phần 1: Encode toàn bộ ảnh trong dataset test - Cải thiện với xử lý batch
        print(f"Đang encode tất cả {len(self.dataset.image_ids)} ảnh test...")
        start_time = time.time()
        
        # Cải thiện: Xử lý batch thay vì từng ảnh một để tăng hiệu suất
        batch_size = self.args.batch_size if hasattr(self.args, 'batch_size') else 32
        
        # Tạo danh sách đường dẫn ảnh
        image_paths = [self.dataset.get_image_path(image_id) for image_id in self.dataset.image_ids]
        
        # Tạo DataLoader để xử lý batch hiệu quả
        image_dataset = self.create_image_dataset(image_paths, model_name)
        image_dataloader = DataLoader(
            image_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True  # Giúp tăng tốc chuyển dữ liệu lên GPU
        )
        
        # Encode theo batch
        image_features = []
        
        with torch.no_grad():
            for batch in tqdm(image_dataloader, desc=f"Encoding ảnh với {model_name} theo batch"):
                # Xử lý khác nhau tùy theo model
                features = self.process_image_batch(batch, model_name)
                image_features.append(features.cpu().numpy())
        
        # Kết hợp tất cả image features
        image_features = np.vstack(image_features)
        encode_image_time = time.time() - start_time
        print(f"Đã encode {len(image_paths)} ảnh trong {encode_image_time:.2f} giây")
        
        # Phần 2: Encode tất cả captions trong dataset test
        print(f"Đang encode tất cả {len(self.dataset.all_captions)} captions...")
        start_time = time.time()
        
        # Cải thiện: Xử lý caption theo batch thay vì từng caption
        captions = [item['caption'] for item in self.dataset.all_captions]
        caption_ids = [item['image_id'] for item in self.dataset.all_captions]
        
        # Encode theo batch
        text_features = []
        processed_caption_ids = []  # Track which captions were successfully processed
        
        if model_name == 'Flava':
            # For FLAVA, process one caption at a time to avoid tensor size mismatches
            for i, caption in enumerate(tqdm(captions, desc=f"Encoding captions với {model_name}")):
                try:
                    feature = self.process_text_batch([caption], model_name)
                    text_features.append(feature)
                    processed_caption_ids.append(caption_ids[i])
                except Exception as e:
                    print(f"Bỏ qua caption {i} do lỗi: {str(e)}")
                    continue
            
            # Stack the features
            if text_features:
                text_features = np.vstack(text_features)
            else:
                print("CẢNH BÁO: Không có caption nào được xử lý thành công!")
                return {}
        else:
            # For other models, process in batches
            batch_size = 32  # Text thường nhỏ hơn nên có thể xử lý batch lớn hơn
            
            for i in tqdm(range(0, len(captions), batch_size), desc=f"Encoding captions với {model_name} theo batch"):
                batch_captions = captions[i:i+batch_size]
                features = self.process_text_batch(batch_captions, model_name)
                text_features.append(features)
                processed_caption_ids.extend(caption_ids[i:i+batch_size])
                
            # Kết hợp tất cả text features
            text_features = np.vstack(text_features)
        
        encode_text_time = time.time() - start_time
        print(f"Đã encode {len(processed_caption_ids)} captions trong {encode_text_time:.2f} giây")
        
        # Phần 3: Đánh giá text-to-image retrieval (t2i) - Standard Flickr30k evaluation
        print("\nĐánh giá text-to-image retrieval...")
        start_time = time.time()
        
        # Tính toán similarity matrix toàn bộ một lần để tăng hiệu suất
        similarity_matrix = np.dot(image_features, text_features.T)
        
        # Map từ image_id đến index trong image_features
        image_id_to_index = {image_id: i for i, image_id in enumerate(self.dataset.image_ids)}
        
        t2i_ranks = []
        for i, image_id in enumerate(tqdm(processed_caption_ids, desc="Text-to-image retrieval")):
            # Tìm index của ground-truth image
            if image_id in image_id_to_index:
                gt_idx = image_id_to_index[image_id]
                
                # Lấy similarity của caption này với tất cả ảnh
                similarities = similarity_matrix[:, i]
                
                # Lấy rank của ground-truth image
                sorted_indices = np.argsort(-similarities)
                rank = np.where(sorted_indices == gt_idx)[0][0] + 1
                t2i_ranks.append(rank)
        
        # Tính metrics cho text-to-image retrieval
        t2i_ranks = np.array(t2i_ranks)
        t2i_metrics = {
            'R@1': (t2i_ranks <= 1).mean(),
            'R@5': (t2i_ranks <= 5).mean(),
            'R@10': (t2i_ranks <= 10).mean(),
            'MRR': (1.0 / t2i_ranks).mean(),
            'Median_Rank': np.median(t2i_ranks),
            'Mean_Rank': np.mean(t2i_ranks)
        }
        
        t2i_time = time.time() - start_time
        results['t2i'] = t2i_metrics
        
        # Hiển thị kết quả dạng bảng thay vì từng dòng
        self.format_metrics_table(t2i_metrics, "Text-to-image retrieval metrics:")
        
        # Phần 4: Đánh giá image-to-text retrieval (i2t) - Standard Flickr30k evaluation
        print("\nĐánh giá image-to-text retrieval...")
        start_time = time.time()
        
        # Tạo mapping từ image_id đến indices của tất cả caption tương ứng
        image_id_to_caption_indices = defaultdict(list)
        for i, item_id in enumerate(processed_caption_ids):
            image_id_to_caption_indices[item_id].append(i)
        
        # Evaluate image-to-text sử dụng similarity matrix đã tính
        i2t_ranks = []
        for j, image_id in enumerate(tqdm(self.dataset.image_ids, desc="Image-to-text retrieval")):
            # Lấy indices của tất cả caption cho image này
            gt_caption_indices = image_id_to_caption_indices[image_id]
            
            if not gt_caption_indices:
                print(f"WARNING: Không tìm thấy caption nào cho image {image_id}")
                continue
            
            # Lấy similarity của image này với tất cả caption
            similarities = similarity_matrix[j, :]
            
            # Lấy ranks của tất cả ground-truth captions và lấy rank tốt nhất
            sorted_indices = np.argsort(-similarities)
            caption_ranks = [np.where(sorted_indices == idx)[0][0] + 1 for idx in gt_caption_indices]
            best_rank = min(caption_ranks)  # Standard Flickr30k evaluation lấy rank tốt nhất trong 5 captions
            
            i2t_ranks.append(best_rank)
        
        # Tính metrics cho image-to-text retrieval
        i2t_ranks = np.array(i2t_ranks)
        i2t_metrics = {
            'R@1': (i2t_ranks <= 1).mean(),
            'R@5': (i2t_ranks <= 5).mean(),
            'R@10': (i2t_ranks <= 10).mean(),
            'MRR': (1.0 / i2t_ranks).mean(),
            'Median_Rank': np.median(i2t_ranks),
            'Mean_Rank': np.mean(i2t_ranks)
        }
        
        i2t_time = time.time() - start_time
        results['i2t'] = i2t_metrics
        
        # Hiển thị kết quả dạng bảng
        self.format_metrics_table(i2t_metrics, "Image-to-text retrieval metrics:")
        
        # Phần 5: Tính metrics trung bình (rsum - sum of recall metrics)
        mean_metrics = {}
        for metric in ['R@1', 'R@5', 'R@10', 'MRR', 'Median_Rank', 'Mean_Rank']:
            mean_metrics[metric] = (t2i_metrics[metric] + i2t_metrics[metric]) / 2
        
        # Tính rsum (sum of recall) là tiêu chuẩn trong Flickr30k evaluation
        mean_metrics['rsum'] = t2i_metrics['R@1'] + t2i_metrics['R@5'] + t2i_metrics['R@10'] + \
                              i2t_metrics['R@1'] + i2t_metrics['R@5'] + i2t_metrics['R@10']
        
        results['mean'] = mean_metrics
        total_processing_time = encode_image_time + encode_text_time + t2i_time + i2t_time
        results['processing_time'] = total_processing_time
        
        # Hiển thị kết quả dạng bảng
        self.format_metrics_table(mean_metrics, "Mean metrics across both directions:")
        
        # Hiển thị bảng tổng hợp
        self.format_combined_metrics_table(t2i_metrics, i2t_metrics, mean_metrics, total_processing_time)
        
        return results
        
    def process_image_batch(self, batch, model_name):
        """Xử lý một batch ảnh theo từng loại model"""
        # Error handling
        try:
            batch = batch.to(self.device)
            
            if model_name == 'CLIP-Finetune':
                # CLIPWithClassifier
                with torch.no_grad():
                    # Đảm bảo sử dụng kiểu dữ liệu float32 với đầu vào
                    batch = batch.float()
                    # Sử dụng encode_image từ CLIP model bên trong
                    features = self.models['CLIP-Finetune'].clip_model.encode_image(batch)
                    # Đảm bảo features là float32
                    features = features.float()
            elif model_name == 'CLIP-ViT-B/32':
                features = self.models['CLIP-ViT-B/32'].encode_image(batch)
            elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K':
                features = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model'].encode_image(batch)
            elif model_name == 'Flava':
                # Flava cần xử lý từng ảnh riêng biệt để tránh lỗi kích thước tensor
                all_features = []
                for i in range(batch.size(0)):
                    try:
                        # Xử lý từng ảnh một
                        img = batch[i:i+1]  # Giữ chiều batch (1, C, H, W)
                        # Chuyển tensor về PIL Image
                        pil_img = transforms.ToPILImage()(img.squeeze(0).cpu())
                        
                        # Sử dụng processor của Flava
                        inputs = self.models['Flava']['processor'](images=[pil_img], return_tensors="pt").to(self.device)
                        
                        # Lấy embedding từ Flava
                        with torch.no_grad():
                            outputs = self.models['Flava']['model'](
                                pixel_values=inputs["pixel_values"],
                                return_dict=True
                            )
                            feature = outputs.image_embeddings
                            # Normalize feature
                            feature = feature / feature.norm(dim=1, keepdim=True)
                        
                        all_features.append(feature)
                    except Exception as e:
                        print(f"CẢNH BÁO: Không thể xử lý ảnh {i}: {str(e)}")
                        feature_dim = 768  # Flava vision encoder output dimension
                        all_features.append(torch.zeros(1, feature_dim, device=self.device))
                
                # Ghép các features lại với nhau
                features = torch.cat(all_features, dim=0)
            elif model_name == 'ViT-B/16-224':
                # Kiểm tra nếu có thuộc tính vit_projection
                if not hasattr(self, 'vit_projection'):
                    self.setup_projection_layers()
                
                outputs = self.models['ViT-B/16-224'](batch)
                features = outputs.last_hidden_state[:, 0]  # CLS token
                features = self.vit_projection(features)
                # Áp dụng chuẩn hóa thống kê nếu hàm tồn tại
                if hasattr(self, 'apply_statistical_normalization'):
                    features = self.apply_statistical_normalization(features)
            else:
                raise ValueError(f"Model không được hỗ trợ: {model_name}")
            
            # Normalize features
            # Thêm xử lý trường hợp features có norm = 0
            norms = features.norm(dim=1, keepdim=True)
            # Thay thế các norm bằng 0 với 1 để tránh chia cho 0
            norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
            features = features / norms
            
            return features
            
        except Exception as e:
            print(f"ERROR: Lỗi khi xử lý batch ảnh với {model_name}: {str(e)}")
            # Thêm traceback đầy đủ để debug
            import traceback
            traceback.print_exc()
            
            # Return zeros as fallback
            feature_dim = 512  # Default dimension
            if model_name == 'Flava':
                feature_dim = 768  # Flava vision encoder output dimension
            elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K' and 'CLIP-ViT-H-14-laion2B-s32B-b79K' in self.models:
                feature_dim = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model'].config.vision_config.hidden_size
            
            return torch.zeros(batch.size(0), feature_dim, device=self.device)

    def process_text_batch(self, captions, model_name):
        """Xử lý một batch caption theo từng loại model"""
        # Error handling
        try:
            if model_name == 'CLIP-Finetune':
                text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                with torch.no_grad():
                    # CLIPWithClassifier - lấy text features từ CLIP model bên trong
                    features = self.models['CLIP-Finetune'].clip_model.encode_text(text_tokens)
                    # Đảm bảo features là float32
                    features = features.float()
            elif model_name == 'CLIP-ViT-B/32':
                text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                with torch.no_grad():
                    features = self.models['CLIP-ViT-B/32'].encode_text(text_tokens)
            elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K':
                tokenizer = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['tokenizer']
                model = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model']
                
                text_tokens = tokenizer(captions).to(self.device)
                with torch.no_grad():
                    features = model.encode_text(text_tokens)
            elif model_name == 'Flava':
                processor = self.models['Flava']['processor']
                model = self.models['Flava']['model']
                
                # Xử lý từng caption riêng biệt để tránh lỗi kích thước tensor
                all_features = []
                for caption in tqdm(captions, desc="Xử lý captions với Flava"):
                    try:
                        inputs = processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
                        with torch.no_grad():
                            outputs = model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                return_dict=True
                            )
                            feature = outputs.text_embeddings
                            # Normalize feature
                            feature = feature / feature.norm(dim=1, keepdim=True)
                        all_features.append(feature.cpu())
                    except Exception as e2:
                        print(f"CẢNH BÁO: Không thể xử lý caption '{caption[:30]}...': {str(e2)}")
                        feature_dim = 768  # Flava text encoder output dimension
                        all_features.append(torch.zeros(1, feature_dim))
                
                # Ghép các features lại với nhau
                features = torch.cat(all_features, dim=0).numpy()
                return features
                
            elif model_name == 'ViT-B/16-224':
                # ViT không có text encoder nên dùng CLIP
                # Đảm bảo CLIP gốc đã được tải nếu dùng ViT
                if 'CLIP-ViT-B/32' not in self.models:
                    print("Đang tải CLIP gốc cho xử lý text của ViT...")
                    model, _ = clip.load("ViT-B/32", device=self.device)
                    self.models['CLIP-ViT-B/32'] = model
                    
                text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                with torch.no_grad():
                    features = self.models['CLIP-ViT-B/32'].encode_text(text_tokens)
            else:
                raise ValueError(f"Model không được hỗ trợ: {model_name}")
            
            # Normalize features
            # Thêm xử lý trường hợp features có norm = 0
            norms = features.norm(dim=1, keepdim=True)
            # Thay thế các norm bằng 0 với 1 để tránh chia cho 0
            norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
            features = features / norms
            
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"ERROR: Lỗi khi xử lý batch caption với {model_name}: {str(e)}")
            # Thêm traceback đầy đủ để debug
            import traceback
            traceback.print_exc()
            
            # Return zeros as fallback
            feature_dim = 512  # Default dimension
            if model_name == 'Flava':
                feature_dim = 768  # Flava text encoder output dimension
            elif model_name == 'CLIP-ViT-H-14-laion2B-s32B-b79K' and 'CLIP-ViT-H-14-laion2B-s32B-b79K' in self.models:
                feature_dim = self.models['CLIP-ViT-H-14-laion2B-s32B-b79K']['model'].config.text_config.hidden_size
            
            return np.zeros((len(captions), feature_dim))

    def run_evaluation(self):
        """Chạy đánh giá trên tất cả các model và lưu kết quả"""
        results = {}
        
        if hasattr(self.args, 'memory_efficient') and self.args.memory_efficient:
            # Memory-efficient mode: tải và đánh giá từng model, sau đó giải phóng
            for model_name in self.args.models:
                if model_name == 'all':
                    continue  # 'all' chỉ là tùy chọn, không phải model thực sự
                    
                print(f"\n--- MEMORY-EFFICIENT MODE: Đánh giá {model_name} ---")
                # Giải phóng bộ nhớ từ model trước nếu có
                if hasattr(self, 'models') and self.models:
                    del self.models
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.models = {}
                
                # Tải model
                if self.load_single_model(model_name):
                    # Đánh giá model
                    results[model_name] = self.evaluate_model(model_name)
                    
                # Giải phóng bộ nhớ sau khi đánh giá
                if hasattr(self, 'models') and model_name in self.models:
                    if model_name in ['CLIP-ViT-H-14-laion2B-s32B-b79K', 'Flava']:
                        for key in self.models[model_name]:
                            if hasattr(self.models[model_name][key], 'to'):
                                self.models[model_name][key] = self.models[model_name][key].to('cpu')
                    elif hasattr(self.models[model_name], 'to'):
                        self.models[model_name] = self.models[model_name].to('cpu')
                    
                    del self.models[model_name]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # Chế độ tiêu chuẩn: đánh giá tất cả các model đã được tải trước
            if self.args.use_finetuned and 'CLIP-Finetune' in self.models:
                results['CLIP-Finetune'] = self.evaluate_model('CLIP-Finetune')
            
            if self.args.use_clip and 'CLIP-ViT-B/32' in self.models:
                results['CLIP-ViT-B/32'] = self.evaluate_model('CLIP-ViT-B/32')
                
            if self.args.use_openclip and 'CLIP-ViT-H-14-laion2B-s32B-b79K' in self.models:
                results['CLIP-ViT-H-14-laion2B-s32B-b79K'] = self.evaluate_model('CLIP-ViT-H-14-laion2B-s32B-b79K')
            
            if self.args.use_flava and 'Flava' in self.models:
                results['Flava'] = self.evaluate_model('Flava')
            
            if self.args.use_vit and 'ViT-B/16-224' in self.models:
                results['ViT-B/16-224'] = self.evaluate_model('ViT-B/16-224')
        
        # In kết quả tổng hợp sau khi đánh giá tất cả các model
        if results:
            print("\n--- KẾT QUẢ TỔNG HỢP ---")
            self.print_results_table(results)
            
            # Lưu kết quả ra file
            result_file = os.path.join(self.args.output_dir, 'comparison_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                # Chuyển đổi numpy array thành list cho JSON serialization
                def convert_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(i) for i in obj]
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                        return float(obj)
                    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                        return int(obj)
                    else:
                        return obj
                
                # Chuyển đổi kết quả trước khi lưu
                json_results = convert_for_json(results)
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"Đã lưu kết quả so sánh vào {result_file}")
            
            # Tạo file Excel với kết quả chi tiết nếu pandas có sẵn
            try:
                excel_file = os.path.join(self.args.output_dir, 'comparison_results.xlsx')
                
                # Tạo DataFrame cho text-to-image retrieval
                t2i_data = {model: results[model]['t2i'] for model in results if 't2i' in results[model]}
                t2i_df = pd.DataFrame(t2i_data).T
                
                # Tạo DataFrame cho image-to-text retrieval
                i2t_data = {model: results[model]['i2t'] for model in results if 'i2t' in results[model]}
                i2t_df = pd.DataFrame(i2t_data).T
                
                # Tạo DataFrame cho mean metrics
                mean_data = {model: results[model]['mean'] for model in results if 'mean' in results[model]}
                mean_df = pd.DataFrame(mean_data).T
                
                # Lưu vào Excel với nhiều sheets
                with pd.ExcelWriter(excel_file) as writer:
                    t2i_df.to_excel(writer, sheet_name='Text-to-Image')
                    i2t_df.to_excel(writer, sheet_name='Image-to-Text')
                    mean_df.to_excel(writer, sheet_name='Mean Metrics')
                    
                print(f"Đã lưu kết quả chi tiết vào Excel: {excel_file}")
            except Exception as e:
                print(f"Không thể tạo file Excel: {str(e)}")
            
            # Vẽ biểu đồ so sánh
            self.plot_comparison(results)
        else:
            print("Không có kết quả đánh giá nào!")
        
        return results

    def print_results_table(self, results):
        """In bảng kết quả so sánh"""
        print("\n--- KẾT QUẢ SO SÁNH CÁC MODEL ---")
        
        # Lập bảng cho text-to-image retrieval
        print("\nText-to-Image Retrieval:")
        metrics = ['R@1', 'R@5', 'R@10', 'MRR', 'Median_Rank', 'Mean_Rank']
        header = ["Model"] + metrics
        
        # Chuẩn bị rows
        rows = []
        for model_name, model_results in results.items():
            if 't2i' not in model_results:
                continue
                
            row = [model_name]
            t2i_results = model_results['t2i']
            for metric in metrics:
                value = t2i_results[metric]
                # Format số thập phân
                if isinstance(value, float):
                    if metric.startswith(('R@', 'MRR')):
                        formatted = f"{value:.4f}"
                    else:
                        formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                row.append(formatted)
            rows.append(row)
        
        # Tính column widths
        col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
        
        # In header
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
        print(header_str)
        print("-" * len(header_str))
        
        # In rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_str)
            
        # Lập bảng cho image-to-text retrieval
        print("\nImage-to-Text Retrieval:")
        header = ["Model"] + metrics
        
        # Chuẩn bị rows
        rows = []
        for model_name, model_results in results.items():
            if 'i2t' not in model_results:
                continue
                
            row = [model_name]
            i2t_results = model_results['i2t']
            for metric in metrics:
                value = i2t_results[metric]
                # Format số thập phân
                if isinstance(value, float):
                    if metric.startswith(('R@', 'MRR')):
                        formatted = f"{value:.4f}"
                    else:
                        formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                row.append(formatted)
            rows.append(row)
        
        # Tính column widths
        col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
        
        # In header
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
        print(header_str)
        print("-" * len(header_str))
        
        # In rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_str)
            
        # Lập bảng cho mean metrics
        print("\nMean Metrics & rsum:")
        metrics = ['R@1', 'R@5', 'R@10', 'MRR', 'rsum']
        header = ["Model"] + metrics
        
        # Chuẩn bị rows
        rows = []
        for model_name, model_results in results.items():
            if 'mean' not in model_results:
                continue
                
            row = [model_name]
            mean_results = model_results['mean']
            for metric in metrics:
                value = mean_results[metric]
                # Format số thập phân
                if isinstance(value, float):
                    if metric == 'rsum':
                        formatted = f"{value:.2f}"
                    else:
                        formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                row.append(formatted)
            rows.append(row)
        
        # Tính column widths
        col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
        
        # In header
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
        print(header_str)
        print("-" * len(header_str))
        
        # In rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_str)

    def plot_comparison(self, results):
        """Vẽ biểu đồ so sánh các metrics giữa các model"""
        # Lọc models có đủ kết quả
        valid_models = []
        for model_name, model_results in results.items():
            if 't2i' in model_results and 'i2t' in model_results and 'mean' in model_results:
                valid_models.append(model_name)
        
        if not valid_models:
            print("Không có model nào có đủ kết quả để vẽ biểu đồ")
            return
            
        # Plot 1: Text-to-Image Recall@K
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        
        metrics_to_plot = ['R@1', 'R@5', 'R@10']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(valid_models)  # Độ rộng của mỗi bar
        
        for i, model_name in enumerate(valid_models):
            values = [results[model_name]['t2i'][metric] for metric in metrics_to_plot]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Text-to-Image Retrieval (R@K)')
        plt.xticks(x + width * (len(valid_models) - 1) / 2, metrics_to_plot)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Image-to-Text Recall@K
        plt.subplot(2, 2, 2)
        
        for i, model_name in enumerate(valid_models):
            values = [results[model_name]['i2t'][metric] for metric in metrics_to_plot]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Image-to-Text Retrieval (R@K)')
        plt.xticks(x + width * (len(valid_models) - 1) / 2, metrics_to_plot)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Mean Recall@K
        plt.subplot(2, 2, 3)
        
        for i, model_name in enumerate(valid_models):
            values = [results[model_name]['mean'][metric] for metric in metrics_to_plot]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Mean Recall@K (Average of T2I and I2T)')
        plt.xticks(x + width * (len(valid_models) - 1) / 2, metrics_to_plot)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 4: rsum
        plt.subplot(2, 2, 4)
        
        rsum_values = [results[model_name]['mean']['rsum'] for model_name in valid_models]
        plt.bar(valid_models, rsum_values, width=0.5)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('rsum (sum of all R@K across T2I and I2T)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Lưu biểu đồ
        plt.tight_layout()
        chart_path = os.path.join(self.args.output_dir, 'comparison_chart.png')
        plt.savefig(chart_path)
        print(f"Đã lưu biểu đồ so sánh vào {chart_path}")
        
        # Vẽ biểu đồ thứ hai: Median Rank và Processing Time
        plt.figure(figsize=(15, 6))
        
        # Plot 1: Median Ranks (t2i vs i2t)
        plt.subplot(1, 2, 1)
        
        t2i_median = [results[model_name]['t2i']['Median_Rank'] for model_name in valid_models]
        i2t_median = [results[model_name]['i2t']['Median_Rank'] for model_name in valid_models]
        
        x = np.arange(len(valid_models))
        width = 0.35
        
        plt.bar(x - width/2, t2i_median, width, label='T2I')
        plt.bar(x + width/2, i2t_median, width, label='I2T')
        
        plt.xlabel('Models')
        plt.ylabel('Median Rank')
        plt.title('Median Ranks (lower is better)')
        plt.xticks(x, valid_models, rotation=15)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Processing Time
        plt.subplot(1, 2, 2)
        
        proc_times = [results[model_name].get('processing_time', 0) for model_name in valid_models]
        plt.bar(valid_models, proc_times, width=0.5)
        
        plt.xlabel('Models')
        plt.ylabel('Seconds')
        plt.title('Total Processing Time')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Lưu biểu đồ
        plt.tight_layout()
        chart_path2 = os.path.join(self.args.output_dir, 'rank_time_chart.png')
        plt.savefig(chart_path2)
        print(f"Đã lưu biểu đồ rank và thời gian vào {chart_path2}")

def main():
    # Khai báo cứng các tham số thay vì dùng argparse
    # Đường dẫn file và thư mục - Điều chỉnh cho môi trường Windows
    base_dir = "E:\\Đồ án chuyên ngành"
    project_dir = "/kaggle/input/finalv3_v1/pytorch/default/1"
    
    # Đường dẫn tới dữ liệu Flickr30k - Nếu có, hoặc sử dụng dataset tùy chỉnh
    flickr_images_dir ="/kaggle/input/flickr30k/flickr30k_images"
    flickr_captions_path = "/kaggle/input/flickr30k/captions.txt"
    
    # Đường dẫn tới model đã fine-tune
    finetuned_model_path = "/kaggle/input/finalv3_v1/pytorch/default/1/final_checkpoint.pt"
    
    # Thư mục lưu kết quả
    output_dir = "/kaggle/working/compare_results"
    
    # Các tùy chọn model (True/False để bật/tắt việc sử dụng model)
    use_finetuned = True   # Có sử dụng model CLIP fine-tuned không
    use_clip = True        # Có sử dụng model CLIP gốc không
    use_openclip = True   # Có sử dụng model OpenCLIP (LAION) không
    use_flava = True        # Có sử dụng model Flava không
    use_vit = False        # Có sử dụng model ViT không
    
    # Tham số khác
    batch_size = 32        # Giảm batch size cho máy cấu hình thấp
    test_size = 1000        # Số lượng ảnh test từ Flickr30k hoặc dataset tùy chỉnh
    memory_efficient = True  # Sử dụng chế độ tiết kiệm bộ nhớ
    
    # Kiểm tra và tìm checkpoint
    if not os.path.exists(finetuned_model_path):
        # Thử tìm trong thư mục hiện tại
        current_dir_checkpoints = glob.glob("*.pt")
        if current_dir_checkpoints:
            finetuned_model_path = current_dir_checkpoints[0]
            print(f"Sử dụng checkpoint trong thư mục hiện tại: {finetuned_model_path}")
        else:
            # Thử tìm checkpoint trong các thư mục khác
            checkpoint_dir = os.path.join(project_dir, "clip_finetune_checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
                if checkpoints:
                    finetuned_model_path = checkpoints[0]  # Lấy checkpoint đầu tiên tìm được
                    print(f"Đã tìm thấy checkpoint tại: {finetuned_model_path}")
    
    # Tạo class chứa tham số để giữ cấu trúc code tương thích 
    class Args:
        def __init__(self):
            self.flickr_images_dir = flickr_images_dir
            self.flickr_captions_path = flickr_captions_path
            self.finetuned_model_path = finetuned_model_path
            self.output_dir = output_dir
            self.use_finetuned = use_finetuned
            self.use_clip = use_clip
            self.use_openclip = use_openclip
            self.use_flava = use_flava
            self.use_vit = use_vit
            self.batch_size = batch_size
            self.test_size = test_size
            self.memory_efficient = memory_efficient
            # Thêm danh sách models để tương thích với phiên bản khác
            self.models = []
            if use_finetuned: self.models.append('CLIP-Finetune')
            if use_clip: self.models.append('CLIP-ViT-B/32')
            if use_openclip: self.models.append('CLIP-ViT-H-14-laion2B-s32B-b79K')
            if use_flava: self.models.append('Flava')
            if use_vit: self.models.append('ViT-B/16-224')
            
    args = Args()
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Kiểm tra các đường dẫn cần thiết
    if args.use_finetuned and not os.path.exists(args.finetuned_model_path):
        print(f"WARNING: Không tìm thấy model CLIP fine-tuned tại {args.finetuned_model_path}")
        print("Tiếp tục nhưng không sử dụng model fine-tuned.")
        args.use_finetuned = False
        if 'CLIP-Finetune' in args.models:
            args.models.remove('CLIP-Finetune')
    
    # Nếu không tìm thấy Flickr30k, thử sử dụng dataset tùy chỉnh
    if not os.path.exists(args.flickr_images_dir) or not os.path.exists(args.flickr_captions_path):
        print(f"WARNING: Không tìm thấy dataset Flickr30k")
        print("Thử tìm dữ liệu thay thế...")
        
        # Thử tìm thư mục dataset tùy chỉnh
        test_dataset_dir = os.path.join(base_dir, "dataset", "testseg", "datasosanh")
        if os.path.exists(test_dataset_dir):
            args.flickr_images_dir = test_dataset_dir
            print(f"Sử dụng thư mục ảnh thay thế: {args.flickr_images_dir}")
            
            # Tạo file caption tạm thời nếu cần
            temp_caption_file = os.path.join(project_dir, "temp_captions.txt")
            with open(temp_caption_file, 'w', encoding='utf-8') as f:
                f.write("image_name,comment_number,comment\n")
                # Quét thư mục ảnh và tạo captions tạm thời
                for img_path in glob.glob(os.path.join(test_dataset_dir, "*.jpg")):
                    img_name = os.path.basename(img_path)
                    f.write(f"{img_name},1,a photo of {img_name}\n")
            
            args.flickr_captions_path = temp_caption_file
            print(f"Đã tạo file caption tạm thời: {args.flickr_captions_path}")
        else:
            print("ERROR: Không tìm thấy thư mục ảnh thay thế.")
            return
    
    print("\n" + "="*50)
    print("CẤU HÌNH SO SÁNH MÔ HÌNH")
    print("="*50)
    print(f"Thư mục ảnh: {args.flickr_images_dir}")
    print(f"File caption: {args.flickr_captions_path}")
    print(f"Model fine-tuned: {args.finetuned_model_path}")
    print(f"Các model sẽ đánh giá: {', '.join(args.models)}")
    print(f"Thư mục lưu kết quả: {args.output_dir}")
    print("="*50 + "\n")
    
    # Chạy so sánh
    comparison = ModelComparison(args)
    results = comparison.run_evaluation()
    
    print(f"\nHoàn thành so sánh các model! Kết quả được lưu vào thư mục: {args.output_dir}")

if __name__ == '__main__':
    main() 