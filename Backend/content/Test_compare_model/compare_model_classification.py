import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import clip
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import json
import open_clip
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
import sys

# Import CLIPWithClassifier from clip_finetune_correct.py
sys.path.append('.')
try:
    from clip_finetune_correct import CLIPWithClassifier
except ImportError:
    print("WARNING: Không thể import CLIPWithClassifier từ clip_finetune_correct.py")
    print("Đảm bảo file clip_finetune_correct.py nằm trong cùng thư mục hoặc trong PYTHONPATH")

class ClassificationDataset(Dataset):
    """Dataset for image classification task"""
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.img_dir = os.path.dirname(csv_path)
        self.class_to_idx = {
            "Sensitive content": 0,
            "Violence": 1,
            "NonViolence": 2  # 3 lớp chính: Sensitive content, Violence, NonViolence
        }
        print(f"Loaded {len(self.data)} images from {csv_path}")
        print(f"Classes: {list(self.class_to_idx.keys())}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # First column is image path/name
        category = self.data.iloc[idx, 1]  # Second column is category
        
        # Handle both full paths and filenames
        if os.path.isabs(img_name):
            img_path = img_name
        else:
            img_path = os.path.join(self.img_dir, img_name)
            
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a black image as fallback
            image = torch.zeros((3, 224, 224))
            
        label = self.class_to_idx[category]
        
        return {
            'image': image,
            'label': label,
            'path': img_path,
            'category': category
        }

def load_clip_with_classifier(checkpoint_path, num_classes=3, device=None):
    """Load a fine-tuned CLIP model with classifier"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Initialize CLIPWithClassifier với num_classes=3 để khớp với model đã train
    model = CLIPWithClassifier(clip_model, num_classes=num_classes)
    model = model.to(device)
    model = model.float()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check checkpoint format
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model_state_dict from checkpoint {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded state_dict directly from {checkpoint_path}")
        
        model.eval()  # Set model to evaluation mode
        return model, preprocess
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {str(e)}")
        raise

class LinearClassifier(nn.Module):
    """Linear classifier for image features"""
    def __init__(self, input_dim, num_classes=3):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, features):
        return self.classifier(features)

class ModelComparisonClassification:
    """Main class for comparing different models on classification task"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set up output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {args.csv_path}")
        self.csv_path = args.csv_path
        
        # Initialize models dictionary
        self.models = {}
        self.classifiers = {}
        self.transforms = {}
        
        # Only load all models immediately if not using memory-efficient mode
        if not hasattr(args, 'memory_efficient') or not args.memory_efficient:
            self.load_models()
            self.setup_transforms()
        
    def load_single_model(self, model_name):
        """Load a specific model"""
        print(f"Loading model {model_name}...")
        
        if model_name == 'clip_finetuned' and self.args.use_finetuned:
            try:
                # Try loading with CLIPWithClassifier
                self.models['clip_finetuned'], _ = load_clip_with_classifier(
                    self.args.finetuned_model_path,
                    num_classes=3,  # 3 lớp: Sensitive content, Violence, NonViolence
                    device=self.device
                )
                print("Successfully loaded CLIPWithClassifier.")
                print("CLIP fine-tuned model đã sẵn có classifier, sẽ dùng trực tiếp và không train lại.")
                
                # No need for a separate classifier for fine-tuned model as it has one
                self.classifiers['clip_finetuned'] = None
                
            except Exception as e:
                print(f"Cannot load CLIPWithClassifier: {str(e)}")
                print("Trying fallback method...")
                
                # Fallback: load CLIP model and add a classifier
                model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
                self.models['clip_finetuned'] = model
                
                # Create a linear classifier for CLIP features - với 3 classes
                self.classifiers['clip_finetuned'] = LinearClassifier(512, 3).to(self.device)
                print("Loaded base CLIP model with separate classifier.")
            
        elif model_name == 'clip_original' and self.args.use_clip:
            print("Loading original CLIP...")
            model, _ = clip.load("ViT-B/32", device=self.device)
            self.models['clip_original'] = model
            
            # Create a linear classifier for CLIP features with 3 classes
            self.classifiers['clip_original'] = LinearClassifier(512, 3).to(self.device)
            
        elif model_name == 'openclip' and self.args.use_openclip:
            print("Loading OpenCLIP (LAION)...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )
            self.models['openclip'] = {
                'model': model,
                'preprocess': preprocess
            }
            
            # Create a linear classifier for OpenCLIP features with 3 classes
            self.classifiers['openclip'] = LinearClassifier(512, 3).to(self.device)
            
        elif model_name == 'florence' and self.args.use_florence:
            print("Loading Microsoft Florence...")
            processor = CLIPProcessor.from_pretrained("microsoft/florence-vit-base")
            model = CLIPModel.from_pretrained("microsoft/florence-vit-base").to(self.device)
            self.models['florence'] = {
                'model': model,
                'processor': processor
            }
            
            # Create a linear classifier for Florence features with 3 classes
            feature_dim = model.config.projection_dim
            self.classifiers['florence'] = LinearClassifier(feature_dim, 3).to(self.device)
            
        elif model_name == 'vit' and self.args.use_vit:
            print("Loading ViT...")
            model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.models['vit'] = model
            
            # Create a linear classifier for ViT features with 3 classes
            self.classifiers['vit'] = LinearClassifier(768, 3).to(self.device)
            
        # Set up transforms for the loaded model
        self.setup_transforms_for_model(model_name)
            
        return model_name in self.models

    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # 1. CLIP fine-tuned
        if self.args.use_finetuned:
            self.load_single_model('clip_finetuned')
            
        # 2. Original CLIP
        if self.args.use_clip:
            self.load_single_model('clip_original')
            
        # 3. OpenCLIP
        if self.args.use_openclip:
            self.load_single_model('openclip')
            
        # 4. Florence
        if self.args.use_florence:
            self.load_single_model('florence')
            
        # 5. ViT
        if self.args.use_vit:
            self.load_single_model('vit')
            
    def setup_transforms(self):
        """Set up transforms for each model"""
        # CLIP and CLIP fine-tuned
        self.transforms['clip'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # OpenCLIP uses its own preprocess
        
        # ViT
        self.transforms['vit'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def setup_transforms_for_model(self, model_name):
        """Set up transform for a specific model"""
        if not hasattr(self, 'transforms'):
            self.transforms = {}
            
        if model_name in ['clip_finetuned', 'clip_original']:
            self.transforms['clip'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
        elif model_name == 'vit':
            self.transforms['vit'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def train_classifier(self, model_name, train_loader):
        """Train a linear classifier on top of model features"""
        if model_name not in self.models or model_name not in self.classifiers:
            print(f"Model {model_name} not loaded")
            return False
            
        # Skip if model is a fine-tuned classifier already
        if model_name == 'clip_finetuned' and self.classifiers[model_name] is None:
            print(f"Skipping training for {model_name} as it's already a classifier")
            return True
            
        print(f"Training classifier for {model_name}...")
        
        classifier = self.classifiers[model_name]
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        num_epochs = self.args.classifier_epochs
        
        for epoch in range(num_epochs):
            classifier.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.extract_features(images, model_name)
                    
                # Forward pass through classifier
                optimizer.zero_grad()
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Print epoch stats
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")
            
        print(f"Finished training classifier for {model_name}")
        return True

    def extract_features(self, images, model_name):
        """Extract features from model for a batch of images"""
        if model_name == 'clip_finetuned' or model_name == 'clip_original':
            model = self.models[model_name]
            features = model.encode_image(images)
            features = features / features.norm(dim=1, keepdim=True)  # Normalize
            
        elif model_name == 'openclip':
            model = self.models[model_name]['model']
            features = model.encode_image(images)
            features = features / features.norm(dim=1, keepdim=True)  # Normalize
            
        elif model_name == 'florence':
            # For florence, we need to convert tensors to PIL and use processor
            try:
                # Process in smaller batches if needed
                if images.size(0) > 8:
                    all_features = []
                    for i in range(0, images.size(0), 4):
                        mini_batch = images[i:i+4]
                        mini_images = [transforms.ToPILImage()(img.cpu()) for img in mini_batch]
                        inputs = self.models['florence']['processor'](
                            images=mini_images, 
                            return_tensors="pt"
                        ).to(self.device)
                        mini_features = self.models['florence']['model'].get_image_features(**inputs)
                        all_features.append(mini_features)
                    features = torch.cat(all_features, dim=0)
                else:
                    images_pil = [transforms.ToPILImage()(img.cpu()) for img in images]
                    inputs = self.models['florence']['processor'](
                        images=images_pil, 
                        return_tensors="pt"
                    ).to(self.device)
                    features = self.models['florence']['model'].get_image_features(**inputs)
                    
                features = features / features.norm(dim=1, keepdim=True)  # Normalize
            except Exception as e:
                print(f"Error processing Florence features: {str(e)}")
                # Fallback to zeros
                feature_dim = self.models['florence']['model'].config.projection_dim
                features = torch.zeros(images.size(0), feature_dim).to(self.device)
            
        elif model_name == 'vit':
            model = self.models['vit']
            outputs = model(images)
            features = outputs.last_hidden_state[:, 0]  # CLS token
            
        return features

    def evaluate_model(self, model_name, test_loader):
        """Evaluate model on test set"""
        print(f"\n--- Evaluating {model_name} using prompt-based classification ---")
        
        start_time = time.time()
        
        # Track predictions
        all_labels = []
        all_preds = []
        all_paths = []
        all_probs = []  # For confidence scores
        
        # Tạo các prompts cố định cho phân loại - dùng chung cho mọi model
        # Sử dụng đúng 3 lớp: "Sensitive content", "Violence", "NonViolence"
        prompts = ["a sensitive content image", "a violence image", "a nonviolence image"]
        
        # Tokenize prompts theo từng loại model
        if model_name == 'clip_finetuned' or model_name == 'clip_original':
            text_tokens = clip.tokenize(prompts).to(self.device)
        elif model_name == 'openclip':
            # OpenCLIP có thể có tokenizer khác nhau
            if 'tokenizer' in self.models['openclip']:
                text_tokens = self.models['openclip']['tokenizer'](prompts).to(self.device)
            else:
                # Sử dụng tokenizer từ open_clip
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                text_tokens = tokenizer(prompts).to(self.device)
        elif model_name == 'florence':
            text_inputs = self.models['florence']['processor'](text=prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                paths = batch['path']
                
                # Xử lý theo từng loại model - đều sử dụng prompt-based
                if model_name == 'clip_finetuned':
                    try:
                        # Lấy image features
                        image_features = self.models['clip_finetuned'].clip_model.encode_image(images)
                        image_features = image_features.float()
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        # Lấy text features từ các prompt cố định
                        text_features = self.models['clip_finetuned'].clip_model.encode_text(text_tokens)
                        text_features = text_features.float()
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Tính similarity
                        logit_scale = self.models['clip_finetuned'].logit_scale.exp().float()
                        logits = logit_scale * image_features @ text_features.t()
                        
                        # Lấy class có similarity cao nhất
                        probabilities = torch.softmax(logits, dim=1)
                    except Exception as e:
                        print(f"Error using prompt-based for fine-tuned CLIP: {e}")
                        # Fallback to class head
                        dummy_texts = clip.tokenize(["placeholder text"] * images.size(0)).to(self.device)
                        _, _, class_logits = self.models['clip_finetuned'](images, dummy_texts)
                        probabilities = torch.softmax(class_logits, dim=1)
                
                elif model_name == 'clip_original':
                    # Lấy image và text features
                    image_features = self.models['clip_original'].encode_image(images)
                    text_features = self.models['clip_original'].encode_text(text_tokens)
                    
                    # Chuẩn hóa features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Tính similarity với logit scale mặc định
                    logit_scale = self.models['clip_original'].logit_scale.exp()
                    logits = logit_scale * image_features @ text_features.t()
                    
                    # Softmax để lấy probabilities
                    probabilities = torch.softmax(logits, dim=1)
                
                elif model_name == 'openclip':
                    # Lấy image và text features từ OpenCLIP
                    image_features = self.models['openclip']['model'].encode_image(images)
                    text_features = self.models['openclip']['model'].encode_text(text_tokens)
                    
                    # Chuẩn hóa features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Tính similarity
                    logits = 100.0 * image_features @ text_features.t()  # OpenCLIP thường dùng 100.0
                    
                    # Softmax để lấy probabilities
                    probabilities = torch.softmax(logits, dim=1)
                
                elif model_name == 'florence':
                    try:
                        # Lấy image features từ Florence
                        image_features = self.models['florence']['model'].get_image_features(**{
                            'pixel_values': images
                        })
                        
                        # Lấy text features từ Florence
                        text_features = self.models['florence']['model'].get_text_features(**text_inputs)
                        
                        # Chuẩn hóa features
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Tính similarity
                        logits = 100.0 * image_features @ text_features.t()
                        
                        # Softmax để lấy probabilities
                        probabilities = torch.softmax(logits, dim=1)
                    except Exception as e:
                        print(f"Error using prompt-based for Florence: {e}")
                        # Fallback to zeros
                        probabilities = torch.zeros((images.size(0), 3)).to(self.device)
                        probabilities[:, 0] = 1.0  # Default to class 0
                
                elif model_name == 'vit':
                    # ViT không có text encoder, nên chúng ta chỉ sử dụng image features
                    # và so sánh với CLIP text features
                    
                    # Lấy ViT image features (CLS token)
                    vit_outputs = self.models['vit'](images)
                    vit_features = vit_outputs.last_hidden_state[:, 0]  # CLS token
                    
                    # Sử dụng projection layer nếu có
                    if hasattr(self, 'vit_projection') and self.vit_projection is not None:
                        vit_features = self.vit_projection(vit_features)
                    
                    # Chuẩn hóa features
                    vit_features = vit_features / vit_features.norm(dim=-1, keepdim=True)
                    
                    # Lấy text features từ CLIP (vì ViT không có text encoder)
                    if 'clip_original' in self.models:
                        clip_text_tokens = clip.tokenize(prompts).to(self.device)
                        text_features = self.models['clip_original'].encode_text(clip_text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Tính similarity
                        logits = 100.0 * vit_features @ text_features.t()
                        
                        # Softmax để lấy probabilities
                        probabilities = torch.softmax(logits, dim=1)
                    else:
                        # Fallback nếu không có CLIP
                        print("WARNING: ViT prompt-based needs CLIP for text encoding, but CLIP not found!")
                        probabilities = torch.zeros((images.size(0), 3)).to(self.device)
                        probabilities[:, 0] = 1.0  # Default to class 0
                
                else:
                    # Fallback for unknown models
                    print(f"ERROR: Model {model_name} không được hỗ trợ để đánh giá prompt-based")
                    probabilities = torch.zeros((images.size(0), 3)).to(self.device)
                    probabilities[:, 0] = 1.0  # Default to class 0
                
                # Lấy class có probability cao nhất
                _, predicted = probabilities.max(1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_paths.extend(paths)
                all_probs.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report - sử dụng đúng tên lớp
        target_names = ["Sensitive content", "Violence", "NonViolence"]
        cls_report = classification_report(all_labels, all_preds, target_names=target_names)
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Save incorrect predictions
        incorrect_samples = []
        for i, (label, pred, path, prob) in enumerate(zip(all_labels, all_preds, all_paths, all_probs)):
            if label != pred:
                incorrect_samples.append({
                    'path': path,
                    'true_label': target_names[label],
                    'pred_label': target_names[pred],
                    'confidence': float(prob[pred])
                })
        
        # Create results dictionary
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': cls_report,
            'processing_time': float(processing_time),
            'incorrect_samples': incorrect_samples
        }
        
        # Print results
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cls_report)
        
        return results

    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Sử dụng đúng 3 lớp: "Sensitive content", "Violence", "NonViolence"
        classes = ["Sensitive content", "Violence", "NonViolence"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure
        cm_path = os.path.join(self.args.output_dir, f'cm_{model_name}.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
    def plot_comparison(self, results):
        """Plot comparison metrics between models"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [results[model][metric] for model in models] for metric in metrics}
        
        # Plot metrics comparison
        plt.figure(figsize=(12, 8))
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + (i - 1.5) * width, metric_values[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Classification Metrics Comparison')
        plt.xticks(x, models, rotation=15)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        comparison_path = os.path.join(self.args.output_dir, 'metrics_comparison.png')
        plt.savefig(comparison_path)
        print(f"Metrics comparison chart saved to {comparison_path}")
        
        # Plot processing time
        plt.figure(figsize=(10, 6))
        times = [results[model]['processing_time'] for model in models]
        plt.bar(models, times)
        plt.xlabel('Models')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time Comparison')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        time_path = os.path.join(self.args.output_dir, 'time_comparison.png')
        plt.savefig(time_path)
        print(f"Processing time chart saved to {time_path}")

    def run_evaluation(self):
        """Main function to run evaluation on all models"""
        # Load dataset
        dataset = ClassificationDataset(self.csv_path, transform=None)
        
        # Calculate split indices
        train_size = int(0.2 * len(dataset))  # Chỉ dùng 20% cho train nếu cần
        test_size = len(dataset) - train_size
        
        # Create train/test splits
        from torch.utils.data import random_split
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        
        # Store results
        results = {}
        
        # Evaluate each model
        for model_name in self.args.models:
            if model_name == 'all':
                continue  # 'all' is just an option, not an actual model
                
            print(f"\n--- EVALUATING MODEL: {model_name} with prompt-based classification ---")
            
            # Memory-efficient mode: load and evaluate one model at a time
            if hasattr(self.args, 'memory_efficient') and self.args.memory_efficient:
                print(f"Memory-efficient mode: loading {model_name}")
                if hasattr(self, 'models') and self.models:
                    # Clear previous model
                    for key in list(self.models.keys()):
                        if key in self.classifiers and self.classifiers[key] is not None:
                            self.classifiers[key] = self.classifiers[key].to('cpu')
                        if key in self.models:
                            if isinstance(self.models[key], dict):
                                for subkey in self.models[key]:
                                    if hasattr(self.models[key][subkey], 'to'):
                                        self.models[key][subkey] = self.models[key][subkey].to('cpu')
                            elif hasattr(self.models[key], 'to'):
                                self.models[key] = self.models[key].to('cpu')
                    
                    self.models = {}
                    self.classifiers = {}
                    torch.cuda.empty_cache()
                
                # Load the current model
                if not self.load_single_model(model_name):
                    print(f"ERROR: Could not load {model_name}, skipping")
                    continue
                    
                # Load CLIP original nếu cần cho ViT prompt-based
                if model_name == 'vit' and 'clip_original' not in self.models and self.args.use_clip:
                    print("Tải CLIP gốc để hỗ trợ ViT prompt-based classification...")
                    self.models['clip_original'], _ = clip.load("ViT-B/32", device=self.device)
            
            # Create dataloaders with appropriate transforms
            if model_name in ['clip_finetuned', 'clip_original']:
                transform = self.transforms['clip']
            elif model_name == 'openclip' and 'openclip' in self.models:
                transform = self.models['openclip']['preprocess']
            elif model_name == 'florence':
                # For Florence we handle preprocessing in extract_features
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
            elif model_name == 'vit':
                transform = self.transforms['vit']
            else:
                print(f"ERROR: No transform available for {model_name}")
                continue
            
            # Create custom datasets with the right transform
            class TransformDataset(Dataset):
                def __init__(self, dataset, transform):
                    self.dataset = dataset
                    self.transform = transform
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    if isinstance(item['image'], torch.Tensor):
                        # Convert tensor to PIL for transform
                        image = transforms.ToPILImage()(item['image'])
                    else:
                        image = item['image']
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    return {
                        'image': image,
                        'label': item['label'],
                        'path': item['path'],
                        'category': item['category'] if 'category' in item else ""
                    }
            
            # Chỉ cần test dataset với prompt-based
            test_dataset_transformed = TransformDataset(test_dataset, transform)
            
            test_loader = DataLoader(
                test_dataset_transformed, 
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=2
            )
            
            # Với prompt-based, không cần train classifier
            print(f"Đánh giá {model_name} với prompt-based, không huấn luyện lại classifier")
            
            # Evaluate model
            model_results = self.evaluate_model(model_name, test_loader)
            results[model_name] = model_results
            
            # Plot confusion matrix
            self.plot_confusion_matrix(np.array(model_results['confusion_matrix']), model_name)
        
        # Plot comparison charts
        if results:
            self.plot_comparison(results)
            
            # Save results to JSON
            result_file = os.path.join(self.args.output_dir, 'prompt_based_classification_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                # Clean up numpy/torch types for JSON serialization
                clean_results = {}
                for model, res in results.items():
                    clean_results[model] = {
                        key: (val.tolist() if hasattr(val, 'tolist') else val) 
                        for key, val in res.items()
                    }
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_file}")
            
            # Create Excel report with results
            try:
                excel_file = os.path.join(self.args.output_dir, 'prompt_based_classification_results.xlsx')
                
                # Create summary DataFrame
                summary_data = {
                    'Model': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1 Score': [],
                    'Processing Time (s)': []
                }
                
                for model, res in results.items():
                    summary_data['Model'].append(model)
                    summary_data['Accuracy'].append(res['accuracy'])
                    summary_data['Precision'].append(res['precision'])
                    summary_data['Recall'].append(res['recall'])
                    summary_data['F1 Score'].append(res['f1_score'])
                    summary_data['Processing Time (s)'].append(res['processing_time'])
                
                summary_df = pd.DataFrame(summary_data)
                
                # Save to Excel
                with pd.ExcelWriter(excel_file) as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Create sheets for incorrect predictions
                    for model, res in results.items():
                        incorrect = pd.DataFrame(res['incorrect_samples'])
                        if not incorrect.empty:
                            incorrect.to_excel(writer, sheet_name=f'{model}_incorrect', index=False)
                
                print(f"Excel report saved to {excel_file}")
            except Exception as e:
                print(f"Error creating Excel report: {str(e)}")
        else:
            print("No evaluation results to report!")
        
        return results

def main():
    """Main function"""
    # Define parameters
    data_path = "/kaggle/input/data-classification/datatest_classification"
    csv_path = os.path.join(data_path, "image_mapping.csv")
    output_dir = "/kaggle/working/prompt_based_classification_results"  # Đổi tên thư mục kết quả
    
    # Đường dẫn đến model fine-tuned của bạn - cập nhật đúng đường dẫn
    finetuned_model_path = "/kaggle/input/finalv3_v1/pytorch/default/1/final_checkpoint.pt"
    
    # Model options
    use_finetuned = True
    use_clip = True
    use_openclip = True
    use_florence = False  # Florence is resource-intensive
    use_vit = False
    
    # Other parameters
    batch_size = 32
    classifier_epochs = 5  # Không còn cần nữa với prompt-based
    memory_efficient = True
    
    # Create Args class
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.csv_path = csv_path
            self.output_dir = output_dir
            self.finetuned_model_path = finetuned_model_path
            self.use_finetuned = use_finetuned
            self.use_clip = use_clip
            self.use_openclip = use_openclip
            self.use_florence = use_florence
            self.use_vit = use_vit
            self.batch_size = batch_size
            self.classifier_epochs = classifier_epochs
            self.memory_efficient = memory_efficient
            # Create list of models to evaluate
            self.models = []
            if use_finetuned: self.models.append('clip_finetuned')
            if use_clip: self.models.append('clip_original')
            if use_openclip: self.models.append('openclip')
            if use_florence: self.models.append('florence')
            if use_vit: self.models.append('vit')
    
    args = Args()
    
    # Check if required files exist
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found at {args.csv_path}")
        return
    
    if args.use_finetuned and not os.path.exists(args.finetuned_model_path):
        print(f"WARNING: Fine-tuned model not found at {args.finetuned_model_path}")
        print("Continuing without fine-tuned model.")
        args.use_finetuned = False
        if 'clip_finetuned' in args.models:
            args.models.remove('clip_finetuned')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the comparison
    comparison = ModelComparisonClassification(args)
    results = comparison.run_evaluation()
    
    print(f"Complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
