"""
EmbeddingService - Quản lý và xử lý embeddings
"""

import os
import numpy as np
import json
import torch
import clip
import faiss
from pathlib import Path
from PIL import Image
import torch.nn as nn

# Define the CLIPWithClassifier class to match the training model structure
class CLIPWithClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=3, freeze_layers=7):
        super(CLIPWithClassifier, self).__init__()
        self.clip_model = clip_model
        
        # Convert model to float32 for consistency
        self.clip_model.float()
        
        # Embedding dimension from CLIP model
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Logit scale parameter
        self.logit_scale = self.clip_model.logit_scale
    
    def forward(self, images, texts=None, get_embeddings=False):
        # For embedding only, we don't need text input
        images = images.type(torch.float32)
        
        # Get image features
        image_features = self.clip_model.encode_image(images)
        image_features = image_features.float()
        
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # For embedding service, we only need image features
        if texts is None:
            return image_features
        
        # Full forward pass with texts if provided
        text_features = self.clip_model.encode_text(texts)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate logits
        logit_scale = self.logit_scale.exp().float()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Classification logits
        class_logits = self.classifier(image_features)
        
        if get_embeddings:
            return image_features, text_features, logits_per_image, logits_per_text, class_logits
        
        return logits_per_image, logits_per_text, class_logits

class EmbeddingService:
    def __init__(self, cache_service, path_service, data_service, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Khởi tạo EmbeddingService
        
        Args:
            cache_service: Đối tượng CacheService
            path_service: Đối tượng PathService
            data_service: Đối tượng DataService
            device: Thiết bị để chạy mô hình (cuda hoặc cpu)
        """
        self.cache_service = cache_service
        self.path_service = path_service
        self.data_service = data_service
        self.device = device
        
        # Load mô hình CLIP gốc
        self.original_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Khởi tạo mô hình tinh chỉnh là None ban đầu
        self.finetuned_model = None
        self.active_model = "original"  # Mặc định sử dụng mô hình gốc
        
        # Đường dẫn mặc định cho mô hình tinh chỉnh
        self.checkpoint_path = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\models\\final_checkpoint.pt"
        
        # Thử tải mô hình tinh chỉnh nếu tệp tồn tại
        if os.path.exists(self.checkpoint_path):
            try:
                self._load_finetuned_model(self.checkpoint_path)
                print(f"Đã tải thành công mô hình CLIP tinh chỉnh từ {self.checkpoint_path}")
            except Exception as e:
                print(f"Lỗi khi tải mô hình tinh chỉnh: {e}")
    
    def _load_finetuned_model(self, checkpoint_path):
        """Tải mô hình CLIP tinh chỉnh từ checkpoint."""
        # Trước tiên tải mô hình CLIP gốc
        base_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        
        # Khởi tạo mô hình tùy chỉnh với mô hình cơ sở
        custom_model = CLIPWithClassifier(base_model, num_classes=3)
        
        # Tải state dict đã lưu
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        custom_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Đặt thành chế độ đánh giá (eval mode)
        custom_model.eval()
        custom_model = custom_model.to(self.device)
        
        # Lưu trữ mô hình
        self.finetuned_model = custom_model
    
    def set_active_model(self, model_name):
        """Đặt mô hình CLIP nào sẽ sử dụng cho embedding.
        
        Args:
            model_name: "original" hoặc "finetuned"
        
        Returns:
            bool: True nếu thành công, False nếu không
        """
        if model_name == "original":
            self.active_model = "original"
            print("Đã chuyển sang sử dụng mô hình CLIP gốc")
            return True
        elif model_name == "finetuned":
            if self.finetuned_model is not None:
                self.active_model = "finetuned"
                print("Đã chuyển sang sử dụng mô hình CLIP tinh chỉnh")
                return True
            else:
                print("Mô hình tinh chỉnh chưa được tải!")
                return False
        else:
            print(f"Tên mô hình không xác định: {model_name}")
            return False
    
    def get_active_model_name(self):
        """Lấy tên của mô hình hiện đang hoạt động."""
        return self.active_model
    
    def get_text_features(self, query, video_name=None):
        """
        Lấy text features từ query.
        
        Args:
            query: Query text
            video_name: Tên video nếu cần lấy trong phạm vi video cụ thể
            
        Returns:
            Text features
        """
        # Check cache first with model type in key
        cache_key = f"{self.active_model}_{query}_{video_name or 'default'}"
        features = self.cache_service.get_text_features(cache_key, video_name)
        if features is not None:
            return features
            
        # Calculate features
        text_input = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            if self.active_model == "finetuned" and self.finetuned_model is not None:
                # Sử dụng encode_text từ mô hình CLIP bên trong mô hình tinh chỉnh
                text_features = self.finetuned_model.clip_model.encode_text(text_input).cpu().numpy()
            else:
                # Sử dụng mô hình CLIP gốc
                text_features = self.original_model.encode_text(text_input).cpu().numpy()
                
        # Normalize
        text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Cache and return
        self.cache_service.set_text_features(cache_key, video_name, text_features)
        return text_features
    
    def get_embeddings(self, video_name=None):
        """
        Lấy và xử lý embeddings từ file.
        
        Args:
            video_name: Tên video nếu cần lấy embeddings của video cụ thể
            
        Returns:
            Embeddings đã chuẩn hóa hoặc None nếu có lỗi
        """
        embeddings_path = self.path_service.get_embeddings_path(video_name)
        
        # Check cache first
        embeddings = self.cache_service.get_embeddings(embeddings_path)
        if embeddings is not None:
            return embeddings
            
        # Load embeddings
        if not os.path.exists(embeddings_path):
            print(f"Warning: Embeddings file not found: {embeddings_path}")
            return None
            
        try:
            embeddings = np.load(embeddings_path)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            
            # Cache and return
            self.cache_service.set_embeddings(embeddings_path, embeddings)
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None
            
    def extract_query_confidence(self, frame_path, query, video_name=None):
        """
        Trích xuất độ tương đồng giữa một khung hình và query text.
        
        Args:
            frame_path: Path to frame file
            query: Query text
            video_name: If provided, will use the specific embeddings for this video
        """
        try:
            # Sử dụng cache để lưu text features và embeddings
            # Tạo một key duy nhất cho query và video_name
            cache_key = f"{self.active_model}_{query}_{video_name or 'default'}"
            
            # Lấy text features từ cache hoặc tính toán mới
            text_features = self.get_text_features(query, video_name)
            
            # Thêm cache cho đường dẫn file
            path_cache_key = f"paths_{video_name or 'default'}"
            
            # Kiểm tra và sử dụng cache đường dẫn
            path_cache = self.cache_service.get_path_cache(video_name)
            if path_cache is None:
                # Lấy đường dẫn embeddings và metadata một lần và lưu vào cache
                embeddings_path = self.path_service.get_embeddings_path(video_name)
                json_path = self.path_service.get_metadata_path(video_name)
                self.cache_service.set_path_cache(video_name, embeddings_path, json_path)
            else:
                # Sử dụng đường dẫn từ cache
                embeddings_path = path_cache['embeddings_path']
                json_path = path_cache['json_path']
            
            # Lấy embeddings từ cache hoặc tính toán mới
            embeddings = self.get_embeddings(video_name)
            if embeddings is None:
                return 0.0
            
            # Lấy danh sách frames từ cache hoặc load mới
            frames = self.cache_service.get_frames_list(json_path)
            if frames is None:
                frames = self.data_service.load_frames_from_json(video_name)
                self.cache_service.set_frames_list(json_path, frames)
            
            # Find frame index
            try:
                index = frames.index(frame_path)
            except ValueError:
                # Try with just the filename
                frame_name = os.path.basename(frame_path)
                matching_frames = [f for f in frames if os.path.basename(f) == frame_name]
                if not matching_frames:
                    return 0.0
                index = frames.index(matching_frames[0])
            
            # Extract embedding for the frame
            frame_embedding = embeddings[index:index+1]
            
            # Calculate similarity
            similarity = np.dot(frame_embedding, text_features.T)[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error extracting query confidence: {e}")
            return 0.0
    
    def search_top_frames(self, query, top_k, video_name=None):
        """
        Tìm kiếm top frames dựa trên query text sử dụng CLIP.
        
        Args:
            query: Query text
            top_k: Số lượng kết quả trả về
            video_name: Nếu được cung cấp, sẽ tìm kiếm trong phạm vi video cụ thể
            
        Returns:
            List of top frame paths
        """
        try:
            # Tạo cache key bao gồm model type
            cache_key = f"search_{self.active_model}_{query}_{top_k}"
            
            # Check cache first
            cached_results = self.cache_service.get_search_results(cache_key, video_name)
            if cached_results is not None:
                return cached_results
                
            # Get text features
            text_features = self.get_text_features(query, video_name)
            
            # Get embeddings
            embeddings = self.get_embeddings(video_name)
            if embeddings is None:
                return []
                
            # Calculate similarities
            similarities = np.dot(embeddings, text_features.T).flatten()
            
            # Get top k indices
            if len(similarities) <= top_k:
                top_indices = np.argsort(similarities)[::-1]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
            # Get frames list
            json_path = self.path_service.get_metadata_path(video_name)
            frames = self.cache_service.get_frames_list(json_path)
            if frames is None:
                frames = self.data_service.load_frames_from_json(video_name)
                self.cache_service.set_frames_list(json_path, frames)
                
            # Create a list of (frame, similarity) pairs
            frame_similarity_pairs = [(frames[i], similarities[i]) for i in top_indices]
            
            # Sort by similarity in descending order
            frame_similarity_pairs.sort(key=lambda pair: pair[1], reverse=True)
            
            # Extract just the frame paths
            top_frames = [pair[0] for pair in frame_similarity_pairs]
            
            # Cache the results
            self.cache_service.set_search_results(cache_key, video_name, top_frames)
            
            return top_frames
        except Exception as e:
            print(f"Error in search_top_frames: {e}")
            return []
    
    def search_top_frames_by_image(self, image_features, top_k, video_name=None):
        """
        Tìm kiếm top frames dựa trên image features sử dụng CLIP.
        
        Args:
            image_features: Image features from uploaded image
            top_k: Số lượng kết quả trả về
            video_name: Nếu được cung cấp, sẽ tìm kiếm trong phạm vi video cụ thể
            
        Returns:
            List of top frame paths
        """
        try:
            # Get embeddings
            embeddings = self.get_embeddings(video_name)
            if embeddings is None:
                return []
                
            # Calculate similarities
            similarities = np.dot(embeddings, image_features.T).flatten()
            
            # Get top k indices
            if len(similarities) <= top_k:
                top_indices = np.argsort(similarities)[::-1]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
            # Get frames list
            json_path = self.path_service.get_metadata_path(video_name)
            frames = self.cache_service.get_frames_list(json_path)
            if frames is None:
                frames = self.data_service.load_frames_from_json(video_name)
                self.cache_service.set_frames_list(json_path, frames)
                
            # Create a list of (frame, similarity) pairs
            frame_similarity_pairs = [(frames[i], similarities[i]) for i in top_indices]
            
            # Sort by similarity in descending order
            frame_similarity_pairs.sort(key=lambda pair: pair[1], reverse=True)
            
            # Extract just the frame paths
            top_frames = [pair[0] for pair in frame_similarity_pairs]
            
            return top_frames
        except Exception as e:
            print(f"Error in search_top_frames_by_image: {e}")
            return []
    
    def extract_image_embedding(self, image_path):
        """
        Extract embeddings from an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            np.array: The embedding vector.
        """
        try:
            # Load and preprocess the image
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.active_model == "finetuned" and self.finetuned_model is not None:
                    # Use the finetuned model for embedding extraction
                    image_features = self.finetuned_model(image)
                else:
                    # Use the original CLIP model
                    image_features = self.original_model.encode_image(image)
                    
                # Convert to numpy and normalize
                embedding = image_features.cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
            
    def extract_and_save_embeddings_from_folder(self, folder_path, model_name=None, video_name=None):
        """
        Trích xuất và lưu embeddings từ thư mục chứa frames.
        
        Args:
            folder_path: Đường dẫn đến thư mục chứa frames
            model_name: Tên mô hình CLIP
            video_name: Tên video
        
        Returns:
            Đường dẫn đến file embeddings đã lưu
        """
        # Lưu trạng thái active model hiện tại
        current_active_model = self.active_model
        
        # Đặt active model nếu được chỉ định
        if model_name:
            if model_name == "finetuned" and self.finetuned_model is None:
                print("Mô hình tinh chỉnh không khả dụng, sử dụng mô hình gốc")
                model_name = "original"
            self.set_active_model(model_name)
        
        # Lấy đường dẫn lưu embeddings
        embeddings_path = self.path_service.get_embeddings_path(video_name)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        
        # Tìm tất cả các file ảnh trong thư mục
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if not frame_files:
            print(f"No image files found in {folder_path}")
            # Khôi phục active model
            self.set_active_model(current_active_model)
            return None
        
        # Initialize embeddings array
        batch_size = 32
        total_files = len(frame_files)
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, total_files, batch_size):
            batch_files = frame_files[i:min(i+batch_size, total_files)]
            batch_images = []
            
            # Process each image in the batch
            for frame_file in batch_files:
                image_path = os.path.join(folder_path, frame_file)
                try:
                    image = self.preprocess(Image.open(image_path).convert('RGB'))
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error preprocessing image {image_path}: {e}")
                    # Add a zero tensor as a placeholder
                    batch_images.append(torch.zeros_like(image))
            
            # Stack images and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                if self.active_model == "finetuned" and self.finetuned_model is not None:
                    batch_embeddings = self.finetuned_model(batch_tensor).cpu().numpy()
                else:
                    batch_embeddings = self.original_model.encode_image(batch_tensor).cpu().numpy()
            
            # Add to embeddings list
            all_embeddings.append(batch_embeddings)
            
            print(f"Processed {min(i+batch_size, total_files)}/{total_files} frames...")
        
        # Combine all batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            
            # Save embeddings to file
            np.save(embeddings_path, embeddings)
            print(f"Saved embeddings for {len(embeddings)} frames to {embeddings_path}")
            
            # Store embedding model type in metadata
            metadata_path = self.path_service.get_metadata_path(video_name)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Add or update embedding model info
                    if isinstance(metadata, list):
                        for item in metadata:
                            item['embedding_model'] = self.active_model
                    elif isinstance(metadata, dict):
                        metadata['embedding_model'] = self.active_model
                    
                    # Save updated metadata
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error updating metadata with model info: {e}")
            
            # Khôi phục active model
            self.set_active_model(current_active_model)
            
            return embeddings_path
        
        # Khôi phục active model
        self.set_active_model(current_active_model)
        
        return None
