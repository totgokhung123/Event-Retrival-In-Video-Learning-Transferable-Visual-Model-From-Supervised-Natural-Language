"""
EmbeddingService - Quản lý và xử lý embeddings
"""

import os
import numpy as np
import torch
import clip
import faiss
from pathlib import Path

class EmbeddingService:
    def __init__(self, model, device, cache_service, path_service, data_service):
        """
        Khởi tạo EmbeddingService
        
        Args:
            model: Đối tượng CLIP model 
            device: Thiết bị để chạy mô hình (cuda hoặc cpu)
            cache_service: Đối tượng CacheService
            path_service: Đối tượng PathService
            data_service: Đối tượng DataService
        """
        self.model = model
        self.device = device
        self.cache_service = cache_service
        self.path_service = path_service
        self.data_service = data_service
    
    def get_text_features(self, query, video_name=None):
        """
        Lấy text features từ query.
        
        Args:
            query: Query text
            video_name: Tên video nếu cần lấy trong phạm vi video cụ thể
            
        Returns:
            Text features
        """
        # Check cache first
        features = self.cache_service.get_text_features(query, video_name)
        if features is not None:
            return features
            
        # Calculate features
        text_input = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Cache and return
        self.cache_service.set_text_features(query, video_name, text_features)
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
            cache_key = f"{query}_{video_name or 'default'}"
            
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
                print(f"Warning: Frame {frame_path} not found in frames list")
                return 0.0
                
            # Check if index is valid
            if index >= embeddings.shape[0]:
                print(f"Warning: Index {index} out of bounds for embeddings shape {embeddings.shape}")
                return 0.0
            
            # Calculate similarity
            similarity = np.dot(embeddings[index], text_features.T).flatten()[0]
            
            # Convert from numpy float to regular float
            if isinstance(similarity, np.float32) or isinstance(similarity, np.float64):
                similarity = float(similarity)
                
            return similarity
        except Exception as e:
            print(f"Error in extract_query_confidence: {e}")
            return 0.0
    
    def search_top_frames(self, query, top_k, video_name=None):
        """
        Tìm kiếm top frames dựa trên query text sử dụng CLIP.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            video_name: If provided, search only within this video's frames
            
        Returns:
            List of frame names
        """
        try:
            # Tạo một key duy nhất cho query và video_name
            cache_key = f"{query}_{video_name or 'default'}"
            
            # Nếu đã có kết quả trong cache, trả về ngay lập tức
            results = self.cache_service.get_search_results(query, video_name)
            if results:
                return results[:top_k]
                
            # Nếu không có cache, thực hiện tìm kiếm
            text_features = self.get_text_features(query, video_name)
            
            # Chuẩn bị đường dẫn
            path_cache = self.cache_service.get_path_cache(video_name)
            if path_cache is None:
                # Determine which embeddings and frames to use based on video_name
                embeddings_path = self.path_service.get_embeddings_path(video_name)
                json_path = self.path_service.get_metadata_path(video_name)
                self.cache_service.set_path_cache(video_name, embeddings_path, json_path)
            else:
                embeddings_path = path_cache['embeddings_path']
                json_path = path_cache['json_path']
        
            # Lấy embeddings
            embeddings = self.get_embeddings(video_name)
            if embeddings is None:
                return []
            
            # Lấy danh sách frames
            frames = self.cache_service.get_frames_list(json_path)
            if frames is None:
                frames = self.data_service.load_frames_from_json(video_name)
                self.cache_service.set_frames_list(json_path, frames)

            # FAISS expects float32
            embeddings = embeddings.astype('float32')
            text_features = text_features.astype('float32')

            # Dùng chỉ mục FAISS cho cosine similarity (Inner Product)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            D, I = index.search(text_features, top_k)  # text_features shape: (1, dim)
            
            top_indices = I[0]
            results = [frames[i] for i in top_indices if i < len(frames)]
            
            # Lưu kết quả vào cache
            self.cache_service.set_search_results(query, video_name, results)
            
            return results
        except Exception as e:
            print(f"Error in search_top_frames: {e}")
            return []
            
    def search_top_frames_by_image(self, image_features, top_k, video_name=None):
        """
        Tìm kiếm top frames dựa trên image features sử dụng CLIP.
        
        Args:
            image_features: Image feature vector
            top_k: Number of top results to return
            video_name: If provided, search only within this video's frames
            
        Returns:
            List of frame names
        """
        try:
            # Lấy embeddings
            embeddings = self.get_embeddings(video_name)
            if embeddings is None:
                return []
            
            # Tính toán similarity
            similarities = np.dot(embeddings, image_features.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Lấy danh sách frames
            json_path = self.path_service.get_metadata_path(video_name)
            frames = self.cache_service.get_frames_list(json_path)
            if frames is None:
                frames = self.data_service.load_frames_from_json(video_name)
                self.cache_service.set_frames_list(json_path, frames)
                
            return [frames[i] for i in top_indices if i < len(frames)]
        except Exception as e:
            print(f"Error in search_top_frames_by_image: {e}")
            return []
