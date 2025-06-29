"""
VisualizationService - Quản lý và trực quan hóa dữ liệu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import json
import time
import umap
from PIL import Image
from matplotlib.colors import ListedColormap, hsv_to_rgb
import matplotlib.cm as cm

class VisualizationService:
    def __init__(self, path_service, embedding_service, data_service, cache_service):
        """
        Khởi tạo VisualizationService
        
        Args:
            path_service: Đối tượng PathService
            embedding_service: Đối tượng EmbeddingService
            data_service: Đối tượng DataService
            cache_service: Đối tượng CacheService
        """
        self.path_service = path_service
        self.embedding_service = embedding_service
        self.data_service = data_service
        self.cache_service = cache_service
        
    def get_combined_embeddings(self, video_names=None):
        """
        Lấy và kết hợp embedding từ nhiều video
        
        Args:
            video_names: Danh sách tên video cần lấy embedding hoặc None để lấy tất cả
            
        Returns:
            combined_embeddings: Mảng embedding đã ghép
            video_labels: Nhãn video tương ứng với từng embedding
            frame_indices: Chỉ số frame tương ứng với từng embedding
            metadata: Metadata tương ứng với từng embedding
        """
        # Lấy mapping của tất cả video
        video_mapping = self.path_service.video_data_mapping
        
        # Lọc các video theo yêu cầu
        if video_names:
            selected_videos = {k: v for k, v in video_mapping.items() if k in video_names}
        else:
            # Lấy tất cả video trừ default
            selected_videos = {k: v for k, v in video_mapping.items() 
                              if not k.startswith("default")}
        
        combined_embeddings = []
        video_labels = []
        frame_indices = []
        all_metadata = []
        
        print(f"Loading embeddings for {len(selected_videos)} videos")
        
        for video_name, video_info in selected_videos.items():
            # Lấy đường dẫn tới file embeddings của video này
            embeddings_path = video_info.get('embeddings_file')
            metadata_path = video_info.get('metadata_file')
            
            if not embeddings_path or not os.path.exists(embeddings_path):
                print(f"Warning: Embeddings file not found for video {video_name}")
                continue
                
            # Tận dụng EmbeddingService để lấy embeddings (sẽ sử dụng cache nếu có)
            embeddings = self.embedding_service.get_embeddings(video_name)
            
            if embeddings is None:
                print(f"Warning: Could not load embeddings for video {video_name}")
                continue
                
            print(f"Loaded embeddings shape for {video_name}: {embeddings.shape}")
            
            # Đọc metadata để lấy thông tin bổ sung
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Đảm bảo số lượng phù hợp
                metadata_count = len(metadata)
                embeddings_count = embeddings.shape[0]
                
                if metadata_count < embeddings_count:
                    print(f"Warning: Metadata count ({metadata_count}) less than embeddings count ({embeddings_count})")
                    # Cắt embeddings để phù hợp với metadata
                    embeddings = embeddings[:metadata_count]
                elif metadata_count > embeddings_count:
                    print(f"Warning: Metadata count ({metadata_count}) greater than embeddings count ({embeddings_count})")
                    # Cắt metadata để phù hợp với embeddings
                    metadata = metadata[:embeddings_count]
                    
                # Thêm vào mảng tổng hợp
                combined_embeddings.append(embeddings)
                
                # Lưu nhãn video và chỉ số frame
                video_labels.extend([video_name] * embeddings.shape[0])
                frame_indices.extend(list(range(embeddings.shape[0])))
                
                # Thêm video_name vào metadata để dễ theo dõi
                for item in metadata:
                    item['video_name'] = video_name
                
                all_metadata.extend(metadata)
                
            except Exception as e:
                print(f"Error loading metadata for video {video_name}: {e}")
                continue
        
        # Ghép tất cả thành một mảng lớn nếu có dữ liệu
        if combined_embeddings:
            return np.vstack(combined_embeddings), video_labels, frame_indices, all_metadata
        
        return None, [], [], []
    
    def generate_umap_visualization(self, video_names=None, n_neighbors=15, 
                                   min_dist=0.1, n_components=2, metric='cosine'):
        """
        Tạo trực quan hóa UMAP từ dữ liệu embedding
        
        Args:
            video_names: Danh sách video cần trực quan hóa (None = tất cả)
            n_neighbors, min_dist, n_components, metric: Tham số UMAP
            
        Returns:
            Dữ liệu kết quả UMAP
        """
        try:
            # Tạo cache key để lưu kết quả UMAP
            cache_key = f"{'-'.join(sorted(video_names)) if video_names else 'all'}_{n_neighbors}_{min_dist}_{metric}"
            cache_result = self.cache_service.get_umap_result(cache_key)
            
            if cache_result:
                print(f"Using cached UMAP result for {cache_key}")
                return cache_result
            
            # Nạp dữ liệu embedding từ các video
            embeddings, video_labels, frame_indices, metadata = self.get_combined_embeddings(video_names)
            
            if embeddings is None or len(embeddings) == 0:
                print("No embeddings found for the requested videos")
                return None
            
            print(f"Running UMAP on {embeddings.shape[0]} embeddings with dimensions {embeddings.shape[1]}")
            
            # Giảm chiều dữ liệu với UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=42,
                verbose=True
            )
            
            print("Fitting UMAP model...")
            start_time = time.time()
            embedding_2d = reducer.fit_transform(embeddings)
            elapsed_time = time.time() - start_time
            print(f"UMAP transformation completed in {elapsed_time:.2f} seconds")
            
            # Xử lý metadata để trích xuất thông tin quan trọng
            simplified_metadata = []
            for i, meta in enumerate(metadata):
                # Lấy đường dẫn gốc từ metadata
                original_filepath = meta.get('filepath', '')
                
                # Chuyển đổi đường dẫn tuyệt đối thành URL tương đối
                api_filepath = original_filepath
                if original_filepath and os.path.exists(original_filepath):
                    # Lấy tên file từ đường dẫn đầy đủ
                    frame_filename = os.path.basename(original_filepath)
                    # Tạo URL API để truy cập file
                    api_filepath = f"/api/frame/{frame_filename}"
                
                frame_info = {
                    'video_name': meta.get('video_name', 'unknown'),
                    'frameidx': meta.get('frameidx', i),
                    'filepath': api_filepath,  # Sử dụng URL API thay vì đường dẫn tuyệt đối
                    'original_filepath': original_filepath,  # Giữ lại đường dẫn gốc để tham khảo
                    'frame_id': i,
                }
                
                # Thêm thông tin text detection nếu có
                text_detections = meta.get('text_detections', {}).get('detections', [])
                if text_detections:
                    best_text = max(text_detections, key=lambda x: x.get('confidence', 0))
                    frame_info['text'] = best_text.get('label', '')
                    frame_info['text_confidence'] = best_text.get('confidence', 0)
                
                # Thêm thông tin object detection nếu có
                object_detections = meta.get('object_detections', {}).get('detections', [])
                if object_detections:
                    best_object = max(object_detections, key=lambda x: x.get('confidence', 0))
                    frame_info['object'] = best_object.get('label', '')
                    frame_info['object_confidence'] = best_object.get('confidence', 0)
                
                simplified_metadata.append(frame_info)
            
            # Tạo kết quả
            result = {
                "coordinates": embedding_2d.tolist(),
                "video_labels": video_labels,
                "frame_indices": frame_indices,
                "metadata": simplified_metadata,
                "videos": list(set(video_labels)),
                "dimensionality_reduction": {
                    "method": "umap",
                    "parameters": {
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "n_components": n_components,
                        "metric": metric
                    }
                }
            }
            
            # Lưu cache với TTL dài hơn vì tính toán UMAP tốn kém
            self.cache_service.set_umap_result(cache_key, result, ttl=24*3600)  # 1 ngày
            
            return result
            
        except Exception as e:
            print(f"Error generating UMAP visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_visualization_image(self, coordinates, video_labels, metadata, width=1000, height=800, title="UMAP Visualization"):
        """
        Tạo hình ảnh trực quan hóa từ dữ liệu UMAP
        
        Args:
            coordinates: Tọa độ sau khi giảm chiều
            video_labels: Nhãn video cho mỗi điểm
            metadata: Metadata cho mỗi điểm
            width, height: Kích thước hình ảnh
            title: Tiêu đề của biểu đồ
            
        Returns:
            Base64 encoded image
        """
        try:
            # Tạo một colormap từ video_labels
            unique_videos = list(set(video_labels))
            n_videos = len(unique_videos)
            
            # Tạo màu sắc cho các video
            hsv_colors = [(i/n_videos, 0.8, 0.8) for i in range(n_videos)]
            rgb_colors = [hsv_to_rgb(h, s, v) for h, s, v in hsv_colors]
            cmap = ListedColormap(rgb_colors)
            
            # Map video_labels sang indices
            color_indices = [unique_videos.index(label) for label in video_labels]
            
            # Tạo hình ảnh
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.scatter(
                coordinates[:, 0], 
                coordinates[:, 1],
                c=color_indices, 
                cmap=cmap,
                alpha=0.6,
                s=10
            )
            
            # Thêm legend
            handles = []
            for i, video in enumerate(unique_videos):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=cmap(i/n_videos), markersize=8, label=video))
            plt.legend(handles=handles, title="Videos", loc='best')
            
            plt.title(title)
            plt.tight_layout()
            
            # Lưu hình ảnh vào buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode hình ảnh thành base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error creating visualization image: {e}")
            import traceback
            traceback.print_exc()
            return None 