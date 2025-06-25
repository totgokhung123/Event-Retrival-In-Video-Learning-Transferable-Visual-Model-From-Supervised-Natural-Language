"""
DataService - Quản lý dữ liệu trong hệ thống
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path

class DataService:
    def __init__(self, path_service, cache_service):
        """
        Khởi tạo DataService
        
        Args:
            path_service: Đối tượng PathService
            cache_service: Đối tượng CacheService
        """
        self.path_service = path_service
        self.cache_service = cache_service
        self.frames_mapping = self.load_frames_mapping_from_json()
    
    def load_frames_from_json(self, video_name=None):
        """
        Load danh sách tên file từ file JSON.
        
        Args:
            video_name: If provided, will load frames for this specific video
            
        Returns:
            Danh sách frame names
        """
        # If video_name is provided, use the specific metadata file for that video
        json_path = self.path_service.get_metadata_path(video_name)
        
        # Kiểm tra trong cache trước
        frames = self.cache_service.get_frames_list(json_path)
        if frames:
            return frames
            
        # Chuẩn hóa đường dẫn để đảm bảo tương thích
        json_path = os.path.normpath(json_path)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                samples = json.load(file)
            frames = [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]
            
            # Lưu vào cache
            self.cache_service.set_frames_list(json_path, frames)
            return frames
        except Exception as e:
            print(f"Error loading frames from JSON: {e}, path: {json_path}")
            return []
    
    def load_frames_mapping_from_json(self, video_name=None):
        """
        Load mapping from frame file names to full paths.
        
        Args:
            video_name: If provided, will load frames for this specific video
            
        Returns:
            Dict mapping frame names to full paths
        """
        # If video_name is provided, use the specific metadata file for that video
        json_path = self.path_service.get_metadata_path(video_name)
        
        # Chuẩn hóa đường dẫn để đảm bảo tương thích
        json_path = os.path.normpath(json_path)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {Path(sample["filepath"]).name: sample["filepath"] for sample in data}
        except Exception as e:
            print(f"Error loading frames mapping from JSON: {e}, path: {json_path}")
            return {}

    def get_video_duration(self, video_path):
        """
        Lấy thời lượng video từ file.
        
        Args:
            video_path: Đường dẫn đến file video
            
        Returns:
            Thời lượng video tính bằng giây
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            return round(duration)
        except:
            return 0

    def get_video_resolution(self, video_path):
        """
        Lấy độ phân giải video.
        
        Args:
            video_path: Đường dẫn đến file video
            
        Returns:
            Chuỗi thể hiện độ phân giải
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "Unknown"
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            return f"{width}x{height}"
        except:
            return "Unknown"
    
    def load_json_data(self, video_name=None):
        """
        Load dữ liệu từ file JSON.
        
        Args:
            video_name: Tên video nếu cần lấy dữ liệu của video cụ thể
            
        Returns:
            Dữ liệu đã load
        """
        json_path = self.path_service.get_metadata_path(video_name)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data from JSON: {e}, path: {json_path}")
            return []
    
    def format_event_for_frontend(self, frame_data):
        """
        Format dữ liệu event phù hợp với frontend React.
        
        Args:
            frame_data: Dữ liệu frame
            
        Returns:
            Object đã được format cho frontend
        """
        video_path = frame_data.get('video', '')
        
        # Tạo videoId từ đường dẫn video
        if video_path:
            # Lấy tên file video (không bao gồm phần mở rộng)
            video_filename = Path(video_path).stem
            video_id = f"video-{video_filename}"
        else:
            video_id = 'unknown'
        
        # Tìm category từ text detection nếu có
        category = "Unknown"
        confidence = 0.7
        text_confidence = 0.0
        object_confidence = 0.0
        clip_similarity = 0.0
        description = "Event detected"
        detection_type = "unknown"
        
        # Xử lý text detections
        text_detections = frame_data.get('text_detections', {})
        if text_detections and 'detections' in text_detections and len(text_detections['detections']) > 0:
            best_detection = max(text_detections['detections'], key=lambda x: x.get('confidence', 0))
            category = best_detection.get('label', '').split(' ')[0] if best_detection.get('label') else "Unknown"
            text_confidence = best_detection.get('confidence', 0.7)
            # Chuyển đổi confidence nếu là numpy.float32
            if isinstance(text_confidence, np.float32) or isinstance(text_confidence, np.float64):
                text_confidence = float(text_confidence)
            description = best_detection.get('label', 'Event detected')
            detection_type = "text"
            confidence = text_confidence  # Mặc định confidence = text_confidence
        
        # Xử lý object detections
        object_detections = frame_data.get('object_detections', {})
        if object_detections and 'detections' in object_detections and len(object_detections['detections']) > 0:
            best_obj = max(object_detections['detections'], key=lambda x: x.get('confidence', 0))
            object_confidence = best_obj.get('confidence', 0.5)
            # Chuyển đổi confidence nếu là numpy.float32
            if isinstance(object_confidence, np.float32) or isinstance(object_confidence, np.float64):
                object_confidence = float(object_confidence)
            # Chỉ sử dụng object làm category nếu không có text hoặc object có confidence cao hơn
            if object_confidence > text_confidence:
                category = best_obj.get('label', 'Unknown')
                description = f"Object detected: {category}"
                detection_type = "object"
                confidence = object_confidence  # Cập nhật confidence chính
        
        # CLIP similarity nếu có
        clip_similarity = frame_data.get('clip_similarity', 0.0)
        if clip_similarity is None:
            clip_similarity = 0.0
        if isinstance(clip_similarity, np.float32) or isinstance(clip_similarity, np.float64):
            clip_similarity = float(clip_similarity)
            
        # Nếu clip_similarity cao hơn confidence hiện tại, sử dụng nó cho confidence chính
        if clip_similarity > confidence:
            detection_type = "clip"
            confidence = clip_similarity  # Cập nhật confidence chính
        
        # Lấy fps từ video nếu có, nếu không thì dùng 25fps (giá trị thực tế)
        fps = 25.0  # Sử dụng fps thực tế của video
        if video_path and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    if actual_fps > 0:
                        fps = actual_fps
                    cap.release()
            except Exception as e:
                print(f"Error getting video fps: {e}")
        
        # Tính timestamp từ frame index và fps
        frame_idx = frame_data.get('frameidx', 0)
        timestamp = frame_idx / fps
        
        # Chuyển đổi timestamp nếu là numpy.float32
        if isinstance(timestamp, np.float32) or isinstance(timestamp, np.float64):
            timestamp = float(timestamp)
        
        return {
            "id": f"event-{frame_idx}",
            "videoId": video_id,
            "title": f"Event at frame {frame_idx}",
            "description": description,
            "timestamp": timestamp,
            "duration": 5,
            "category": category,
            "confidence": confidence,
            "text_confidence": text_confidence,
            "object_confidence": object_confidence,
            "clip_similarity": clip_similarity,
            "detection_type": detection_type,
            "thumbnailUrl": frame_data.get('filepath')
        }
    
    def get_frame_by_index(self, frame_idx, video_name=None):
        """
        Lấy dữ liệu frame dựa vào index.
        
        Args:
            frame_idx: Index của frame
            video_name: Tên video nếu cần lấy dữ liệu của video cụ thể
            
        Returns:
            Dữ liệu của frame hoặc None nếu không tìm thấy
        """
        data = self.load_json_data(video_name)
        return next((item for item in data if item.get('frameidx') == frame_idx), None) 