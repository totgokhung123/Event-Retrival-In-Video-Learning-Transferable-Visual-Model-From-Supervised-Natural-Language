"""
PathService - Quản lý các đường dẫn trong hệ thống
"""

import os
import json

class PathService:
    def __init__(self, base_dir, metadata_dir, embedding_dir):
        """
        Khởi tạo PathService
        
        Args:
            base_dir: Thư mục base chứa frames
            metadata_dir: Thư mục chứa metadata
            embedding_dir: Thư mục chứa embeddings
        """
        self.base_dir = base_dir
        self.metadata_dir = metadata_dir
        self.embedding_dir = embedding_dir
        self.video_data_mapping = self._load_video_data_mapping()
        
    def _load_video_data_mapping(self):
        """Load mapping between videos and their metadata/embedding files."""
        mapping_file = os.path.join(self.metadata_dir, "video_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_video_data_mapping(self):
        """Save the current video data mapping to a file."""
        mapping_file = os.path.join(self.metadata_dir, "video_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.video_data_mapping, f, ensure_ascii=False, indent=4)
            
    def get_default_metadata_path(self):
        """Get the default metadata path for backward compatibility."""
        # Check if there's any video in the mapping
        if self.video_data_mapping:
            # Return the metadata file of the first video
            first_video = list(self.video_data_mapping.keys())[0]
            return self.video_data_mapping[first_video]["metadata_file"]
        # Fallback to a default path
        return os.path.join(self.metadata_dir, "output_samples.json")
    
    def get_default_embeddings_path(self):
        """Get the default embeddings path for backward compatibility."""
        # Check if there's any video in the mapping
        if self.video_data_mapping:
            # Return the embeddings file of the first video
            first_video = list(self.video_data_mapping.keys())[0]
            return self.video_data_mapping[first_video]["embeddings_file"]
        # Fallback to a default path
        return os.path.join(self.embedding_dir, "image_embeddings.npy")
            
    def get_metadata_path(self, video_name=None):
        """
        Get the metadata file path for a specific video or the default one.
        
        Args:
            video_name: Name of the video
        
        Returns:
            Path to the metadata file
        """
        if video_name and video_name in self.video_data_mapping:
            # Chuẩn hóa đường dẫn để đảm bảo tương thích
            path = self.video_data_mapping[video_name]["metadata_file"]
            # Đảm bảo đường dẫn sử dụng dấu gạch chéo phù hợp với hệ thống
            path = os.path.normpath(path)
            print(f"Using metadata file lấy dữ liệu: {path}")
            return path
        return self.get_default_metadata_path()
    
    def get_embeddings_path(self, video_name=None):
        """
        Get the embeddings file path for a specific video or the default one.
        
        Args:
            video_name: Name of the video
        
        Returns:
            Path to the embeddings file
        """
        if video_name and video_name in self.video_data_mapping:
            # Chuẩn hóa đường dẫn để đảm bảo tương thích
            path = self.video_data_mapping[video_name]["embeddings_file"]
            # Đảm bảo đường dẫn sử dụng dấu gạch chéo phù hợp với hệ thống
            path = os.path.normpath(path)
            print(f"Using embeddings file lấy dữ liệu: {path}")
            return path
        return self.get_default_embeddings_path()
    
    def add_video_mapping(self, video_name, metadata_file, embeddings_file, video_path, frames_dir):
        """
        Add a new video to the mapping
        
        Args:
            video_name: Name of the video
            metadata_file: Path to the metadata file
            embeddings_file: Path to the embeddings file
            video_path: Path to the video file
            frames_dir: Directory containing the extracted frames
        """
        self.video_data_mapping[video_name] = {
            "metadata_file": metadata_file,
            "embeddings_file": embeddings_file,
            "video_path": video_path,
            "frames_dir": frames_dir
        }
        self.save_video_data_mapping() 