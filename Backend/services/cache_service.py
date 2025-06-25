"""
CacheService - Quản lý cache trong hệ thống
"""

class CacheService:
    def __init__(self):
        """
        Khởi tạo CacheService với các cache cho text features, embeddings và frames
        """
        self.text_features_cache = {}
        self.embeddings_cache = {}
        self.frames_list_cache = {}
        self.path_cache = {}
        self.search_results_cache = {}
    
    def get_path_cache(self, video_name):
        """
        Lấy thông tin cache đường dẫn cho video
        
        Args:
            video_name: Tên video hoặc None cho video mặc định
            
        Returns:
            Dictionary chứa thông tin cache hoặc None
        """
        path_cache_key = f"paths_{video_name or 'default'}"
        return self.path_cache.get(path_cache_key)
    
    def set_path_cache(self, video_name, embeddings_path, json_path):
        """
        Lưu thông tin cache đường dẫn cho video
        
        Args:
            video_name: Tên video hoặc None cho video mặc định
            embeddings_path: Đường dẫn đến file embeddings
            json_path: Đường dẫn đến file metadata
        """
        path_cache_key = f"paths_{video_name or 'default'}"
        self.path_cache[path_cache_key] = {
            'embeddings_path': embeddings_path,
            'json_path': json_path
        }
        
    def get_text_features(self, query, video_name):
        """
        Lấy text features từ cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            
        Returns:
            Text features hoặc None
        """
        cache_key = f"{query}_{video_name or 'default'}"
        return self.text_features_cache.get(cache_key)
    
    def set_text_features(self, query, video_name, features):
        """
        Lưu text features vào cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            features: Text features cần lưu
        """
        cache_key = f"{query}_{video_name or 'default'}"
        self.text_features_cache[cache_key] = features
        
    def get_embeddings(self, embeddings_path):
        """
        Lấy embeddings từ cache
        
        Args:
            embeddings_path: Đường dẫn đến file embeddings
            
        Returns:
            Embeddings hoặc None
        """
        return self.embeddings_cache.get(embeddings_path)
    
    def set_embeddings(self, embeddings_path, embeddings):
        """
        Lưu embeddings vào cache
        
        Args:
            embeddings_path: Đường dẫn đến file embeddings
            embeddings: Embeddings cần lưu
        """
        self.embeddings_cache[embeddings_path] = embeddings
        
    def get_frames_list(self, json_path):
        """
        Lấy danh sách frames từ cache
        
        Args:
            json_path: Đường dẫn đến file metadata
            
        Returns:
            Danh sách frames hoặc None
        """
        return self.frames_list_cache.get(json_path)
    
    def set_frames_list(self, json_path, frames_list):
        """
        Lưu danh sách frames vào cache
        
        Args:
            json_path: Đường dẫn đến file metadata
            frames_list: Danh sách frames cần lưu
        """
        self.frames_list_cache[json_path] = frames_list
        
    def get_search_results(self, query, video_name):
        """
        Lấy kết quả tìm kiếm từ cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            
        Returns:
            Kết quả tìm kiếm hoặc None
        """
        cache_key = f"{query}_{video_name or 'default'}"
        return self.search_results_cache.get(cache_key)
        
    def set_search_results(self, query, video_name, results):
        """
        Lưu kết quả tìm kiếm vào cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            results: Kết quả tìm kiếm cần lưu
        """
        cache_key = f"{query}_{video_name or 'default'}"
        self.search_results_cache[cache_key] = results 