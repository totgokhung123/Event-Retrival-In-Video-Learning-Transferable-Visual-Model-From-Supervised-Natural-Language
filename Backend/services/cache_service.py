"""
CacheService - Quản lý cache trong hệ thống
"""
import time

class CacheService:
    def __init__(self, default_ttl=3600):
        """
        Khởi tạo CacheService với các cache cho text features, embeddings và frames
        
        Args:
            default_ttl: Thời gian sống mặc định cho cache entries (giây), mặc định 1 giờ
        """
        self.text_features_cache = {}
        self.embeddings_cache = {}
        self.frames_list_cache = {}
        self.path_cache = {}
        self.search_results_cache = {}
        self.timestamp_cache = {}  # Lưu timestamp cho TTL
        self.default_ttl = default_ttl
    
    def get_path_cache(self, video_name):
        """
        Lấy thông tin cache đường dẫn cho video
        
        Args:
            video_name: Tên video hoặc None cho video mặc định
            
        Returns:
            Dictionary chứa thông tin cache hoặc None
        """
        path_cache_key = f"paths_{video_name or 'default'}"
        # Kiểm tra TTL trước khi trả về
        if self._is_cache_expired(path_cache_key):
            self._remove_entry(path_cache_key, self.path_cache)
            return None
        return self.path_cache.get(path_cache_key)
    
    def set_path_cache(self, video_name, embeddings_path, json_path, ttl=None):
        """
        Lưu thông tin cache đường dẫn cho video
        
        Args:
            video_name: Tên video hoặc None cho video mặc định
            embeddings_path: Đường dẫn đến file embeddings
            json_path: Đường dẫn đến file metadata
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        path_cache_key = f"paths_{video_name or 'default'}"
        self.path_cache[path_cache_key] = {
            'embeddings_path': embeddings_path,
            'json_path': json_path
        }
        self._set_timestamp(path_cache_key, ttl)
        
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
        # Kiểm tra TTL trước khi trả về
        if self._is_cache_expired(cache_key):
            self._remove_entry(cache_key, self.text_features_cache)
            return None
        return self.text_features_cache.get(cache_key)
    
    def set_text_features(self, query, video_name, features, ttl=None):
        """
        Lưu text features vào cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            features: Text features cần lưu
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        cache_key = f"{query}_{video_name or 'default'}"
        self.text_features_cache[cache_key] = features
        self._set_timestamp(cache_key, ttl)
        
    def get_embeddings(self, embeddings_path):
        """
        Lấy embeddings từ cache
        
        Args:
            embeddings_path: Đường dẫn đến file embeddings
            
        Returns:
            Embeddings hoặc None
        """
        # Kiểm tra TTL trước khi trả về
        if self._is_cache_expired(embeddings_path):
            self._remove_entry(embeddings_path, self.embeddings_cache)
            return None
        return self.embeddings_cache.get(embeddings_path)
    
    def set_embeddings(self, embeddings_path, embeddings, ttl=None):
        """
        Lưu embeddings vào cache
        
        Args:
            embeddings_path: Đường dẫn đến file embeddings
            embeddings: Embeddings cần lưu
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        self.embeddings_cache[embeddings_path] = embeddings
        self._set_timestamp(embeddings_path, ttl)
        
    def get_frames_list(self, json_path):
        """
        Lấy danh sách frames từ cache
        
        Args:
            json_path: Đường dẫn đến file metadata
            
        Returns:
            Danh sách frames hoặc None
        """
        # Kiểm tra TTL trước khi trả về
        if self._is_cache_expired(json_path):
            self._remove_entry(json_path, self.frames_list_cache)
            return None
        return self.frames_list_cache.get(json_path)
    
    def set_frames_list(self, json_path, frames_list, ttl=None):
        """
        Lưu danh sách frames vào cache
        
        Args:
            json_path: Đường dẫn đến file metadata
            frames_list: Danh sách frames cần lưu
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        self.frames_list_cache[json_path] = frames_list
        self._set_timestamp(json_path, ttl)
        
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
        # Kiểm tra TTL trước khi trả về
        if self._is_cache_expired(cache_key):
            self._remove_entry(cache_key, self.search_results_cache)
            return None
        return self.search_results_cache.get(cache_key)
        
    def set_search_results(self, query, video_name, results, ttl=None):
        """
        Lưu kết quả tìm kiếm vào cache
        
        Args:
            query: Câu query
            video_name: Tên video hoặc None cho video mặc định
            results: Kết quả tìm kiếm cần lưu
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        cache_key = f"{query}_{video_name or 'default'}"
        self.search_results_cache[cache_key] = results
        self._set_timestamp(cache_key, ttl)
    
    # Phương thức xử lý TTL
    def _set_timestamp(self, key, ttl=None):
        """
        Lưu timestamp và thời gian hết hạn cho cache entry
        
        Args:
            key: Cache key
            ttl: Thời gian sống cho cache entry (giây), mặc định sử dụng default_ttl
        """
        if ttl is None:
            ttl = self.default_ttl
        self.timestamp_cache[key] = {
            'created_at': time.time(),
            'ttl': ttl
        }
        
    def _is_cache_expired(self, key):
        """
        Kiểm tra xem cache entry có hết hạn hay không
        
        Args:
            key: Cache key
            
        Returns:
            True nếu cache entry đã hết hạn hoặc không tồn tại
        """
        if key not in self.timestamp_cache:
            return True
        
        timestamp_info = self.timestamp_cache[key]
        created_at = timestamp_info['created_at']
        ttl = timestamp_info['ttl']
        
        # Kiểm tra hết hạn
        return (time.time() - created_at) > ttl
    
    # Các phương thức quản lý cache
    def clear_all_caches(self):
        """
        Xóa toàn bộ cache trong hệ thống
        """
        self.text_features_cache.clear()
        self.embeddings_cache.clear()
        self.frames_list_cache.clear()
        self.path_cache.clear()
        self.search_results_cache.clear()
        self.timestamp_cache.clear()
        
    def clear_cache_for_video(self, video_name):
        """
        Xóa toàn bộ cache liên quan đến video cụ thể
        
        Args:
            video_name: Tên video cần xóa cache
        """
        # Tạo prefix cho cache keys
        prefix = f"_{video_name}"
        
        # Xóa cache đường dẫn
        path_key = f"paths_{video_name}"
        if path_key in self.path_cache:
            self._remove_entry(path_key, self.path_cache)
            
        # Xóa các cache khác có chứa video_name
        for cache_dict, cache_type in [
            (self.text_features_cache, "text_features"),
            (self.search_results_cache, "search_results")
        ]:
            keys_to_remove = [k for k in cache_dict.keys() if k.endswith(prefix)]
            for key in keys_to_remove:
                self._remove_entry(key, cache_dict)
                
    def _remove_entry(self, key, cache_dict):
        """
        Xóa một entry cụ thể khỏi cache dictionary
        
        Args:
            key: Cache key cần xóa
            cache_dict: Cache dictionary chứa key
        """
        if key in cache_dict:
            del cache_dict[key]
        if key in self.timestamp_cache:
            del self.timestamp_cache[key]
            
    def refresh_entry(self, key, cache_dict, ttl=None):
        """
        Làm mới thời gian hết hạn cho một cache entry
        
        Args:
            key: Cache key cần làm mới
            cache_dict: Cache dictionary chứa key
            ttl: Thời gian sống mới (giây), mặc định sử dụng default_ttl
        """
        if key in cache_dict:
            self._set_timestamp(key, ttl)
            return True
        return False
        
    def refresh_all_entries(self, new_ttl=None):
        """
        Làm mới thời gian hết hạn cho tất cả cache entries
        
        Args:
            new_ttl: Thời gian sống mới (giây), mặc định sử dụng default_ttl
        """
        for key in list(self.timestamp_cache.keys()):
            self._set_timestamp(key, new_ttl)
            
    def remove_expired_entries(self):
        """
        Xóa tất cả các cache entries đã hết hạn
        
        Returns:
            Số entries đã xóa
        """
        count = 0
        for key in list(self.timestamp_cache.keys()):
            if self._is_cache_expired(key):
                # Tìm cache dict chứa key này
                for cache_dict in [
                    self.text_features_cache, 
                    self.embeddings_cache, 
                    self.frames_list_cache, 
                    self.path_cache, 
                    self.search_results_cache
                ]:
                    if key in cache_dict:
                        del cache_dict[key]
                        count += 1
                # Xóa khỏi timestamp cache
                del self.timestamp_cache[key]
        
        return count 