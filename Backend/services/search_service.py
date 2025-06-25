"""
SearchService - Quản lý các chức năng tìm kiếm
"""

import numpy as np
from pathlib import Path
from unidecode import unidecode

class SearchService:
    def __init__(self, embedding_service, data_service, path_service, cache_service):
        """
        Khởi tạo SearchService
        
        Args:
            embedding_service: Đối tượng EmbeddingService
            data_service: Đối tượng DataService
            path_service: Đối tượng PathService
            cache_service: Đối tượng CacheService
        """
        self.embedding_service = embedding_service
        self.data_service = data_service
        self.path_service = path_service
        self.cache_service = cache_service
    
    def search_frames_by_keyword(self, keyword, top_k, video_name=None):
        """
        Tìm kiếm frame theo từ khóa text.
        
        Args:
            keyword: Từ khóa cần tìm
            top_k: Số kết quả trả về tối đa
            video_name: Nếu có, chỉ tìm kiếm trong video này
            
        Returns:
            Danh sách frame IDs
        """
        matching_frames = []
        
        # Lấy dữ liệu từ file JSON
        frames_data = self.data_service.load_json_data(video_name)
            
        keyword_without_accents = unidecode(keyword.lower())
        for frame_data in frames_data:
            detections = frame_data.get("text_detections", {}).get("detections", [])
            for detection in detections:
                detection_label = detection.get("label", "")
                if not detection_label:
                    continue 
                detection_label = detection_label.lower()
                label_without_accents = unidecode(detection_label)
                if keyword_without_accents in label_without_accents:
                    matching_frames.append({
                        "frameid": frame_data.get("frameid", ""),
                        "confidence": detection.get("confidence", 0) 
                    })
                    break 
        matching_frames.sort(key=lambda x: x["confidence"], reverse=True)
        return [frame["frameid"] for frame in matching_frames[:top_k]]
    
    def search_by_keyword(self, query, adaptive_threshold, top_k, video_name=None):
        """
        Tìm kiếm theo từ khóa.
        
        Args:
            query: Từ khóa cần tìm
            adaptive_threshold: Ngưỡng độ tin cậy
            top_k: Số kết quả trả về tối đa
            video_name: Nếu có, chỉ tìm kiếm trong video này
            
        Returns:
            Danh sách events
        """
        try:
            keyword_frame_ids = self.search_frames_by_keyword(query, top_k * 3, video_name)
            print(f"Found {len(keyword_frame_ids)} frames containing keyword '{query}'")
            
            # Lấy dữ liệu từ file JSON
            data = self.data_service.load_json_data(video_name)
            
            # Lọc kết quả theo keyword và adaptive threshold
            results = []
            for frame_id in keyword_frame_ids:
                frame_data = next((item for item in data if item.get('frameid') == frame_id), None)
                if frame_data:
                    detections = frame_data.get("text_detections", {}).get("detections", [])
                    keyword_without_accents = unidecode(query.lower())
                    
                    # Tìm detection có chứa keyword và có confidence cao nhất
                    best_match = None
                    max_conf = 0
                    for detection in detections:
                        detection_label = detection.get("label", "").lower()
                        # Tìm kiếm cả từ khóa chính xác và từ khóa không dấu
                        if keyword_without_accents in unidecode(detection_label):
                            conf = detection.get("confidence", 0)
                            if conf > max_conf:
                                max_conf = conf
                                best_match = detection
                    
                    if best_match and max_conf >= adaptive_threshold:
                        # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                        frame_data_copy = frame_data.copy()
                        
                        # Đặt các giá trị confidence
                        frame_data_copy['text_confidence'] = max_conf
                        frame_data_copy['clip_similarity'] = 0.0
                        
                        # Format event cho frontend
                        event = self.data_service.format_event_for_frontend(frame_data_copy)
                        results.append(event)
            
            print(f"Found {len(results)} results after applying threshold {adaptive_threshold}")
            
            # Sắp xếp theo confidence (giảm dần)
            results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def search_semantic_with_clip(self, query, adaptive_threshold, top_k, video_name=None):
        """
        Tìm kiếm ngữ nghĩa với CLIP.
        
        Args:
            query: Query text
            adaptive_threshold: Ngưỡng độ tin cậy
            top_k: Số kết quả trả về tối đa
            video_name: Nếu có, chỉ tìm kiếm trong video này
            
        Returns:
            Danh sách events
        """
        try:
            # Use embedding service to search frames
            query_frames = self.embedding_service.search_top_frames(query, top_k * 3, video_name)
            print(f"Found {len(query_frames)} candidate frames")
            
            # Get JSON data
            data = self.data_service.load_json_data(video_name)
            
            # Filter frame data
            semantic_results = []
            for frame_name in query_frames:
                try:
                    frame_idx = int(Path(frame_name).stem)
                    frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
                    
                    if frame_data:
                        # Calculate CLIP similarity
                        confidence = self.embedding_service.extract_query_confidence(
                            frame_name, query, video_name)
                        
                        # Add to results if above threshold
                        if confidence >= adaptive_threshold:
                            # Store confidence in frame_data for format_event_for_frontend
                            frame_data_copy = frame_data.copy()
                            frame_data_copy['clip_similarity'] = confidence
                            
                            event = self.data_service.format_event_for_frontend(frame_data_copy)
                            semantic_results.append(event)
                except Exception as e:
                    print(f"Error processing frame {frame_name}: {e}")
            
            # Debug info
            print(f"Found {len(semantic_results)} results after applying threshold {adaptive_threshold}")
            
            # Sort results by confidence
            semantic_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Return top_k results
            return semantic_results[:top_k]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def search_by_object(self, query, adaptive_threshold, top_k, video_name=None):
        """
        Tìm kiếm theo object.
        
        Args:
            query: Từ khóa object cần tìm
            adaptive_threshold: Ngưỡng độ tin cậy
            top_k: Số kết quả trả về tối đa
            video_name: Nếu có, chỉ tìm kiếm trong video này
            
        Returns:
            Danh sách events
        """
        try:
            # Get the appropriate JSON file path
            data = self.data_service.load_json_data(video_name)
                
            object_results = []
            query_lower = query.lower()
            query_without_accents = unidecode(query_lower)
            
            for frame_data in data:
                max_conf = 0
                object_found = False
                best_label = ""
                
                # 1. Tìm trong object detections (ưu tiên)
                object_detect_data = frame_data.get("object_detections", {})
                object_detections = object_detect_data.get("detections", [])
                
                for obj in object_detections:
                    obj_label = obj.get("label", "").lower()
                    if query_lower in obj_label or query_without_accents in unidecode(obj_label):
                        conf = obj.get("confidence", 0)
                        if conf > max_conf:
                            max_conf = conf
                            object_found = True
                            best_label = obj_label
                            print(f"Found object '{obj_label}' in frame {frame_data.get('frameidx')} with confidence {conf}")
                
                # 2. Tìm trong metadata hoặc caption (nếu có)
                metadata = frame_data.get("metadata", {})
                caption = metadata.get("caption", "").lower() if metadata else ""
                if caption and (query_lower in caption or query_without_accents in unidecode(caption)):
                    # Nếu tìm thấy trong caption, gán confidence mặc định cao hơn
                    caption_conf = 0.65
                    if caption_conf > max_conf:
                        object_found = True
                        max_conf = caption_conf
                        best_label = query
                
                # 3. Tìm trong tags hoặc labels (nếu có)
                tags = frame_data.get("tags", [])
                for tag in tags:
                    tag_lower = str(tag).lower()
                    if query_lower in tag_lower or query_without_accents in unidecode(tag_lower):
                        tag_conf = 0.75
                        if tag_conf > max_conf:
                            object_found = True
                            max_conf = tag_conf
                            best_label = tag
                        
                # 4. Text detections (thứ yếu)
                text_detect_data = frame_data.get("text_detections", {})
                text_detections = text_detect_data.get("detections", [])
                
                for detection in text_detections:
                    detection_label = detection.get("label", "").lower()
                    # Tìm object name trong text đã nhận diện
                    if query_lower in detection_label or query_without_accents in unidecode(detection_label):
                        conf = detection.get("confidence", 0) * 0.7  # Giảm confidence vì đây là text
                        if conf > max_conf:
                            max_conf = conf
                            object_found = True
                            best_label = detection_label
                
                # Giữ ngưỡng adaptive_threshold nhưng tối đa là 0.65
                actual_threshold = min(adaptive_threshold, 0.65)
                
                # Nếu có object và đủ confidence
                if object_found and max_conf >= actual_threshold:
                    # Thêm object_confidence trực tiếp vào frame_data
                    frame_data_copy = frame_data.copy()
                    frame_data_copy['object_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    # Đánh dấu rằng đây là kết quả từ object detection
                    frame_data_copy['detection_type'] = 'object'
                    # Cập nhật thông tin object được tìm thấy
                    if not frame_data_copy.get('object_label'):
                        frame_data_copy['object_label'] = best_label
                    
                    event = self.data_service.format_event_for_frontend(frame_data_copy)
                    object_results.append(event)
            
            # Sắp xếp kết quả theo confidence
            object_results.sort(key=lambda x: x["confidence"], reverse=True)
            return object_results[:top_k]
        except Exception as e:
            print(f"Error in object search: {e}")
            return []
    
    def search_by_text_and_keyword(self, query, keyword, adaptive_threshold, text_confidence, 
                                  top_k, video_name=None):
        """
        Tìm kiếm kết hợp text + keyword.
        
        Args:
            query: Query text
            keyword: Keyword to search for
            adaptive_threshold: Threshold for CLIP confidence
            text_confidence: Threshold for text confidence
            top_k: Number of top results to return
            video_name: If provided, search only within this video
            
        Returns:
            Danh sách events
        """
        try:
            # Tìm frames với CLIP
            query_frames = self.embedding_service.search_top_frames(query, top_k * 3, video_name)
            
            # Tìm frames với keyword
            keyword_frame_ids = self.search_frames_by_keyword(keyword, top_k * 3, video_name)
            
            # Lấy dữ liệu JSON
            data = self.data_service.load_json_data(video_name)
            
            # Kết quả từ CLIP
            clip_results = []
            for frame_name in query_frames:
                frame_idx = int(Path(frame_name).stem)
                frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
                if frame_data:
                    # Tính CLIP confidence
                    confidence = self.embedding_service.extract_query_confidence(
                        frame_name, query, video_name)
                    
                    # Áp dụng adaptive threshold cho CLIP
                    if confidence >= adaptive_threshold:
                        # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                        frame_data_copy = frame_data.copy()
                        frame_data_copy['clip_similarity'] = confidence
                        
                        # Lọc chỉ những frame có cả keyword và đủ text_confidence
                        frameid = frame_data.get('frameid')
                        if frameid in keyword_frame_ids:
                            # Tìm text detection có chứa keyword
                            detections = frame_data.get("text_detections", {}).get("detections", [])
                            keyword_without_accents = unidecode(keyword.lower())
                            
                            best_match = None
                            max_conf = 0
                            for detection in detections:
                                detection_label = detection.get("label", "").lower()
                                if keyword_without_accents in unidecode(detection_label):
                                    conf = detection.get("confidence", 0)
                                    if conf > max_conf:
                                        max_conf = conf
                                        best_match = detection
                            
                            # Chỉ giữ lại frame nếu text confidence đủ cao
                            if best_match and max_conf >= text_confidence:
                                # Đặt các giá trị confidence
                                frame_data_copy['text_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                                if 'object_confidence' not in frame_data_copy:
                                    frame_data_copy['object_confidence'] = 0.0
                                
                                # Format event cho frontend
                                event = self.data_service.format_event_for_frontend(frame_data_copy)
                                
                                # Đảm bảo confidence được đặt đúng (lấy max của cả hai)
                                event["confidence"] = max(confidence, max_conf)
                                event["clip_similarity"] = confidence
                                event["text_confidence"] = max_conf
                                event["detection_type"] = "text+clip"
                                
                                clip_results.append(event)
            
            # Sắp xếp theo confidence (giảm dần)
            clip_results.sort(key=lambda x: x["confidence"], reverse=True)
            return clip_results[:top_k]
        except Exception as e:
            print(f"Error in text and keyword query: {e}")
            return []
    
    def search_by_text_and_object(self, query, object_keyword, adaptive_threshold, object_confidence,
                                 top_k, video_name=None):
        """
        Tìm kiếm kết hợp text + object.
        
        Args:
            query: Query text
            object_keyword: Object keyword to search for
            adaptive_threshold: Threshold for CLIP confidence
            object_confidence: Threshold for object confidence
            top_k: Number of top results to return
            video_name: If provided, search only within this video
            
        Returns:
            Danh sách events
        """
        try:
            # Tìm frames với CLIP
            query_frames = self.embedding_service.search_top_frames(query, top_k * 3, video_name)
            
            # Lấy dữ liệu JSON
            data = self.data_service.load_json_data(video_name)
            
            # Kết quả từ CLIP
            results = []
            for frame_name in query_frames:
                frame_idx = int(Path(frame_name).stem)
                frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
                if frame_data:
                    # Tính CLIP confidence
                    clip_confidence = self.embedding_service.extract_query_confidence(
                        frame_name, query, video_name)
                    
                    # Áp dụng adaptive threshold cho CLIP
                    if clip_confidence >= adaptive_threshold:
                        # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                        frame_data_copy = frame_data.copy()
                        frame_data_copy['clip_similarity'] = clip_confidence
                        
                        # Tìm object trong frame này
                        object_found = False
                        max_obj_conf = 0
                        best_label = ""
                        
                        # 1. Tìm trong object detections
                        object_detect_data = frame_data.get("object_detections", {})
                        object_detections = object_detect_data.get("detections", [])
                        
                        object_to_use_lower = object_keyword.lower()
                        object_without_accents = unidecode(object_to_use_lower)
                        
                        for obj in object_detections:
                            obj_label = obj.get("label", "").lower()
                            if object_to_use_lower in obj_label or object_without_accents in unidecode(obj_label):
                                conf = obj.get("confidence", 0)
                                if conf > max_obj_conf:
                                    max_obj_conf = conf
                                    object_found = True
                                    best_label = obj_label
                        
                        # 2. Tìm trong metadata hoặc caption (nếu có)
                        metadata = frame_data.get("metadata", {})
                        caption = metadata.get("caption", "").lower() if metadata else ""
                        if caption and (object_to_use_lower in caption or object_without_accents in unidecode(caption)):
                            caption_conf = 0.65
                            if caption_conf > max_obj_conf:
                                object_found = True
                                max_obj_conf = caption_conf
                                best_label = object_keyword
                        
                        # 3. Tìm trong tags hoặc labels (nếu có)
                        tags = frame_data.get("tags", [])
                        for tag in tags:
                            tag_lower = str(tag).lower()
                            if object_to_use_lower in tag_lower or object_without_accents in unidecode(tag_lower):
                                tag_conf = 0.75
                                if tag_conf > max_obj_conf:
                                    object_found = True
                                    max_obj_conf = tag_conf
                                    best_label = tag
                        
                        # Chỉ giữ lại frame nếu object confidence đủ cao
                        if object_found and max_obj_conf >= object_confidence:
                            # Đặt các giá trị confidence
                            frame_data_copy['object_confidence'] = float(max_obj_conf) if isinstance(max_obj_conf, np.float32) else max_obj_conf
                            if 'text_confidence' not in frame_data_copy:
                                frame_data_copy['text_confidence'] = 0.0
                            
                            # Cập nhật thông tin object được tìm thấy
                            frame_data_copy['object_label'] = best_label
                            
                            # Format event cho frontend
                            event = self.data_service.format_event_for_frontend(frame_data_copy)
                            
                            # Đảm bảo confidence được đặt đúng (lấy max của cả hai)
                            event["confidence"] = max(clip_confidence, max_obj_conf)
                            event["clip_similarity"] = clip_confidence
                            event["object_confidence"] = max_obj_conf
                            event["detection_type"] = "object+clip"
                            
                            results.append(event)
            
            # Sắp xếp theo confidence (giảm dần)
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error in text and object query: {e}")
            return []
    
    def search_by_text_object_and_keyword(self, query, keyword, object_keyword, adaptive_threshold, 
                                         text_confidence, object_confidence, top_k, video_name=None):
        """
        Tìm kiếm kết hợp text + object + keyword.
        
        Args:
            query: Query text
            keyword: Keyword to search for
            object_keyword: Object keyword to search for 
            adaptive_threshold: Threshold for CLIP confidence
            text_confidence: Threshold for text confidence
            object_confidence: Threshold for object confidence
            top_k: Number of top results to return
            video_name: If provided, search only within this video
            
        Returns:
            Danh sách events
        """
        try:
            # Tìm frames với CLIP
            query_frames = self.embedding_service.search_top_frames(query, top_k * 3, video_name)
            
            # Tìm frames với keyword
            keyword_frame_ids = self.search_frames_by_keyword(keyword, top_k * 3, video_name)
            
            # Lấy dữ liệu JSON
            data = self.data_service.load_json_data(video_name)
            
            # Kết quả từ CLIP
            results = []
            for frame_name in query_frames:
                frame_idx = int(Path(frame_name).stem)
                frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
                if frame_data:
                    # Tính CLIP confidence
                    clip_confidence = self.embedding_service.extract_query_confidence(
                        frame_name, query, video_name)
                    
                    # Áp dụng adaptive threshold cho CLIP
                    if clip_confidence >= adaptive_threshold:
                        # Kiểm tra nếu frame này cũng có trong kết quả keyword
                        frameid = frame_data.get('frameid')
                        if frameid in keyword_frame_ids:
                            # Tìm text detection có chứa keyword
                            detections = frame_data.get("text_detections", {}).get("detections", [])
                            keyword_without_accents = unidecode(keyword.lower())
                            
                            best_text_match = None
                            max_text_conf = 0
                            for detection in detections:
                                detection_label = detection.get("label", "").lower()
                                if keyword_without_accents in unidecode(detection_label):
                                    conf = detection.get("confidence", 0)
                                    if conf > max_text_conf:
                                        max_text_conf = conf
                                        best_text_match = detection
                            
                            # Chỉ tiếp tục nếu text confidence đủ cao
                            if best_text_match and max_text_conf >= text_confidence:
                                # Tìm object trong frame này
                                object_found = False
                                max_obj_conf = 0
                                best_label = ""
                                
                                # 1. Tìm trong object detections
                                object_detect_data = frame_data.get("object_detections", {})
                                object_detections = object_detect_data.get("detections", [])
                                
                                object_to_use_lower = object_keyword.lower()
                                object_without_accents = unidecode(object_to_use_lower)
                                
                                for obj in object_detections:
                                    obj_label = obj.get("label", "").lower()
                                    if object_to_use_lower in obj_label or object_without_accents in unidecode(obj_label):
                                        conf = obj.get("confidence", 0)
                                        if conf > max_obj_conf:
                                            max_obj_conf = conf
                                            object_found = True
                                            best_label = obj_label
                                
                                # 2. Tìm trong metadata hoặc caption (nếu có)
                                metadata = frame_data.get("metadata", {})
                                caption = metadata.get("caption", "").lower() if metadata else ""
                                if caption and (object_to_use_lower in caption or object_without_accents in unidecode(caption)):
                                    caption_conf = 0.65
                                    if caption_conf > max_obj_conf:
                                        object_found = True
                                        max_obj_conf = caption_conf
                                        best_label = object_keyword
                                
                                # 3. Tìm trong tags hoặc labels (nếu có)
                                tags = frame_data.get("tags", [])
                                for tag in tags:
                                    tag_lower = str(tag).lower()
                                    if object_to_use_lower in tag_lower or object_without_accents in unidecode(tag_lower):
                                        tag_conf = 0.75
                                        if tag_conf > max_obj_conf:
                                            object_found = True
                                            max_obj_conf = tag_conf
                                            best_label = tag
                                
                                # Chỉ giữ lại frame nếu object confidence đủ cao
                                if object_found and max_obj_conf >= object_confidence:
                                    # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                                    frame_data_copy = frame_data.copy()
                                    
                                    # Đặt các giá trị confidence
                                    frame_data_copy['clip_similarity'] = clip_confidence
                                    frame_data_copy['text_confidence'] = max_text_conf
                                    frame_data_copy['object_confidence'] = max_obj_conf
                                    
                                    # Cập nhật thông tin object được tìm thấy
                                    frame_data_copy['object_label'] = best_label
                                    
                                    # Format event cho frontend
                                    event = self.data_service.format_event_for_frontend(frame_data_copy)
                                    
                                    # Đảm bảo confidence được đặt đúng (lấy max của cả ba)
                                    event["confidence"] = max(clip_confidence, max_text_conf, max_obj_conf)
                                    event["clip_similarity"] = clip_confidence
                                    event["text_confidence"] = max_text_conf
                                    event["object_confidence"] = max_obj_conf
                                    event["detection_type"] = "text+object+clip"
                                    
                                    results.append(event)
            
            # Sắp xếp theo confidence (giảm dần)
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error in text, object and keyword query: {e}")
            return [] 