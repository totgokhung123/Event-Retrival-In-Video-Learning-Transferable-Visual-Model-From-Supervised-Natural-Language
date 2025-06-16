import numpy as np
import json
import os
from pathlib import Path
from unidecode import unidecode
from word_processing import VietnameseTextProcessor

# Base directories - these should match the ones in app.py
METADATA_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\metadata"
EMBEDDING_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\embedding"

def get_default_metadata_path():
    """Get the default metadata path for backward compatibility."""
    # Fallback to a default path
    return os.path.join(METADATA_DIR, "output_samples.json")

def get_video_metadata_path(video_name=None, video_data_mapping=None):
    """
    Get the metadata file path for a specific video or the default one.
    
    Args:
        video_name: Name of the video
        video_data_mapping: Mapping of videos to their metadata files
    
    Returns:
        Path to the metadata file
    """
    if video_name and video_data_mapping and video_name in video_data_mapping:
        return video_data_mapping[video_name]["metadata_file"]
    return get_default_metadata_path()

def query_by_text_clip(query, top_k, search_top_frames, extract_query_confidence, format_event_for_frontend, video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng văn bản thuần (CLIP Similarity).
    
    Args:
        query: Query text
        top_k: Number of top results to return
        search_top_frames: Function to search top frames
        extract_query_confidence: Function to extract confidence
        format_event_for_frontend: Function to format event data
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 2, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                # Extract video name from frame data if available
                video_path = frame_data.get('video', '')
                frame_video_name = Path(video_path).stem if video_path else None
                
                # Use either the provided video_name or the one from the frame
                effective_video_name = video_name or frame_video_name
                
                confidence = extract_query_confidence(frame_name, processed_text, effective_video_name)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                
                # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                frame_data_copy = frame_data.copy()
                
                # Đặt giá trị clip_similarity vào frame_data trước khi gọi format_event_for_frontend
                frame_data_copy['clip_similarity'] = confidence
                
                # Đảm bảo các trường confidence khác có giá trị mặc định
                if 'text_confidence' not in frame_data_copy:
                    frame_data_copy['text_confidence'] = 0.0
                if 'object_confidence' not in frame_data_copy:
                    frame_data_copy['object_confidence'] = 0.0
                
                # Format event cho frontend
                event = format_event_for_frontend(frame_data_copy)
                
                # Đảm bảo confidence và clip_similarity được đặt đúng
                event["confidence"] = confidence
                event["clip_similarity"] = confidence
                event["detection_type"] = "clip"
                
                results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in text clip query: {e}")
        return []

def query_by_text_with_adaptive_threshold(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, format_event_for_frontend, video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng văn bản + Adaptive threshold.
    
    Args:
        query: Query text
        adaptive_threshold: Threshold for confidence
        top_k: Number of top results to return
        search_top_frames: Function to search top frames
        extract_query_confidence: Function to extract confidence
        format_event_for_frontend: Function to format event data
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data với adaptive threshold
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                # Extract video name from frame data if available
                video_path = frame_data.get('video', '')
                frame_video_name = Path(video_path).stem if video_path else None
                
                # Use either the provided video_name or the one from the frame
                effective_video_name = video_name or frame_video_name
                
                confidence = extract_query_confidence(frame_name, processed_text, effective_video_name)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                # Áp dụng adaptive threshold
                if confidence >= adaptive_threshold:
                    # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                    frame_data_copy = frame_data.copy()
                    
                    # Đặt giá trị clip_similarity vào frame_data trước khi gọi format_event_for_frontend
                    frame_data_copy['clip_similarity'] = confidence
                    
                    # Đảm bảo các trường confidence khác có giá trị mặc định
                    if 'text_confidence' not in frame_data_copy:
                        frame_data_copy['text_confidence'] = 0.0
                    if 'object_confidence' not in frame_data_copy:
                        frame_data_copy['object_confidence'] = 0.0
                    
                    # Format event cho frontend
                    event = format_event_for_frontend(frame_data_copy)
                    
                    # Đảm bảo confidence và clip_similarity được đặt đúng
                    event["confidence"] = confidence
                    event["clip_similarity"] = confidence
                    event["detection_type"] = "clip"
                    
                    results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in text with adaptive threshold query: {e}")
        return []

def query_by_keyword(query, adaptive_threshold, top_k, search_frames_by_keyword, format_event_for_frontend, video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng keyword.
    
    Args:
        query: Query text
        adaptive_threshold: Threshold for confidence
        top_k: Number of top results to return
        search_frames_by_keyword: Function to search frames by keyword
        format_event_for_frontend: Function to format event data
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    try:
        keyword_frame_ids = search_frames_by_keyword(query, top_k * 3, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
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
                    
                    # Thay thế text_detections bằng chỉ detection chứa keyword
                    # Điều này đảm bảo format_event_for_frontend sẽ chọn detection này
                    frame_data_copy['text_detections'] = {
                        'detections': [best_match]
                    }
                    
                    # Đặt các giá trị confidence
                    frame_data_copy['text_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    frame_data_copy['clip_similarity'] = 0.0
                    frame_data_copy['object_confidence'] = 0.0
                    
                    # Đặt trường description để hiển thị đúng label đã tìm thấy
                    frame_data_copy['description'] = best_match.get('label', 'Event detected')
                    
                    # Format event cho frontend
                    event = format_event_for_frontend(frame_data_copy)
                    
                    # Đảm bảo confidence chính là confidence của text detection
                    event["confidence"] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    event["text_confidence"] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    event["detection_type"] = "text"
                    
                    results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in keyword query: {e}")
        return []

def query_by_text_and_keyword(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                              search_frames_by_keyword, format_event_for_frontend, keyword=None, text_confidence=None,
                              video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng văn bản + keyword.
    
    Args:
        query: Query text
        adaptive_threshold: Threshold for CLIP confidence
        top_k: Number of top results to return
        search_top_frames: Function to search top frames
        extract_query_confidence: Function to extract confidence
        search_frames_by_keyword: Function to search by keyword
        format_event_for_frontend: Function to format event data
        keyword: Keyword to search for (if None, uses query)
        text_confidence: Threshold for text confidence (if None, uses adaptive_threshold)
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    # Sử dụng keyword truyền vào nếu có, nếu không thì dùng query
    keyword_to_use = keyword if keyword else query
    # Sử dụng text_confidence truyền vào nếu có, nếu không thì dùng adaptive_threshold
    keyword_threshold = text_confidence if text_confidence is not None else adaptive_threshold
    
    print(f"Tìm kiếm kết hợp: query='{query}' với CLIP threshold={adaptive_threshold}, keyword='{keyword_to_use}' với text threshold={keyword_threshold}")
    
    try:
        # Xử lý query text
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3, video_name)
        
        # Tìm frame với keyword
        keyword_frame_ids = search_frames_by_keyword(keyword_to_use, top_k * 3, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kết quả từ CLIP
        clip_results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                # Extract video name from frame data if available
                video_path = frame_data.get('video', '')
                frame_video_name = Path(video_path).stem if video_path else None
                
                # Use either the provided video_name or the one from the frame
                effective_video_name = video_name or frame_video_name
                
                confidence = extract_query_confidence(frame_name, processed_text, effective_video_name)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                
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
                        keyword_without_accents = unidecode(keyword_to_use.lower())
                        
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
                        if best_match and max_conf >= keyword_threshold:
                            # Đặt các giá trị confidence
                            frame_data_copy['text_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                            if 'object_confidence' not in frame_data_copy:
                                frame_data_copy['object_confidence'] = 0.0
                            
                            # Format event cho frontend
                            event = format_event_for_frontend(frame_data_copy)
                            
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

def query_by_object(query, adaptive_threshold, top_k, format_event_for_frontend, video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng object.
    
    Args:
        query: Từ khóa object cần tìm
        adaptive_threshold: Ngưỡng độ tin cậy
        top_k: Số kết quả trả về tối đa
        format_event_for_frontend: Hàm format dữ liệu cho frontend
        video_name: Nếu có, chỉ tìm kiếm trong video này
        video_data_mapping: Mapping of videos to their metadata files
    """
    try:
        # Determine which JSON file to use
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
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
                
                event = format_event_for_frontend(frame_data_copy)
                object_results.append(event)
        
        # Sắp xếp kết quả theo confidence
        object_results.sort(key=lambda x: x["confidence"], reverse=True)
        return object_results[:top_k]
    except Exception as e:
        print(f"Error in object search: {e}")
        return []

def query_by_text_and_object(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                            format_event_for_frontend, object_keyword=None, object_confidence=None,
                            video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng văn bản + Object.
    
    Args:
        query: Query text
        adaptive_threshold: Threshold for CLIP confidence
        top_k: Number of top results to return
        search_top_frames: Function to search top frames
        extract_query_confidence: Function to extract confidence
        format_event_for_frontend: Function to format event data
        object_keyword: Object keyword to search for (if None, uses query)
        object_confidence: Threshold for object confidence (if None, uses adaptive_threshold)
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    # Sử dụng object_keyword truyền vào nếu có, nếu không thì dùng query
    object_to_use = object_keyword if object_keyword else query
    # Sử dụng object_confidence truyền vào nếu có, nếu không thì dùng adaptive_threshold
    obj_threshold = object_confidence if object_confidence is not None else adaptive_threshold
    
    print(f"Tìm kiếm kết hợp: query='{query}' với CLIP threshold={adaptive_threshold}, object='{object_to_use}' với object threshold={obj_threshold}")
    
    try:
        # Xử lý query text
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kết quả từ CLIP
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                # Extract video name from frame data if available
                video_path = frame_data.get('video', '')
                frame_video_name = Path(video_path).stem if video_path else None
                
                # Use either the provided video_name or the one from the frame
                effective_video_name = video_name or frame_video_name
                
                # Tính CLIP similarity
                clip_confidence = extract_query_confidence(frame_name, processed_text, effective_video_name)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(clip_confidence, np.float32):
                    clip_confidence = float(clip_confidence)
                
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
                    
                    object_to_use_lower = object_to_use.lower()
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
                            best_label = object_to_use
                    
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
                    if object_found and max_obj_conf >= obj_threshold:
                        # Đặt các giá trị confidence
                        frame_data_copy['object_confidence'] = float(max_obj_conf) if isinstance(max_obj_conf, np.float32) else max_obj_conf
                        if 'text_confidence' not in frame_data_copy:
                            frame_data_copy['text_confidence'] = 0.0
                        
                        # Cập nhật thông tin object được tìm thấy
                        frame_data_copy['object_label'] = best_label
                        
                        # Format event cho frontend
                        event = format_event_for_frontend(frame_data_copy)
                        
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

def query_by_text_object_and_keyword(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                                    search_frames_by_keyword, format_event_for_frontend, 
                                    keyword=None, text_confidence=None,
                                    object_keyword=None, object_confidence=None,
                                    video_name=None, video_data_mapping=None):
    """
    Truy vấn bằng văn bản + Object + keyword.
    
    Args:
        query: Query text
        adaptive_threshold: Threshold for CLIP confidence
        top_k: Number of top results to return
        search_top_frames: Function to search top frames
        extract_query_confidence: Function to extract confidence
        search_frames_by_keyword: Function to search by keyword
        format_event_for_frontend: Function to format event data
        keyword: Keyword to search for (if None, uses query)
        text_confidence: Threshold for text confidence (if None, uses adaptive_threshold)
        object_keyword: Object keyword to search for (if None, uses query)
        object_confidence: Threshold for object confidence (if None, uses adaptive_threshold)
        video_name: If provided, search only within this video
        video_data_mapping: Mapping of videos to their metadata files
    """
    # Sử dụng các tham số truyền vào nếu có, nếu không thì dùng query
    keyword_to_use = keyword if keyword else query
    object_to_use = object_keyword if object_keyword else query
    
    # Sử dụng các threshold truyền vào nếu có, nếu không thì dùng adaptive_threshold
    keyword_threshold = text_confidence if text_confidence is not None else adaptive_threshold
    obj_threshold = object_confidence if object_confidence is not None else adaptive_threshold
    
    print(f"Tìm kiếm kết hợp 3 yếu tố: query='{query}' với CLIP threshold={adaptive_threshold}, "
          f"keyword='{keyword_to_use}' với text threshold={keyword_threshold}, "
          f"object='{object_to_use}' với object threshold={obj_threshold}")
    
    try:
        # Xử lý query text
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3, video_name)
        
        # Tìm frame với keyword
        keyword_frame_ids = search_frames_by_keyword(keyword_to_use, top_k * 3, video_name)
        
        # Get the appropriate JSON file path
        json_path = get_video_metadata_path(video_name, video_data_mapping)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kết quả từ CLIP
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                # Extract video name from frame data if available
                video_path = frame_data.get('video', '')
                frame_video_name = Path(video_path).stem if video_path else None
                
                # Use either the provided video_name or the one from the frame
                effective_video_name = video_name or frame_video_name
                
                # Tính CLIP similarity
                clip_confidence = extract_query_confidence(frame_name, processed_text, effective_video_name)
                if isinstance(clip_confidence, np.float32):
                    clip_confidence = float(clip_confidence)
                
                # Áp dụng adaptive threshold cho CLIP
                if clip_confidence >= adaptive_threshold:
                    # Kiểm tra nếu frame này cũng có trong kết quả keyword
                    frameid = frame_data.get('frameid')
                    if frameid in keyword_frame_ids:
                        # Tìm text detection có chứa keyword
                        detections = frame_data.get("text_detections", {}).get("detections", [])
                        keyword_without_accents = unidecode(keyword_to_use.lower())
                        
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
                        if best_text_match and max_text_conf >= keyword_threshold:
                            # Tìm object trong frame này
                            object_found = False
                            max_obj_conf = 0
                            best_label = ""
                            
                            # 1. Tìm trong object detections
                            object_detect_data = frame_data.get("object_detections", {})
                            object_detections = object_detect_data.get("detections", [])
                            
                            object_to_use_lower = object_to_use.lower()
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
                                    best_label = object_to_use
                            
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
                            
                            # 4. Text detections (thứ yếu)
                            for detection in detections:
                                detection_label = detection.get("label", "").lower()
                                if object_to_use_lower in detection_label or object_without_accents in unidecode(detection_label):
                                    conf = detection.get("confidence", 0) * 0.7  # Giảm confidence vì đây là text
                                    if conf > max_obj_conf:
                                        max_obj_conf = conf
                                        object_found = True
                                        best_label = detection_label
                            
                            # Chỉ giữ lại frame nếu object confidence đủ cao
                            if object_found and max_obj_conf >= obj_threshold:
                                # Tạo bản sao của frame_data để không ảnh hưởng đến dữ liệu gốc
                                frame_data_copy = frame_data.copy()
                                
                                # Đặt các giá trị confidence
                                frame_data_copy['clip_similarity'] = clip_confidence
                                frame_data_copy['text_confidence'] = max_text_conf
                                frame_data_copy['object_confidence'] = max_obj_conf
                                
                                # Cập nhật thông tin object được tìm thấy
                                frame_data_copy['object_label'] = best_label
                                
                                # Format event cho frontend
                                event = format_event_for_frontend(frame_data_copy)
                                
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