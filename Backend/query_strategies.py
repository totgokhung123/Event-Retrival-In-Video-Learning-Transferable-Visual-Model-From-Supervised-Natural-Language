import numpy as np
import json
from pathlib import Path
from unidecode import unidecode
from word_processing import VietnameseTextProcessor

# Đường dẫn file
FRAMES_JSON = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\output_samples.json"

def query_by_text_clip(query, top_k, search_top_frames, extract_query_confidence, format_event_for_frontend):
    """Truy vấn bằng văn bản thuần (CLIP Similarity)."""
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 2)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                confidence = extract_query_confidence(frame_name, processed_text)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                event = format_event_for_frontend(frame_data)
                event["confidence"] = confidence
                results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in text clip query: {e}")
        return []

def query_by_text_with_adaptive_threshold(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, format_event_for_frontend):
    """Truy vấn bằng văn bản + Adaptive threshold."""
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data với adaptive threshold
        results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                confidence = extract_query_confidence(frame_name, processed_text)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                # Áp dụng adaptive threshold
                if confidence >= adaptive_threshold:
                    event = format_event_for_frontend(frame_data)
                    event["confidence"] = confidence
                    results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in text with adaptive threshold query: {e}")
        return []

def query_by_keyword(query, adaptive_threshold, top_k, search_frames_by_keyword, format_event_for_frontend):
    """Truy vấn bằng keyword."""
    try:
        keyword_frame_ids = search_frames_by_keyword(query, top_k * 3)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
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
                    # Đảm bảo frame_data có chứa text_confidence
                    frame_data['text_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    
                    # Thêm các thông tin confidence khác nếu có
                    if 'clip_similarity' not in frame_data:
                        frame_data['clip_similarity'] = 0.0
                        
                    event = format_event_for_frontend(frame_data)
                    event["confidence"] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    results.append(event)
        
        # Sắp xếp theo confidence (giảm dần)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in keyword query: {e}")
        return []

def query_by_text_and_keyword(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                              search_frames_by_keyword, format_event_for_frontend, keyword=None, text_confidence=None):
    """Truy vấn bằng văn bản + keyword."""
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
        query_frames = search_top_frames(processed_text, top_k * 3)
        
        # Tìm frames chứa keyword
        keyword_frame_ids = search_frames_by_keyword(keyword_to_use, top_k * 3)
        print(f"Tìm thấy {len(keyword_frame_ids)} frames chứa từ khóa '{keyword_to_use}'")
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kết quả cuối cùng
        final_results = []
        
        # Xử lý các frame từ tìm kiếm CLIP
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if not frame_data:
                continue
                
            # Tính CLIP similarity
            clip_similarity = extract_query_confidence(frame_name, processed_text)
            if isinstance(clip_similarity, np.float32):
                clip_similarity = float(clip_similarity)
            
            # Kiểm tra nếu frame này cũng có trong kết quả keyword
            frame_id = frame_data.get('frameid')
            
            if frame_id in keyword_frame_ids:
                # Tìm text confidence từ keyword
                detections = frame_data.get("text_detections", {}).get("detections", [])
                keyword_without_accents = unidecode(keyword_to_use.lower())
                
                # Tìm detection có chứa keyword và có confidence cao nhất
                best_match = None
                max_text_conf = 0
                matching_label = ""
                
                for detection in detections:
                    detection_label = detection.get("label", "").lower()
                    if keyword_without_accents in unidecode(detection_label):
                        conf = detection.get("confidence", 0)
                        if conf > max_text_conf:
                            max_text_conf = conf
                            best_match = detection
                            matching_label = detection_label
                
                # Chỉ lấy kết quả thỏa mãn cả hai điều kiện
                if clip_similarity >= adaptive_threshold and max_text_conf >= keyword_threshold:
                    # Tạo bản sao của frame_data
                    frame_data_copy = frame_data.copy()
                    
                    # Đặt các giá trị confidence
                    frame_data_copy['clip_similarity'] = clip_similarity
                    frame_data_copy['text_confidence'] = max_text_conf
                    if 'object_confidence' not in frame_data_copy:
                        frame_data_copy['object_confidence'] = 0.0
                    
                    # Format event cho frontend
                    event = format_event_for_frontend(frame_data_copy)
                    
                    # Tăng điểm cho kết quả có trong cả hai phương pháp (lấy max và cộng thêm 0.1)
                    combined_conf = max(clip_similarity, max_text_conf) + 0.1
                    # Đảm bảo không vượt quá 1.0
                    event["confidence"] = min(combined_conf, 1.0)
                    event["clip_similarity"] = clip_similarity
                    event["text_confidence"] = max_text_conf
                    
                    print(f"Found combined match: frame={frame_id}, clip={clip_similarity:.2f}, text={max_text_conf:.2f}, label='{matching_label}'")
                    final_results.append(event)
        
        if not final_results:
            print(f"Không tìm thấy kết quả nào thỏa mãn cả hai điều kiện: CLIP >= {adaptive_threshold} và text confidence >= {keyword_threshold}")
        
        # Sắp xếp lại theo confidence
        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        return final_results[:top_k]
    except Exception as e:
        print(f"Error in text and keyword query: {e}")
        return []

def query_by_object(query, adaptive_threshold, top_k, format_event_for_frontend):
    """Truy vấn bằng Object."""
    try:
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        object_results = []
        query_lower = query.lower()
        query_without_accents = unidecode(query_lower)
        
        # Danh sách các từ đồng nghĩa thông dụng (mở rộng theo nhu cầu)
        synonyms = {
            "car": ["vehicle", "automobile", "xe hơi", "xe", "ô tô"],
            "person": ["human", "people", "man", "woman", "người", "người đàn ông", "người phụ nữ"],
            "airplane": ["aircraft", "plane", "máy bay"],
            "dog": ["canine", "puppy", "chó"],
            "cat": ["feline", "kitten", "mèo"],
            # Thêm các từ đồng nghĩa khác tùy theo nhu cầu
        }
        
        # Tạo danh sách từ khóa tìm kiếm, bao gồm cả từ đồng nghĩa
        search_terms = [query_lower, query_without_accents]
        for key, terms in synonyms.items():
            if query_lower in terms or query_without_accents in [unidecode(term.lower()) for term in terms]:
                search_terms.extend([key] + terms)
        if query_lower in synonyms:
            search_terms.extend(synonyms[query_lower])
        
        # Loại bỏ trùng lặp và chuẩn hóa các từ khóa
        search_terms = list(set([term.lower() for term in search_terms]))
        search_terms_without_accents = [unidecode(term) for term in search_terms]
        
        print(f"Searching for object terms: {search_terms}")
        
        for frame_data in data:
            max_conf = 0
            object_found = False
            best_label = ""
            matching_term = ""
            
            # 1. Tìm trong object detections (ưu tiên)
            object_detect_data = frame_data.get("object_detections", {})
            object_detections = object_detect_data.get("detections", [])
            
            for obj in object_detections:
                obj_label = obj.get("label", "").lower()
                obj_label_without_accents = unidecode(obj_label)
                
                # Kiểm tra nếu nhãn đối tượng khớp với bất kỳ từ khóa nào
                for term_idx, term in enumerate(search_terms):
                    if term in obj_label or search_terms_without_accents[term_idx] in obj_label_without_accents:
                        conf = obj.get("confidence", 0)
                        if conf > max_conf:
                            max_conf = conf
                            object_found = True
                            best_label = obj_label
                            matching_term = term
            
            # 2. Tìm trong metadata hoặc caption (nếu có)
            metadata = frame_data.get("metadata", {})
            caption = metadata.get("caption", "").lower() if metadata else ""
            caption_without_accents = unidecode(caption)
            
            for term_idx, term in enumerate(search_terms):
                if (term in caption or search_terms_without_accents[term_idx] in caption_without_accents):
                    # Nếu tìm thấy trong caption, gán confidence mặc định cao hơn
                    caption_conf = 0.65
                    if caption_conf > max_conf:
                        object_found = True
                        max_conf = caption_conf
                        best_label = term
                        matching_term = term
            
            # 3. Tìm trong tags hoặc labels (nếu có)
            tags = metadata.get("tags", []) if metadata else []
            for tag in tags:
                tag_lower = str(tag).lower()
                tag_without_accents = unidecode(tag_lower)
                
                for term_idx, term in enumerate(search_terms):
                    if term == tag_lower or search_terms_without_accents[term_idx] == tag_without_accents:
                        tag_conf = 0.75
                        if tag_conf > max_conf:
                            object_found = True
                            max_conf = tag_conf
                            best_label = tag
                            matching_term = term
                    
            # 4. Text detections (thứ yếu)
            text_detect_data = frame_data.get("text_detections", {})
            text_detections = text_detect_data.get("detections", [])
            
            for detection in text_detections:
                detection_label = detection.get("label", "").lower()
                detection_label_without_accents = unidecode(detection_label)
                
                # Tìm object name trong text đã nhận diện
                for term_idx, term in enumerate(search_terms):
                    if term in detection_label or search_terms_without_accents[term_idx] in detection_label_without_accents:
                        conf = detection.get("confidence", 0) * 0.7  # Giảm confidence vì đây là text
                        if conf > max_conf:
                            max_conf = conf
                            object_found = True
                            best_label = detection_label
                            matching_term = term
            
            # Giữ ngưỡng adaptive_threshold nhưng tối đa là 0.65
            actual_threshold = min(adaptive_threshold, 0.65)
            
            # Nếu có object và đủ confidence
            if object_found and max_conf >= actual_threshold:
                # Thêm object_confidence trực tiếp vào frame_data
                frame_data_copy = frame_data.copy()
                frame_data_copy['object_confidence'] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                
                # Đảm bảo các trường confidence khác có giá trị mặc định
                if 'text_confidence' not in frame_data_copy:
                    frame_data_copy['text_confidence'] = 0.0
                if 'clip_similarity' not in frame_data_copy:
                    frame_data_copy['clip_similarity'] = 0.0
                    
                # Đánh dấu rằng đây là kết quả từ object detection
                frame_data_copy['detection_type'] = 'object'
                
                # Cập nhật thông tin object được tìm thấy
                if not frame_data_copy.get('object_label'):
                    frame_data_copy['object_label'] = best_label
                
                # Thêm thông tin về từ khóa đã khớp để debug
                print(f"Found object match: term='{matching_term}', label='{best_label}', confidence={max_conf}")
                
                event = format_event_for_frontend(frame_data_copy)
                object_results.append(event)
        
        # Sắp xếp kết quả theo confidence
        object_results.sort(key=lambda x: x["confidence"], reverse=True)
        return object_results[:top_k]
    except Exception as e:
        print(f"Error in object search: {e}")
        return []

def query_by_text_and_object(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                            format_event_for_frontend, object_keyword=None, object_confidence=None):
    """Truy vấn bằng văn bản + Object."""
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
        query_frames = search_top_frames(processed_text, top_k * 3)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Danh sách các từ đồng nghĩa cho object search
        synonyms = {
            "car": ["vehicle", "automobile", "xe hơi", "xe", "ô tô"],
            "person": ["human", "people", "man", "woman", "người", "người đàn ông", "người phụ nữ"],
            "airplane": ["aircraft", "plane", "máy bay"],
            "dog": ["canine", "puppy", "chó"],
            "cat": ["feline", "kitten", "mèo"],
        }
        
        # Tạo danh sách từ khóa tìm kiếm object, bao gồm cả từ đồng nghĩa
        obj_search_terms = [object_to_use.lower(), unidecode(object_to_use.lower())]
        for key, terms in synonyms.items():
            if object_to_use.lower() in terms or unidecode(object_to_use.lower()) in [unidecode(term.lower()) for term in terms]:
                obj_search_terms.extend([key] + terms)
        if object_to_use.lower() in synonyms:
            obj_search_terms.extend(synonyms[object_to_use.lower()])
        
        # Loại bỏ trùng lặp và chuẩn hóa các từ khóa
        obj_search_terms = list(set([term.lower() for term in obj_search_terms]))
        obj_search_terms_without_accents = [unidecode(term) for term in obj_search_terms]
        
        # Kết quả cuối cùng
        final_results = []
        
        # Xử lý các frame từ tìm kiếm CLIP
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if not frame_data:
                continue
                
            # Tính CLIP similarity
            clip_similarity = extract_query_confidence(frame_name, processed_text)
            if isinstance(clip_similarity, np.float32):
                clip_similarity = float(clip_similarity)
            
            # Nếu CLIP similarity không đạt ngưỡng, bỏ qua
            if clip_similarity < adaptive_threshold:
                continue
            
            # Tìm object trong frame
            max_obj_conf = 0
            object_found = False
            best_obj_label = ""
            
            # 1. Tìm trong object detections (ưu tiên)
            object_detect_data = frame_data.get("object_detections", {})
            object_detections = object_detect_data.get("detections", [])
            
            for obj in object_detections:
                obj_label = obj.get("label", "").lower()
                obj_label_without_accents = unidecode(obj_label)
                
                # Kiểm tra nếu nhãn đối tượng khớp với bất kỳ từ khóa nào
                for term_idx, term in enumerate(obj_search_terms):
                    if term in obj_label or obj_search_terms_without_accents[term_idx] in obj_label_without_accents:
                        conf = obj.get("confidence", 0)
                        if conf > max_obj_conf:
                            max_obj_conf = conf
                            object_found = True
                            best_obj_label = obj_label
            
            # 2. Tìm trong metadata hoặc caption (nếu có)
            metadata = frame_data.get("metadata", {})
            caption = metadata.get("caption", "").lower() if metadata else ""
            caption_without_accents = unidecode(caption)
            
            for term_idx, term in enumerate(obj_search_terms):
                if (term in caption or obj_search_terms_without_accents[term_idx] in caption_without_accents):
                    caption_conf = 0.65
                    if caption_conf > max_obj_conf:
                        object_found = True
                        max_obj_conf = caption_conf
                        best_obj_label = term
            
            # 3. Tìm trong text detections (thứ yếu)
            text_detections = frame_data.get("text_detections", {}).get("detections", [])
            for detection in text_detections:
                detection_label = detection.get("label", "").lower()
                detection_label_without_accents = unidecode(detection_label)
                
                for term_idx, term in enumerate(obj_search_terms):
                    if term in detection_label or obj_search_terms_without_accents[term_idx] in detection_label_without_accents:
                        conf = detection.get("confidence", 0) * 0.7  # Giảm confidence vì đây là text
                        if conf > max_obj_conf:
                            max_obj_conf = conf
                            object_found = True
                            best_obj_label = detection_label
            
            # Nếu không tìm thấy object match với đủ confidence, bỏ qua
            if not object_found or max_obj_conf < obj_threshold:
                continue
            
            # Chỉ lấy kết quả thỏa mãn cả hai điều kiện
            # Tạo bản sao của frame_data
            frame_data_copy = frame_data.copy()
            
            # Đặt các giá trị confidence
            frame_data_copy['clip_similarity'] = clip_similarity
            frame_data_copy['object_confidence'] = max_obj_conf
            if 'text_confidence' not in frame_data_copy:
                frame_data_copy['text_confidence'] = 0.0
            
            # Format event cho frontend
            event = format_event_for_frontend(frame_data_copy)
            
            # Tăng điểm cho kết quả có trong cả hai phương pháp (lấy max và cộng thêm 0.1)
            combined_conf = max(clip_similarity, max_obj_conf) + 0.1
            # Đảm bảo không vượt quá 1.0
            event["confidence"] = min(combined_conf, 1.0)
            event["clip_similarity"] = clip_similarity
            event["object_confidence"] = max_obj_conf
            
            print(f"Found combined match: frame={frame_idx}, clip={clip_similarity:.2f}, object={max_obj_conf:.2f} ('{best_obj_label}')")
            final_results.append(event)
        
        if not final_results:
            print(f"Không tìm thấy kết quả nào thỏa mãn cả hai điều kiện: CLIP >= {adaptive_threshold} và object confidence >= {obj_threshold}")
        
        # Sắp xếp lại theo confidence
        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        return final_results[:top_k]
    except Exception as e:
        print(f"Error in text and object query: {e}")
        return []

def query_by_text_object_and_keyword(query, adaptive_threshold, top_k, search_top_frames, extract_query_confidence, 
                                    search_frames_by_keyword, format_event_for_frontend, 
                                    keyword=None, text_confidence=None,
                                    object_keyword=None, object_confidence=None):
    """Truy vấn bằng văn bản + Object + Keyword."""
    # Sử dụng các tham số truyền vào nếu có, nếu không thì dùng query
    keyword_to_use = keyword if keyword else query
    object_to_use = object_keyword if object_keyword else query
    
    # Sử dụng thresholds truyền vào nếu có, nếu không thì dùng adaptive_threshold
    keyword_threshold = text_confidence if text_confidence is not None else adaptive_threshold
    obj_threshold = object_confidence if object_confidence is not None else adaptive_threshold
    
    print(f"Tìm kiếm kết hợp 3 điều kiện: query='{query}' với CLIP threshold={adaptive_threshold}, " +
          f"keyword='{keyword_to_use}' với text threshold={keyword_threshold}, " +
          f"object='{object_to_use}' với object threshold={obj_threshold}")
    
    try:
        # Xử lý query text
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        
        # Tìm top frame với CLIP
        query_frames = search_top_frames(processed_text, top_k * 3)
        
        # Tìm frames chứa keyword
        keyword_frame_ids = search_frames_by_keyword(keyword_to_use, top_k * 3)
        print(f"Tìm thấy {len(keyword_frame_ids)} frames chứa từ khóa '{keyword_to_use}'")
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kết quả cuối cùng
        final_results = []
        
        # Danh sách các từ đồng nghĩa cho object search
        synonyms = {
            "car": ["vehicle", "automobile", "xe hơi", "xe", "ô tô"],
            "person": ["human", "people", "man", "woman", "người", "người đàn ông", "người phụ nữ"],
            "airplane": ["aircraft", "plane", "máy bay"],
            "dog": ["canine", "puppy", "chó"],
            "cat": ["feline", "kitten", "mèo"],
        }
        
        # Tạo danh sách từ khóa tìm kiếm object, bao gồm cả từ đồng nghĩa
        obj_search_terms = [object_to_use.lower(), unidecode(object_to_use.lower())]
        for key, terms in synonyms.items():
            if object_to_use.lower() in terms or unidecode(object_to_use.lower()) in [unidecode(term.lower()) for term in terms]:
                obj_search_terms.extend([key] + terms)
        if object_to_use.lower() in synonyms:
            obj_search_terms.extend(synonyms[object_to_use.lower()])
        
        # Loại bỏ trùng lặp và chuẩn hóa các từ khóa
        obj_search_terms = list(set([term.lower() for term in obj_search_terms]))
        obj_search_terms_without_accents = [unidecode(term) for term in obj_search_terms]
        
        # Xử lý các frame từ tìm kiếm CLIP
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if not frame_data:
                continue
                
            # Tính CLIP similarity
            clip_similarity = extract_query_confidence(frame_name, processed_text)
            if isinstance(clip_similarity, np.float32):
                clip_similarity = float(clip_similarity)
            
            # Kiểm tra nếu frame này có trong kết quả keyword
            frame_id = frame_data.get('frameid')
            if frame_id not in keyword_frame_ids:
                continue
                
            # Tìm text confidence từ keyword
            detections = frame_data.get("text_detections", {}).get("detections", [])
            keyword_without_accents = unidecode(keyword_to_use.lower())
            
            # Tìm detection có chứa keyword và có confidence cao nhất
            best_match = None
            max_text_conf = 0
            matching_label = ""
            
            for detection in detections:
                detection_label = detection.get("label", "").lower()
                if keyword_without_accents in unidecode(detection_label):
                    conf = detection.get("confidence", 0)
                    if conf > max_text_conf:
                        max_text_conf = conf
                        best_match = detection
                        matching_label = detection_label
            
            # Nếu không tìm thấy text match với đủ confidence, bỏ qua
            if max_text_conf < keyword_threshold:
                continue
            
            # Tìm object trong frame
            max_obj_conf = 0
            object_found = False
            best_obj_label = ""
            
            # 1. Tìm trong object detections (ưu tiên)
            object_detect_data = frame_data.get("object_detections", {})
            object_detections = object_detect_data.get("detections", [])
            
            for obj in object_detections:
                obj_label = obj.get("label", "").lower()
                obj_label_without_accents = unidecode(obj_label)
                
                # Kiểm tra nếu nhãn đối tượng khớp với bất kỳ từ khóa nào
                for term_idx, term in enumerate(obj_search_terms):
                    if term in obj_label or obj_search_terms_without_accents[term_idx] in obj_label_without_accents:
                        conf = obj.get("confidence", 0)
                        if conf > max_obj_conf:
                            max_obj_conf = conf
                            object_found = True
                            best_obj_label = obj_label
            
            # 2. Tìm trong metadata hoặc caption (nếu có)
            metadata = frame_data.get("metadata", {})
            caption = metadata.get("caption", "").lower() if metadata else ""
            caption_without_accents = unidecode(caption)
            
            for term_idx, term in enumerate(obj_search_terms):
                if (term in caption or obj_search_terms_without_accents[term_idx] in caption_without_accents):
                    caption_conf = 0.65
                    if caption_conf > max_obj_conf:
                        object_found = True
                        max_obj_conf = caption_conf
                        best_obj_label = term
            
            # 3. Tìm trong text detections (thứ yếu)
            for detection in detections:
                detection_label = detection.get("label", "").lower()
                detection_label_without_accents = unidecode(detection_label)
                
                for term_idx, term in enumerate(obj_search_terms):
                    if term in detection_label or obj_search_terms_without_accents[term_idx] in detection_label_without_accents:
                        conf = detection.get("confidence", 0) * 0.7  # Giảm confidence vì đây là text
                        if conf > max_obj_conf:
                            max_obj_conf = conf
                            object_found = True
                            best_obj_label = detection_label
            
            # Nếu không tìm thấy object match với đủ confidence, bỏ qua
            if not object_found or max_obj_conf < obj_threshold:
                continue
            
            # Chỉ lấy kết quả thỏa mãn cả ba điều kiện
            if clip_similarity >= adaptive_threshold and max_text_conf >= keyword_threshold and max_obj_conf >= obj_threshold:
                # Tạo bản sao của frame_data
                frame_data_copy = frame_data.copy()
                
                # Đặt các giá trị confidence
                frame_data_copy['clip_similarity'] = clip_similarity
                frame_data_copy['text_confidence'] = max_text_conf
                frame_data_copy['object_confidence'] = max_obj_conf
                
                # Format event cho frontend
                event = format_event_for_frontend(frame_data_copy)
                
                # Tăng điểm cho kết quả có trong cả ba phương pháp (lấy max và cộng thêm 0.15)
                combined_conf = max(clip_similarity, max_text_conf, max_obj_conf) + 0.15
                # Đảm bảo không vượt quá 1.0
                event["confidence"] = min(combined_conf, 1.0)
                event["clip_similarity"] = clip_similarity
                event["text_confidence"] = max_text_conf
                event["object_confidence"] = max_obj_conf
                
                print(f"Found triple match: frame={frame_id}, clip={clip_similarity:.2f}, " +
                      f"text={max_text_conf:.2f} ('{matching_label}'), " +
                      f"object={max_obj_conf:.2f} ('{best_obj_label}')")
                      
                final_results.append(event)
        
        if not final_results:
            print(f"Không tìm thấy kết quả nào thỏa mãn cả ba điều kiện")
        
        # Sắp xếp lại theo confidence
        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        return final_results[:top_k]
    except Exception as e:
        print(f"Error in text, object and keyword query: {e}")
        return []