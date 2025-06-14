from flask import Flask, request, stream_with_context, send_file, url_for, jsonify, abort
from flask_cors import CORS
import os
from io import BytesIO
import numpy as np
import clip
import torch
from PIL import Image
import requests
import base64
import re
import json
from word_processing import VietnameseTextProcessor
from pathlib import Path
from segment_video import extract_frames_from_video
from JSON_sample_DOC import process_images_in_folder
from embedding import extract_and_save_embeddings_from_folder
import shutil
import time  
from unidecode import unidecode
import faiss 
import cv2
from query_strategies import (
    query_by_text_clip,
    query_by_text_with_adaptive_threshold,
    query_by_keyword,
    query_by_text_and_keyword,
    query_by_object,
    query_by_text_and_object,
    query_by_text_object_and_keyword
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

FRAMES_JSON = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\output_samples.json" 
EMBEDDINGS_FILE = "E:/Đồ án tôt nghiệp/source_code/Backend/embedding/image_embeddings.npy"
text_processor = VietnameseTextProcessor()

def load_frames_from_json(json_path):
    """Load danh sách tên file từ file JSON."""
    with open(json_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
    return [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]

def load_frames_mapping_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {Path(sample["filepath"]).name: sample["filepath"] for sample in data}

FRAMES_MAPPING = load_frames_mapping_from_json(FRAMES_JSON)
app = Flask(__name__)
# Thêm CORS support
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Utility functions 
def get_video_duration(video_path):
    """Lấy thời lượng video từ file."""
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

def get_video_resolution(video_path):
    """Lấy độ phân giải video."""
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

def format_event_for_frontend(frame_data):
    """Format dữ liệu event phù hợp với frontend React."""
    video_path = frame_data.get('video', '')
    video_id = video_path.split('/')[-1] if video_path else 'unknown'
    
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
        "videoId": f"video-{video_id}",
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

# Các hàm core được sử dụng bởi cả template cũ và API mới
def search_frames_by_keyword(keyword, top_k):
    """Tìm kiếm frame theo từ khóa text."""
    matching_frames = []
    with open(FRAMES_JSON, "r", encoding="utf-8") as f:
        frames_data = json.load(f)
        
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

def extract_query_confidence(frame_path, query):
    """Trích xuất độ tương đồng giữa một khung hình và query text."""
    try:
        # Tokenize và encode query text
        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Kiểm tra tồn tại file embeddings
        embeddings_path = "E:/Đồ án tôt nghiệp/source_code/Backend/embedding/image_embeddings.npy"
        if not os.path.exists(embeddings_path):
            print(f"Warning: Embeddings file not found: {embeddings_path}")
            return 0.0  # Trả về 0 nếu không tìm thấy file
            
        # Tải embeddings
        embeddings = np.load(embeddings_path)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        
        # Tải danh sách files
        all_files = load_frames_from_json(FRAMES_JSON)
        
        # Tìm index của frame trong danh sách
        try:
            index = all_files.index(frame_path)
        except ValueError:
            print(f"Warning: Frame {frame_path} not found in frames list")
            return 0.0  # Trả về 0 nếu không tìm thấy frame
            
        # Kiểm tra index có hợp lệ không
        if index >= embeddings.shape[0]:
            print(f"Warning: Index {index} out of bounds for embeddings shape {embeddings.shape}")
            return 0.0  # Trả về 0 nếu index không hợp lệ
        
        # Tính toán similarity
        similarity = np.dot(embeddings[index], text_features.T).flatten()[0]
        
        # Chuyển đổi numpy.float32 sang float thông thường
        if isinstance(similarity, np.float32) or isinstance(similarity, np.float64):
            similarity = float(similarity)
            
        return similarity
    except Exception as e:
        print(f"Error in extract_query_confidence: {e}")
        return 0.0  # Trả về 0 nếu có lỗi

def filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence):
    """Lọc frame dựa trên từ khóa và độ tin cậy."""
    if frame not in keyword_frames:
        return False
    keyword_without_accents = unidecode(keyword.lower())
    with open(FRAMES_JSON, "r", encoding="utf-8") as f:
        frames_data = json.load(f)
    
    frame_data = next((f for f in frames_data if f.get('frameid') == frame), {})
    detections = frame_data.get("text_detections", {}).get("detections", [])
    for detection in detections:
        detection_label = detection.get("label", "").lower()
        # Kiểm tra nếu label chứa keyword và có confidence >= min_confidence
        if keyword_without_accents in unidecode(detection_label) and detection.get("confidence", 0) >= min_confidence:
            return True
    return False

def search_top_frames(query, top_k):
    """Tìm kiếm top frames dựa trên query text sử dụng CLIP."""
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()

    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

    embeddings = np.load(EMBEDDINGS_FILE)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    # FAISS expects float32
    embeddings = embeddings.astype('float32')
    text_features = text_features.astype('float32')

    # Dùng chỉ mục FAISS cho cosine similarity (Inner Product)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(text_features, top_k)  # text_features shape: (1, dim)

    all_files = load_frames_from_json(FRAMES_JSON)
    top_indices = I[0]
    return [all_files[i] for i in top_indices]

def search_top_frames_by_image(image_features, top_k):
    """Tìm kiếm top frames dựa trên image features sử dụng CLIP."""
    embeddings = np.load(EMBEDDINGS_FILE)
    similarities = np.dot(embeddings, image_features.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    all_files = load_frames_from_json(FRAMES_JSON)
    return [all_files[i] for i in top_indices]

# Thêm 7 hàm xử lý truy vấn mới
def search_by_image(image_url, adaptive_threshold, top_k):
    """Tìm kiếm bằng hình ảnh."""
    try:
        if re.match(r"^data:image\/[a-zA-Z]+;base64,", image_url):
            base64_str = image_url.split(",", 1)[1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        else:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).cpu().numpy()
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)

        # Tìm kiếm các frame phù hợp
        top_frames = search_top_frames_by_image(image_features, top_k * 5)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data
        image_results = []
        for frame_name in top_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                event = format_event_for_frontend(frame_data)
                # Chuyển đổi tất cả giá trị numpy.float32 sang float thông thường
                if isinstance(event["confidence"], np.float32):
                    event["confidence"] = float(event["confidence"])
                if event["confidence"] >= adaptive_threshold:
                    image_results.append(event)
        
        return image_results[:top_k]
        
    except Exception as e:
        print(f"Error processing image search: {e}")
        return []

def search_semantic_with_clip(query, adaptive_threshold, top_k):
    """Tìm kiếm ngữ nghĩa với CLIP."""
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        query_frames = search_top_frames(processed_text, top_k * 5)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data
        semantic_results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                confidence = extract_query_confidence(frame_name, processed_text)
                # Chuyển đổi numpy.float32 sang float thông thường
                if isinstance(confidence, np.float32):
                    confidence = float(confidence)
                if confidence >= adaptive_threshold:
                    event = format_event_for_frontend(frame_data)
                    event["confidence"] = confidence  # Cập nhật confidence từ tìm kiếm
                    semantic_results.append(event)
        
        return semantic_results[:top_k]
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

def search_by_keyword(query, adaptive_threshold, top_k):
    """Tìm kiếm theo từ khóa."""
    try:
        keyword_frame_ids = search_frames_by_keyword(query, top_k * 3)
        keyword_results = []
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
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
                    if keyword_without_accents in unidecode(detection_label):
                        conf = detection.get("confidence", 0)
                        if conf > max_conf:
                            max_conf = conf
                            best_match = detection
                
                if best_match and max_conf >= adaptive_threshold:
                    event = format_event_for_frontend(frame_data)
                    event["confidence"] = float(max_conf) if isinstance(max_conf, np.float32) else max_conf
                    keyword_results.append(event)
        
        return keyword_results[:top_k]
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

def search_by_object(query, adaptive_threshold, top_k):
    """Tìm kiếm theo object."""
    try:
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
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

def search_fallback(query, top_k):
    """Tìm kiếm dự phòng khi không có kết quả."""
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        query_frames = search_top_frames(processed_text, top_k)
        
        # Đọc dữ liệu từ file JSON
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        fallback_results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                event = format_event_for_frontend(frame_data)
                fallback_results.append(event)
        
        return fallback_results
    except Exception as e:
        print(f"Error in fallback search: {e}")
        return []

def deduplicate_results(all_results):
    """Loại bỏ các kết quả trùng lặp."""
    unique_results = []
    unique_ids = set()
    for result in all_results:
        if result["id"] not in unique_ids:
            unique_ids.add(result["id"])
            unique_results.append(result)
    return unique_results

def sort_by_confidence(results):
    """Sắp xếp kết quả theo độ tin cậy."""
    return sorted(results, key=lambda x: x["confidence"], reverse=True)

@app.route('/api/videos', methods=['GET'])
def api_get_videos():
    """API lấy danh sách tất cả video."""
    videos = []
    try:
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        video_dict = {}
        for frame_data in data:
            video_path = frame_data.get('video')
            if video_path and video_path not in video_dict:
                video_name = os.path.basename(os.path.dirname(video_path))
                video_filename = os.path.basename(video_path)
                
                # Tạo ID duy nhất cho video
                video_id = f"video-{len(video_dict)+1}"
                
                video_dict[video_path] = {
                    "id": video_id,
                    "title": video_name or video_filename,
                    "thumbnail": frame_data.get('filepath'),
                    "duration": get_video_duration(video_path),
                    "uploadDate": time.strftime('%Y-%m-%d', time.gmtime(os.path.getctime(video_path))),
                    "size": f"{os.path.getsize(video_path) // (1024 * 1024)} MB",
                    "resolution": get_video_resolution(video_path),
                    "path": video_path
                }
        
        videos = list(video_dict.values())
        return jsonify(videos)
    except Exception as e:
        print(f"Error getting videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/<video_id>/events', methods=['GET'])
def api_get_video_events(video_id):
    """API lấy sự kiện của một video cụ thể."""
    try:
        # Trích xuất video_path từ video_id
        video_path = None
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Tìm tất cả frame thuộc video này
        events = []
        for frame_data in data:
            if video_id.startswith('video-'):
                # ID dạng video-1, video-2...
                video_num = int(video_id.split('-')[1])
                # Lấy danh sách các video paths
                all_video_paths = sorted(list(set([d.get('video') for d in data if d.get('video')])))
                if len(all_video_paths) >= video_num:
                    video_path = all_video_paths[video_num-1]
            
            if frame_data.get('video') == video_path:
                events.append(format_event_for_frontend(frame_data))
        
        # Chỉ lấy những events nổi bật
        # Có thể cải thiện thuật toán chọn events ở đây
        if len(events) > 20:
            step = len(events) // 20
            filtered_events = [events[i] for i in range(0, len(events), step)][:20]
        else:
            filtered_events = events
            
        return jsonify(filtered_events)
    except Exception as e:
        print(f"Error getting video events: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """API tìm kiếm sự kiện bằng text hoặc hình ảnh."""
    try:
        data = request.json
        search_type = data.get("search_type", "text")
        query = data.get("query", "")
        image_url = data.get("image_url")
        top_k = data.get("top_k", 10)
        adaptive_threshold = float(data.get("adaptive_threshold", 0.5))
        # Lấy thêm các threshold cụ thể nếu có
        text_confidence = float(data.get("text_confidence", adaptive_threshold))
        object_confidence = float(data.get("object_confidence", adaptive_threshold))
        # Các tham số khác
        use_keyword = data.get("use_keyword", False)
        use_object = data.get("use_object", False)
        search_method = data.get("search_method", "text")
        keyword = data.get("keyword", "")
        object_keyword = data.get("object", "")
        
        # Nhận thông tin filter từ frontend
        enable_text_keyword = data.get("enableTextKeyword", use_keyword)
        enable_object_keyword = data.get("enableObjectKeyword", use_object) 
        enable_clip_similarity = data.get("enableClipSimilarity", False)
        
        # Debug log
        print(f"Search request: type={search_type}, method={search_method}, query='{query}', object='{object_keyword}'")
        print(f"Thresholds: adaptive={adaptive_threshold}, text={text_confidence}, object={object_confidence}")
        print(f"Enabled filters: text={enable_text_keyword}, object={enable_object_keyword}, clip={enable_clip_similarity}")
        print(f"Keyword: '{keyword}'")
        
        results = []
        
        if search_type == "image" and image_url:
            # Tìm kiếm bằng hình ảnh - áp dụng ngưỡng adaptive_threshold trực tiếp
            image_results = search_by_image(image_url, adaptive_threshold, top_k)
            results = image_results
            print(f"Image search returned {len(image_results)} results")
        
        elif search_type == "text" and query:
            # Sử dụng phương pháp truy vấn dựa trên search_method và áp dụng các ngưỡng trực tiếp
            if search_method == "text_clip":
                # Truy vấn bằng văn bản thuần (CLIP Similarity)
                results = query_by_text_clip(query, top_k, search_top_frames, extract_query_confidence, format_event_for_frontend)
                print(f"Text CLIP search returned {len(results)} results")
            
            elif search_method == "text_adaptive":
                # Truy vấn bằng văn bản + Adaptive threshold - áp dụng ngưỡng trực tiếp
                results = query_by_text_with_adaptive_threshold(
                    query, adaptive_threshold, top_k, 
                    search_top_frames, extract_query_confidence, format_event_for_frontend
                )
                print(f"Text adaptive search returned {len(results)} results")
            
            elif search_method == "keyword_only":
                # Truy vấn bằng keyword - áp dụng text_confidence trực tiếp
                actual_query = keyword if keyword else query
                results = query_by_keyword(
                    actual_query, text_confidence, top_k, 
                    search_frames_by_keyword, format_event_for_frontend
                )
                print(f"Keyword only search returned {len(results)} results")
            
            elif search_method == "text_keyword":
                # Truy vấn bằng văn bản + keyword - áp dụng cả hai ngưỡng
                actual_keyword = keyword if keyword else query
                print(f"Text+keyword search for: text='{query}', keyword='{actual_keyword}'")
                
                results = query_by_text_and_keyword(
                    query, adaptive_threshold, top_k,
                    search_top_frames, extract_query_confidence, 
                    search_frames_by_keyword, format_event_for_frontend,
                    keyword=actual_keyword, text_confidence=text_confidence
                )
                print(f"Text+keyword search returned {len(results)} results")
            
            elif search_method == "object_only":
                # Truy vấn bằng Object - áp dụng object_confidence trực tiếp
                actual_query = object_keyword if object_keyword else query
                print(f"Searching for object with keyword: '{actual_query}'")
                
                results = query_by_object(actual_query, object_confidence, top_k, format_event_for_frontend)
                print(f"Object only search returned {len(results)} results")
                
            elif search_method == "text_object":
                # Truy vấn bằng văn bản + Object - áp dụng cả hai ngưỡng
                actual_object = object_keyword if object_keyword else query
                print(f"Text+object search for: text='{query}', object='{actual_object}'")
                
                results = query_by_text_and_object(
                    query, adaptive_threshold, top_k,
                    search_top_frames, extract_query_confidence, 
                    format_event_for_frontend,
                    object_keyword=actual_object, object_confidence=object_confidence
                )
                print(f"Text+object search returned {len(results)} results")
            
            elif search_method == "text_object_keyword":
                # Truy vấn bằng văn bản + Object + keyword - áp dụng cả ba ngưỡng
                actual_keyword = keyword if keyword else query
                actual_object = object_keyword if object_keyword else query
                
                results = query_by_text_object_and_keyword(
                    query, adaptive_threshold, top_k,
                    search_top_frames, extract_query_confidence,
                    search_frames_by_keyword, format_event_for_frontend,
                    keyword=actual_keyword, text_confidence=text_confidence,
                    object_keyword=actual_object, object_confidence=object_confidence
                )
                print(f"Text+object+keyword search returned {len(results)} results")
            
            # Fallback nếu không chọn phương pháp cụ thể
            else:
                results = query_by_text_with_adaptive_threshold(
                    query, adaptive_threshold, top_k, 
                    search_top_frames, extract_query_confidence, format_event_for_frontend
                )
        
        # Đảm bảo tất cả kết quả đều có các trường confidence đầy đủ
        for result in results:
            if "text_confidence" not in result:
                result["text_confidence"] = 0.0
            if "object_confidence" not in result:
                result["object_confidence"] = 0.0
            if "clip_similarity" not in result:
                result["clip_similarity"] = 0.0
        
        # Sắp xếp kết quả theo confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        return jsonify({"events": results[:top_k]})
            
    except Exception as e:
        print(f"Error searching: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    """API endpoint để upload video cho frontend React."""
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400
        
    video_name = os.path.splitext(video_file.filename)[0]
    frame_dir = os.path.join(BASE_DIR, video_name)
    frame_dir_process = r"{}".format(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    
    BaseVideo_dir = "E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\static\\video_frame"
    dir_save_video = os.path.join(BaseVideo_dir, video_name)
    os.makedirs(dir_save_video, exist_ok=True)
    path_save_video = os.path.join(dir_save_video, video_file.filename)
    video_file.save(path_save_video)
    
    try:
        # Xử lý video
        extract_frames_from_video(path_save_video, frame_dir, threshold=30.0)
        
        # Trích xuất embedding
        model_name = "ViT-B/32"
        output_file = "E:/Đồ án tôt nghiệp/source_code/Backend/embedding/image_embeddings.npy"
        extract_and_save_embeddings_from_folder(frame_dir, model_name, output_file)
        
        # Xử lý metadata
        json_output_path = FRAMES_JSON
        process_images_in_folder(frame_dir_process, json_output_path, path_save_video)
        
        # Tạo thông tin video response
        video_info = {
            "id": f"video-{int(time.time())}",
            "title": video_name,
            "path": path_save_video,
            "uploadDate": time.strftime('%Y-%m-%d'),
            "size": f"{os.path.getsize(path_save_video) // (1024 * 1024)} MB",
            "resolution": get_video_resolution(path_save_video),
            "duration": get_video_duration(path_save_video)
        }
        
        return jsonify({
            "status": "success",
            "message": "Video processed successfully",
            "video": video_info
        })
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/frame/<path:frame_path>')
def api_serve_frame(frame_path):
    """API để serve frame image."""
    try:
        # Kiểm tra nếu là đường dẫn đầy đủ
        if os.path.exists(frame_path):
            return send_file(frame_path, mimetype="image/jpeg")
        
        # Kiểm tra trong frames mapping
        if frame_path in FRAMES_MAPPING:
            return send_file(FRAMES_MAPPING[frame_path], mimetype="image/jpeg")
            
        # Thử tìm trong output_samples.json
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for frame_data in data:
            filepath = frame_data.get('filepath')
            if filepath and (os.path.basename(filepath) == frame_path or filepath == frame_path):
                return send_file(filepath, mimetype="image/jpeg")
        
        abort(404, description=f"Frame {frame_path} not found")
    except Exception as e:
        print(f"Error serving frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/<path:video_path>')
def api_serve_video(video_path):
    """API để serve video file."""
    try:
        # Kiểm tra nếu là đường dẫn đầy đủ
        if os.path.exists(video_path):
            return send_file(video_path, mimetype="video/mp4")
            
        # Thử tìm trong output_samples.json
        with open(FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for frame_data in data:
            filepath = frame_data.get('video')
            if filepath and (os.path.basename(filepath) == video_path or filepath == video_path):
                return send_file(filepath, mimetype="video/mp4")
        
        abort(404, description=f"Video {video_path} not found")
    except Exception as e:
        print(f"Error serving video: {e}")
        return jsonify({"error": str(e)}), 500

# Cần định nghĩa biến này vì được sử dụng trong hàm api_upload_video
BASE_DIR = "E:\\Đồ án chuyên ngành\\testing\\19_12_2024\\static\\processed_frames"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)