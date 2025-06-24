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
from werkzeug.utils import secure_filename

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Base directories
BASE_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\static\\processed_frames"
METADATA_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\metadata"
EMBEDDING_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\embedding"

# Create necessary directories if they don't exist
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Dictionary to track metadata and embedding files for each video
video_data_mapping = {}

# Create a directory for voice uploads
VOICE_DIR = os.path.join(os.path.dirname(__file__), "static", "voice")
os.makedirs(VOICE_DIR, exist_ok=True)

def load_video_data_mapping():
    """Load mapping between videos and their metadata/embedding files."""
    mapping_file = os.path.join(METADATA_DIR, "video_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_video_data_mapping():
    """Save the current video data mapping to a file."""
    mapping_file = os.path.join(METADATA_DIR, "video_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(video_data_mapping, f, ensure_ascii=False, indent=4)

# Load existing mapping at startup
video_data_mapping = load_video_data_mapping()

def get_default_metadata_path():
    """Get the default metadata path for backward compatibility."""
    # Check if there's any video in the mapping
    if video_data_mapping:
        # Return the metadata file of the first video
        first_video = list(video_data_mapping.keys())[0]
        return video_data_mapping[first_video]["metadata_file"]
    # Fallback to a default path
    return os.path.join(METADATA_DIR, "output_samples.json")

def get_default_embeddings_path():
    """Get the default embeddings path for backward compatibility."""
    # Check if there's any video in the mapping
    if video_data_mapping:
        # Return the embeddings file of the first video
        first_video = list(video_data_mapping.keys())[0]
        return video_data_mapping[first_video]["embeddings_file"]
    # Fallback to a default path
    return os.path.join(EMBEDDING_DIR, "image_embeddings.npy")

def get_video_metadata_path(video_name=None):
    """
    Get the metadata file path for a specific video or the default one.
    
    Args:
        video_name: Name of the video
    
    Returns:
        Path to the metadata file
    """
    if video_name and video_name in video_data_mapping:
        # Chuẩn hóa đường dẫn để đảm bảo tương thích
        path = video_data_mapping[video_name]["metadata_file"]
        # Đảm bảo đường dẫn sử dụng dấu gạch chéo phù hợp với hệ thống
        path = os.path.normpath(path)
        print(f"Using metadata file lấy dữ liệu: {path}")
        return path
    return get_default_metadata_path()

def get_video_embeddings_path(video_name=None):
    """
    Get the embeddings file path for a specific video or the default one.
    
    Args:
        video_name: Name of the video
    
    Returns:
        Path to the embeddings file
    """
    if video_name and video_name in video_data_mapping:
        # Chuẩn hóa đường dẫn để đảm bảo tương thích
        path = video_data_mapping[video_name]["embeddings_file"]
        # Đảm bảo đường dẫn sử dụng dấu gạch chéo phù hợp với hệ thống
        path = os.path.normpath(path)
        print(f"Using embeddings file lấy dữ liệu: {path}")
        return path
    return get_default_embeddings_path()

def load_frames_from_json(json_path, video_name=None):
    """
    Load danh sách tên file từ file JSON.
    Args:
        json_path: Path to JSON file or video name
        video_name: If provided, will load frames for this specific video
    """
    # If video_name is provided, use the specific metadata file for that video
    if video_name:
        json_path = get_video_metadata_path(video_name)
    
    # Chuẩn hóa đường dẫn để đảm bảo tương thích
    json_path = os.path.normpath(json_path)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            samples = json.load(file)
        return [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]
    except Exception as e:
        print(f"Error loading frames from JSON: {e}, path: {json_path}")
        return []

def load_frames_mapping_from_json(json_path, video_name=None):
    """
    Load mapping from frame file names to full paths.
    
    Args:
        json_path: Path to JSON file or video name
        video_name: If provided, will load frames for this specific video
    """
    # If video_name is provided, use the specific metadata file for that video
    if video_name:
        json_path = get_video_metadata_path(video_name)
    
    # Chuẩn hóa đường dẫn để đảm bảo tương thích
    json_path = os.path.normpath(json_path)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {Path(sample["filepath"]).name: sample["filepath"] for sample in data}
    except Exception as e:
        print(f"Error loading frames mapping from JSON: {e}, path: {json_path}")
        return {}

FRAMES_MAPPING = load_frames_mapping_from_json(get_default_metadata_path())
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

# Các hàm core được sử dụng bởi cả template cũ và API mới
def search_frames_by_keyword(keyword, top_k, video_name=None):
    """
    Tìm kiếm frame theo từ khóa text.
    
    Args:
        keyword: Từ khóa cần tìm
        top_k: Số kết quả trả về tối đa
        video_name: Nếu có, chỉ tìm kiếm trong video này
    """
    matching_frames = []
    
    # Determine which JSON file to use
    json_path = get_video_metadata_path(video_name)
        
    with open(json_path, "r", encoding="utf-8") as f:
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

def extract_query_confidence(frame_path, query, video_name=None):
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
        
        # Kiểm tra xem đã có text features trong cache chưa
        if not hasattr(extract_query_confidence, 'text_features_cache'):
            extract_query_confidence.text_features_cache = {}
            
        if not hasattr(extract_query_confidence, 'embeddings_cache'):
            extract_query_confidence.embeddings_cache = {}
            
        if not hasattr(extract_query_confidence, 'frames_list_cache'):
            extract_query_confidence.frames_list_cache = {}
            
        # Thêm cache cho đường dẫn file
        if not hasattr(extract_query_confidence, 'path_cache'):
            extract_query_confidence.path_cache = {}
            
        # Tạo key cho cache đường dẫn
        path_cache_key = f"paths_{video_name or 'default'}"
        
        # Nếu chưa có text features cho query này, tính và cache lại
        if cache_key not in extract_query_confidence.text_features_cache:
            text_input = clip.tokenize([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_input).cpu().numpy()
            text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
            extract_query_confidence.text_features_cache[cache_key] = text_features
        else:
            text_features = extract_query_confidence.text_features_cache[cache_key]
        
        # Kiểm tra và sử dụng cache đường dẫn
        if path_cache_key not in extract_query_confidence.path_cache:
            print("Dong nay lay embeddings cho extract_query_confidence\n")
            # Lấy đường dẫn embeddings và metadata một lần và lưu vào cache
            embeddings_path = get_video_embeddings_path(video_name)
            json_path = get_video_metadata_path(video_name)
            extract_query_confidence.path_cache[path_cache_key] = {
                'embeddings_path': embeddings_path,
                'json_path': json_path
            }
        else:
            # Sử dụng đường dẫn từ cache
            embeddings_path = extract_query_confidence.path_cache[path_cache_key]['embeddings_path']
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        
        # Kiểm tra nếu đã có embeddings trong cache
        if embeddings_path not in extract_query_confidence.embeddings_cache:
            # Check if embeddings file exists
            if not os.path.exists(embeddings_path):
                print(f"Warning: Embeddings file not found: {embeddings_path}")
                return 0.0
            
            print("Dong nay tính toán embeddings\n")    
            # Load embeddings
            embeddings = np.load(embeddings_path)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            extract_query_confidence.embeddings_cache[embeddings_path] = embeddings
        else:
            embeddings = extract_query_confidence.embeddings_cache[embeddings_path]
        
        print("Dong nay lay frames cho extract_query_confidence\n")
        # Lấy danh sách frames từ cache hoặc load mới
        if json_path not in extract_query_confidence.frames_list_cache:
            # Load frame list
            all_files = load_frames_from_json(json_path, video_name)
            extract_query_confidence.frames_list_cache[json_path] = all_files
        else:
            all_files = extract_query_confidence.frames_list_cache[json_path]
        
        # Find frame index
        try:
            index = all_files.index(frame_path)
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

def filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence, video_name=None):
    """
    Lọc frame dựa trên từ khóa và độ tin cậy.
    
    Args:
        frame: Frame ID to check
        keyword: Keyword to search for
        keyword_frames: List of frames containing the keyword
        min_confidence: Minimum confidence threshold
        video_name: If provided, search only within this video's frames
    """
    if frame not in keyword_frames:
        return False
        
    keyword_without_accents = unidecode(keyword.lower())
    
    # Determine which JSON file to use
    json_path = get_video_metadata_path(video_name)
    
    with open(json_path, "r", encoding="utf-8") as f:
        frames_data = json.load(f)
    
    frame_data = next((f for f in frames_data if f.get('frameid') == frame), {})
    detections = frame_data.get("text_detections", {}).get("detections", [])
    
    for detection in detections:
        detection_label = detection.get("label", "").lower()
        # Kiểm tra nếu label chứa keyword và có confidence >= min_confidence
        if keyword_without_accents in unidecode(detection_label) and detection.get("confidence", 0) >= min_confidence:
            return True
    
    return False

def search_top_frames(query, top_k, video_name=None):
    """
    Tìm kiếm top frames dựa trên query text sử dụng CLIP.
    
    Args:
        query: Query text
        top_k: Number of top results to return
        video_name: If provided, search only within this video's frames
    """
    try:
        # Tạo một key duy nhất cho query và video_name
        cache_key = f"{query}_{video_name or 'default'}"
        
        # Khởi tạo cache cho kết quả tìm kiếm nếu chưa có
        if not hasattr(search_top_frames, 'results_cache'):
            search_top_frames.results_cache = {}
            
        # Nếu đã có kết quả trong cache, trả về ngay lập tức
        if cache_key in search_top_frames.results_cache:
            return search_top_frames.results_cache[cache_key][:top_k]
            
        # Nếu không có cache, thực hiện tìm kiếm
        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input).cpu().numpy()

        text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        print("Dong nay lấy dữ liệu cả 2 trong search_top_frames\n")
        
        # Kiểm tra xem có cache đường dẫn từ extract_query_confidence không
        path_cache_key = f"paths_{video_name or 'default'}"
        if (hasattr(extract_query_confidence, 'path_cache') and 
            path_cache_key in extract_query_confidence.path_cache):
            # Sử dụng đường dẫn từ cache
            embeddings_file = extract_query_confidence.path_cache[path_cache_key]['embeddings_path']
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        else:
            # Determine which embeddings and frames to use based on video_name
            embeddings_file = get_video_embeddings_path(video_name)
            json_path = get_video_metadata_path(video_name)
            
            # Lưu vào cache nếu extract_query_confidence có cache
            if hasattr(extract_query_confidence, 'path_cache'):
                extract_query_confidence.path_cache[path_cache_key] = {
                    'embeddings_path': embeddings_file,
                    'json_path': json_path
                }
        
        # Load embeddings
        try:
            # Sử dụng embedding cache từ extract_query_confidence nếu có
            if hasattr(extract_query_confidence, 'embeddings_cache') and embeddings_file in extract_query_confidence.embeddings_cache:
                embeddings = extract_query_confidence.embeddings_cache[embeddings_file]
            else:
                embeddings = np.load(embeddings_file)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
                # Lưu vào cache cho lần sau
                if not hasattr(extract_query_confidence, 'embeddings_cache'):
                    extract_query_confidence.embeddings_cache = {}
                extract_query_confidence.embeddings_cache[embeddings_file] = embeddings

            # FAISS expects float32
            embeddings = embeddings.astype('float32')
            text_features = text_features.astype('float32')

            # Dùng chỉ mục FAISS cho cosine similarity (Inner Product)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            D, I = index.search(text_features, top_k)  # text_features shape: (1, dim)

            # Sử dụng frame list cache từ extract_query_confidence nếu có
            if hasattr(extract_query_confidence, 'frames_list_cache') and json_path in extract_query_confidence.frames_list_cache:
                all_files = extract_query_confidence.frames_list_cache[json_path]
            else:
                all_files = load_frames_from_json(json_path, video_name)
                # Lưu vào cache cho lần sau
                if not hasattr(extract_query_confidence, 'frames_list_cache'):
                    extract_query_confidence.frames_list_cache = {}
                extract_query_confidence.frames_list_cache[json_path] = all_files
            
            top_indices = I[0]
            results = [all_files[i] for i in top_indices if i < len(all_files)]
            
            # Lưu kết quả vào cache
            search_top_frames.results_cache[cache_key] = results
            
            return results
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return []
    except Exception as e:
        print(f"Error in search_top_frames: {e}")
        return []

def search_top_frames_by_image(image_features, top_k, video_name=None):
    """
    Tìm kiếm top frames dựa trên image features sử dụng CLIP.
    
    Args:
        image_features: Image feature vector
        top_k: Number of top results to return
        video_name: If provided, search only within this video's frames
    """
    try:
        # Determine which embeddings and frames to use based on video_name
        embeddings_file = get_video_embeddings_path(video_name)
        json_path = get_video_metadata_path(video_name)
        
        print(f"Loading embeddings from: {embeddings_file}")
        print(f"Loading JSON from: {json_path}")
            
        # Load embeddings
        try:
            embeddings = np.load(embeddings_file)
            similarities = np.dot(embeddings, image_features.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            all_files = load_frames_from_json(json_path, video_name)
            return [all_files[i] for i in top_indices if i < len(all_files)]
        except Exception as e:
            print(f"Error loading embeddings for image search: {e}")
            return []
    except Exception as e:
        print(f"Error in search_top_frames_by_image: {e}")
        return []

# Thêm 7 hàm xử lý truy vấn mới
def search_by_image(image_url, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm bằng hình ảnh.
    
    Args:
        image_url: URL của hình ảnh để tìm kiếm
        adaptive_threshold: Ngưỡng độ tin cậy
        top_k: Số kết quả trả về tối đa
        video_name: Nếu có, chỉ tìm kiếm trong video này
    """
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
        top_frames = search_top_frames_by_image(image_features, top_k * 5, video_name)
        
        # Kiểm tra xem có cache đường dẫn từ extract_query_confidence không
        path_cache_key = f"paths_{video_name or 'default'}"
        if (hasattr(extract_query_confidence, 'path_cache') and 
            path_cache_key in extract_query_confidence.path_cache):
            # Sử dụng đường dẫn từ cache
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        else:
            # Determine which JSON file to use
            json_path = get_video_metadata_path(video_name)
            
            # Lưu vào cache nếu extract_query_confidence có cache
            if hasattr(extract_query_confidence, 'path_cache'):
                if 'embeddings_path' not in extract_query_confidence.path_cache.get(path_cache_key, {}):
                    # Nếu chưa có embeddings_path, cần lấy thêm
                    embeddings_path = get_video_embeddings_path(video_name)
                    extract_query_confidence.path_cache[path_cache_key] = {
                        'embeddings_path': embeddings_path,
                        'json_path': json_path
                    }
                else:
                    # Nếu đã có embeddings_path, chỉ cập nhật json_path
                    extract_query_confidence.path_cache[path_cache_key]['json_path'] = json_path
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
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

def search_semantic_with_clip(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm ngữ nghĩa với CLIP.
    
    Args:
        query: Query text
        adaptive_threshold: Ngưỡng độ tin cậy
        top_k: Số kết quả trả về tối đa
        video_name: Nếu có, chỉ tìm kiếm trong video này
    """
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        
        query_frames = search_top_frames(processed_text, top_k * 5, video_name)
        
        print("Dong nay lấy dữ liệu metadata trong search_semantic_with_clip\n")
        
        # Kiểm tra xem có cache đường dẫn từ extract_query_confidence không
        path_cache_key = f"paths_{video_name or 'default'}"
        if (hasattr(extract_query_confidence, 'path_cache') and 
            path_cache_key in extract_query_confidence.path_cache):
            # Sử dụng đường dẫn từ cache
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        else:
            # Determine which JSON file to use
            json_path = get_video_metadata_path(video_name)
            
            # Lưu vào cache nếu extract_query_confidence có cache
            if hasattr(extract_query_confidence, 'path_cache'):
                if 'embeddings_path' not in extract_query_confidence.path_cache.get(path_cache_key, {}):
                    # Nếu chưa có embeddings_path, cần lấy thêm
                    embeddings_path = get_video_embeddings_path(video_name)
                    extract_query_confidence.path_cache[path_cache_key] = {
                        'embeddings_path': embeddings_path,
                        'json_path': json_path
                    }
                else:
                    # Nếu đã có embeddings_path, chỉ cập nhật json_path
                    extract_query_confidence.path_cache[path_cache_key]['json_path'] = json_path
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc frame data
        semantic_results = []
        for frame_name in query_frames:
            frame_idx = int(Path(frame_name).stem)
            frame_data = next((item for item in data if item.get('frameidx') == frame_idx), None)
            if frame_data:
                confidence = extract_query_confidence(frame_name, processed_text, video_name)
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

def search_by_keyword(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm theo từ khóa.
    
    Args:
        query: Từ khóa cần tìm
        adaptive_threshold: Ngưỡng độ tin cậy
        top_k: Số kết quả trả về tối đa
        video_name: Nếu có, chỉ tìm kiếm trong video này
    """
    try:
        keyword_frame_ids = search_frames_by_keyword(query, top_k * 3, video_name)
        keyword_results = []
        
        # Kiểm tra xem có cache đường dẫn từ extract_query_confidence không
        path_cache_key = f"paths_{video_name or 'default'}"
        if (hasattr(extract_query_confidence, 'path_cache') and 
            path_cache_key in extract_query_confidence.path_cache):
            # Sử dụng đường dẫn từ cache
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        else:
            # Determine which JSON file to use
            json_path = get_video_metadata_path(video_name)
            
            # Lưu vào cache nếu extract_query_confidence có cache
            if hasattr(extract_query_confidence, 'path_cache'):
                if 'embeddings_path' not in extract_query_confidence.path_cache.get(path_cache_key, {}):
                    # Nếu chưa có embeddings_path, cần lấy thêm
                    embeddings_path = get_video_embeddings_path(video_name)
                    extract_query_confidence.path_cache[path_cache_key] = {
                        'embeddings_path': embeddings_path,
                        'json_path': json_path
                    }
                else:
                    # Nếu đã có embeddings_path, chỉ cập nhật json_path
                    extract_query_confidence.path_cache[path_cache_key]['json_path'] = json_path
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
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

def search_by_object(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm theo object.
    
    Args:
        query: Từ khóa object cần tìm
        adaptive_threshold: Ngưỡng độ tin cậy
        top_k: Số kết quả trả về tối đa
        video_name: Nếu có, chỉ tìm kiếm trong video này
    """
    try:
        # Kiểm tra xem có cache đường dẫn từ extract_query_confidence không
        path_cache_key = f"paths_{video_name or 'default'}"
        if (hasattr(extract_query_confidence, 'path_cache') and 
            path_cache_key in extract_query_confidence.path_cache):
            # Sử dụng đường dẫn từ cache
            json_path = extract_query_confidence.path_cache[path_cache_key]['json_path']
        else:
            # Determine which JSON file to use
            json_path = get_video_metadata_path(video_name)
            
            # Lưu vào cache nếu extract_query_confidence có cache
            if hasattr(extract_query_confidence, 'path_cache'):
                if 'embeddings_path' not in extract_query_confidence.path_cache.get(path_cache_key, {}):
                    # Nếu chưa có embeddings_path, cần lấy thêm
                    embeddings_path = get_video_embeddings_path(video_name)
                    extract_query_confidence.path_cache[path_cache_key] = {
                        'embeddings_path': embeddings_path,
                        'json_path': json_path
                    }
                else:
                    # Nếu đã có embeddings_path, chỉ cập nhật json_path
                    extract_query_confidence.path_cache[path_cache_key]['json_path'] = json_path
        
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

def search_fallback(query, top_k, video_name=None):
    """
    Tìm kiếm dự phòng khi không có kết quả.
    
    Args:
        query: Query text
        top_k: Number of top results to return
        video_name: If provided, search only within this video's frames
    """
    try:
        processor = VietnameseTextProcessor()
        processed_text = processor.preprocess_and_translate(query)
        query_frames = search_top_frames(processed_text, top_k, video_name)
        
        # Determine which JSON file to use
        json_path = get_video_metadata_path(video_name)
        
        # Đọc dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
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
        # Use the video_data_mapping instead of scanning the old JSON file
        video_dict = {}
        videos_to_remove = []
        
        for idx, (video_name, video_info) in enumerate(list(video_data_mapping.items()), 1):
            video_path = video_info.get('video_path')
            if not video_path or not os.path.exists(video_path):
                # Mark this video for removal from the mapping
                videos_to_remove.append(video_name)
                print(f"Video not found, will be removed from mapping: {video_path}")
                continue
                
            # Create a unique ID for the video
            video_id = f"video-{idx}"
            
            # Get the first frame from the frames directory as thumbnail
            frames_dir = video_info.get('frames_dir')
            thumbnail = None
            if frames_dir and os.path.exists(frames_dir):
                frame_files = os.listdir(frames_dir)
                if frame_files:
                    # Sort frames to get the first one
                    frame_files.sort()
                    thumbnail = os.path.join(frames_dir, frame_files[0])
            
            video_dict[video_path] = {
                "id": video_id,
                "title": video_name,
                "thumbnail": thumbnail,
                "duration": get_video_duration(video_path),
                "uploadDate": time.strftime('%Y-%m-%d', time.gmtime(os.path.getctime(video_path))),
                "size": f"{os.path.getsize(video_path) // (1024 * 1024)} MB",
                "resolution": get_video_resolution(video_path),
                "path": video_path
            }
        
        # Remove videos that no longer exist from the mapping
        if videos_to_remove:
            for video_name in videos_to_remove:
                del video_data_mapping[video_name]
            # Save the updated mapping
            save_video_data_mapping()
            print(f"Removed {len(videos_to_remove)} non-existent videos from mapping")
        
        # If no videos found in mapping, fall back to the old method
        if not video_dict:
            with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for frame_data in data:
                video_path = frame_data.get('video')
                if video_path and video_path not in video_dict:
                    # Skip if video doesn't exist
                    if not os.path.exists(video_path):
                        continue
                        
                    video_name = os.path.basename(os.path.dirname(video_path))
                    video_filename = os.path.basename(video_path)
                    
                    # Create a unique ID for the video
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
    """
    API lấy sự kiện của một video cụ thể.
    
    Args:
        video_id: ID của video (ví dụ: video-1)
    """
    try:
        # Extract video name from video_id
        video_name = None
        video_path = None
        
        # If video_id is in format "video-X" where X is a number
        if video_id.startswith('video-'):
            video_num = int(video_id.split('-')[1])
            
            # Get list of videos from mapping
            all_videos = list(video_data_mapping.keys())
            
            if all_videos and len(all_videos) >= video_num:
                # Get video name
                video_name = all_videos[video_num-1]
                video_path = video_data_mapping[video_name]["video_path"]
                
        # If we couldn't find the video, try the legacy approach
        if not video_path:
            with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Find all videos
            all_video_paths = sorted(list(set([d.get('video') for d in data if d.get('video')])))
            
            if video_id.startswith('video-'):
                video_num = int(video_id.split('-')[1])
                if len(all_video_paths) >= video_num:
                    video_path = all_video_paths[video_num-1]
                    video_name = Path(video_path).stem
        
        # If we still couldn't find the video, return an error
        if not video_path:
            return jsonify({"error": f"Video with ID {video_id} not found"}), 404
            
        # Get events for this video
        events = []
        
        # Use video-specific metadata if available
        if video_name and video_name in video_data_mapping:
            metadata_file = video_data_mapping[video_name]["metadata_file"]
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # All frames in this file belong to the video
            for frame_data in data:
                events.append(format_event_for_frontend(frame_data))
        else:
            # Fallback to scanning the combined JSON file
            with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for frame_data in data:
                if frame_data.get('video') == video_path:
                    events.append(format_event_for_frontend(frame_data))
        
        # Filter to show a reasonable number of events
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
        start_time = time.time()
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
        
        # Lấy video_name nếu có để tìm kiếm trong phạm vi video cụ thể
        video_id = data.get("videoId")
        video_name = None
        
        if video_id and video_id.startswith('video-'):
            video_num = int(video_id.split('-')[1])
            all_videos = list(video_data_mapping.keys())
            if all_videos and len(all_videos) >= video_num:
                video_name = all_videos[video_num-1]
        
        # Nhận thông tin filter từ frontend
        enable_text_keyword = data.get("enableTextKeyword", use_keyword)
        enable_object_keyword = data.get("enableObjectKeyword", use_object) 
        enable_clip_similarity = data.get("enableClipSimilarity", False)
        
        # Debug log
        print(f"Search request: type={search_type}, method={search_method}, query='{query}', object='{object_keyword}'")
        print(f"Thresholds: adaptive={adaptive_threshold}, text={text_confidence}, object={object_confidence}")
        print(f"Enabled filters: text={enable_text_keyword}, object={enable_object_keyword}, clip={enable_clip_similarity}")
        print(f"Video filter: {video_name}")
        print(f"Keyword: '{keyword}'")
        
        results = []
        
        if search_type == "image" and image_url:
            # Tìm kiếm bằng hình ảnh - áp dụng ngưỡng adaptive_threshold trực tiếp
            image_results = search_by_image(image_url, adaptive_threshold, top_k, video_name)
            results = image_results
            print(f"Image search returned {len(image_results)} results")
        
        elif search_type == "text" and query:
            # Sử dụng phương pháp truy vấn dựa trên search_method và áp dụng các ngưỡng trực tiếp
            if search_method == "text_clip":
                # Truy vấn bằng văn bản thuần (CLIP Similarity)
                results = query_by_text_clip(
                    query, top_k, 
                    search_top_frames,
                    extract_query_confidence, 
                    format_event_for_frontend,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Text CLIP search returned {len(results)} results")
            
            elif search_method == "text_adaptive":
                # Truy vấn bằng văn bản + Adaptive threshold - áp dụng ngưỡng trực tiếp
                results = query_by_text_with_adaptive_threshold(
                    query, adaptive_threshold, top_k, 
                    search_top_frames,
                    extract_query_confidence, 
                    format_event_for_frontend,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Text adaptive search returned {len(results)} results")
            
            elif search_method == "keyword_only":
                # Truy vấn bằng keyword - áp dụng text_confidence trực tiếp
                actual_query = keyword if keyword else query
                results = query_by_keyword(
                    actual_query, text_confidence, top_k, 
                    search_frames_by_keyword, 
                    format_event_for_frontend,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Keyword only search returned {len(results)} results")
            
            elif search_method == "text_keyword":
                # Truy vấn bằng văn bản + keyword - áp dụng cả hai ngưỡng
                actual_keyword = keyword if keyword else query
                print(f"Text+keyword search for: text='{query}', keyword='{actual_keyword}'")
                
                results = query_by_text_and_keyword(
                    query, adaptive_threshold, top_k,
                    search_top_frames,
                    extract_query_confidence, 
                    search_frames_by_keyword,
                    format_event_for_frontend,
                    keyword=actual_keyword, 
                    text_confidence=text_confidence,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Text+keyword search returned {len(results)} results")
            
            elif search_method == "object_only":
                # Truy vấn bằng Object - áp dụng object_confidence trực tiếp
                actual_query = object_keyword if object_keyword else query
                print(f"Searching for object with keyword: '{actual_query}'")
                
                results = query_by_object(
                    actual_query, object_confidence, top_k, 
                    format_event_for_frontend, 
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Object only search returned {len(results)} results")
                
            elif search_method == "text_object":
                # Truy vấn bằng văn bản + Object - áp dụng cả hai ngưỡng
                actual_object = object_keyword if object_keyword else query
                print(f"Text+object search for: text='{query}', object='{actual_object}'")
                
                results = query_by_text_and_object(
                    query, adaptive_threshold, top_k,
                    search_top_frames,
                    extract_query_confidence, 
                    format_event_for_frontend,
                    object_keyword=actual_object, 
                    object_confidence=object_confidence,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Text+object search returned {len(results)} results")
            
            elif search_method == "text_object_keyword":
                # Truy vấn bằng văn bản + Object + keyword - áp dụng cả ba ngưỡng
                actual_keyword = keyword if keyword else query
                actual_object = object_keyword if object_keyword else query
                
                results = query_by_text_object_and_keyword(
                    query, adaptive_threshold, top_k,
                    search_top_frames,
                    extract_query_confidence,
                    search_frames_by_keyword,
                    format_event_for_frontend,
                    keyword=actual_keyword, 
                    text_confidence=text_confidence,
                    object_keyword=actual_object, 
                    object_confidence=object_confidence,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
                print(f"Text+object+keyword search returned {len(results)} results")
            
            # Fallback nếu không chọn phương pháp cụ thể
            else:
                results = query_by_text_with_adaptive_threshold(
                    query, adaptive_threshold, top_k, 
                    search_top_frames,
                    extract_query_confidence, 
                    format_event_for_frontend,
                    video_name=video_name,
                    video_data_mapping=video_data_mapping
                )
        
        # Đảm bảo tất cả kết quả đều có các trường confidence đầy đủ
        for result in results:
            if "text_confidence" not in result:
                result["text_confidence"] = 0.0
            if "object_confidence" not in result:
                result["object_confidence"] = 0.0
            if "clip_similarity" not in result:
                result["clip_similarity"] = 0.0
        
        # Filter video-specific events if necessary
        if video_name:
            # Ensure all results come from the specified video
            video_path = video_data_mapping.get(video_name, {}).get("video_path")
            if video_path:
                print(f"Before filtering: {len(results)} results")
                
                # Lấy tên file video để so sánh
                video_filename = Path(video_path).stem
                expected_video_id = f"video-{video_filename}"
                
                # Sửa điều kiện lọc để kiểm tra cả tên file video
                filtered_results = []
                for r in results:
                    r_video_id = r.get("videoId")
                    # Kiểm tra nhiều dạng videoId có thể có
                    if (r_video_id == f"video-{video_name}" or 
                        r_video_id == expected_video_id or
                        video_filename in r_video_id):
                        filtered_results.append(r)
                
                results = filtered_results
                print(f"After filtering for video '{video_name}': {len(results)} results")
        
        # Sắp xếp kết quả theo confidence
        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        
        search_time = time.time() - start_time
        print(f"""
                Search completed in {search_time:.2f}s:
                - Query: '{query}'
                - Method: {search_method}
                - Video filter: {video_name or 'All videos'}
                - Results: {len(results)} items found
                """)
                    
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
    
    # Directory for extracted frames
    frame_dir = os.path.join(BASE_DIR, video_name)
    frame_dir_process = r"{}".format(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Directory for saving the video
    BaseVideo_dir = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\static\\video_frame"
    dir_save_video = os.path.join(BaseVideo_dir, video_name)
    os.makedirs(dir_save_video, exist_ok=True)
    path_save_video = os.path.join(dir_save_video, video_file.filename)
    video_file.save(path_save_video)
    
    try:
        # Xử lý video
        extract_frames_from_video(path_save_video, frame_dir, threshold=30.0)
        
        # Trích xuất embedding (now saves to video-specific file)
        model_name = "ViT-B/32"
        embeddings_file = extract_and_save_embeddings_from_folder(frame_dir, model_name, video_name)
        
        # Xử lý metadata (now saves to video-specific file)
        metadata_file = process_images_in_folder(frame_dir_process, get_default_metadata_path(), path_save_video)
        
        # Update the video data mapping with new files
        video_data_mapping[video_name] = {
            "metadata_file": metadata_file,
            "embeddings_file": embeddings_file,
            "video_path": path_save_video,
            "frames_dir": frame_dir
        }
        
        # Save the updated mapping
        save_video_data_mapping()
        
        print(f"""
                Video processing complete:
                - Video name: {video_name}
                - Frames extracted: {len(os.listdir(frame_dir))}
                - Embeddings saved to: {embeddings_file}
                - Metadata saved to: {metadata_file}
                - Video mapping updated with {len(video_data_mapping)} videos total
                """)
        
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
            
        # Try to find in video-specific directories
        frame_name = os.path.basename(frame_path)
        for video_name, video_info in video_data_mapping.items():
            frames_dir = video_info.get('frames_dir')
            if frames_dir and os.path.exists(frames_dir):
                potential_path = os.path.join(frames_dir, frame_name)
                if os.path.exists(potential_path):
                    return send_file(potential_path, mimetype="image/jpeg")
        
        # Fallback: Thử tìm trong output_samples.json
        with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
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
        
        # Check in video mapping
        video_name = os.path.basename(video_path)
        for name, info in video_data_mapping.items():
            if name == video_name or os.path.basename(info.get('video_path', '')) == video_name:
                full_path = info.get('video_path')
                if full_path and os.path.exists(full_path):
                    return send_file(full_path, mimetype="video/mp4")
            
        # Fallback: Thử tìm trong output_samples.json
        with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for frame_data in data:
            filepath = frame_data.get('video')
            if filepath and (os.path.basename(filepath) == video_path or filepath == video_path):
                return send_file(filepath, mimetype="video/mp4")
        
        abort(404, description=f"Video {video_path} not found")
    except Exception as e:
        print(f"Error serving video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe-voice', methods=['POST'])
def transcribe_voice():
    """API endpoint for transcribing voice recordings"""
    try:
        # Check if a file was uploaded
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        
        # Get language selection (default to English)
        language = request.form.get('language', 'en_us')
        
        # Check if the file is valid
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
            
        # Save the file temporarily
        filename = secure_filename(f"voice_{int(time.time())}.mp3")
        filepath = os.path.join(VOICE_DIR, filename)
        audio_file.save(filepath)
        
        # Set up AssemblyAI API
        base_url = "https://api.assemblyai.com"
        headers = {
            "authorization": "96acc0cd7c1d486aa2e4fe3671647ff2",
            "content-type": "application/json"
        }
        
        # Upload the file to AssemblyAI
        with open(filepath, "rb") as f:
            upload_resp = requests.post(
                base_url + "/v2/upload",
                headers={"authorization": headers["authorization"]},
                data=f
            )
            
        # Check if upload was successful
        if upload_resp.status_code != 200:
            return jsonify({"error": "Failed to upload audio to transcription service"}), 500
            
        audio_url = upload_resp.json()["upload_url"]
        
        # Configure transcription request
        data = {
            "audio_url": audio_url,
            "speech_model": "universal",
            "language_code": language  # Use selected language
        }
        
        # Create transcription job
        url = base_url + "/v2/transcript"
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to create transcription job"}), 500
            
        transcript_id = response.json()['id']
        polling_endpoint = f"{base_url}/v2/transcript/{transcript_id}"
        
        # Poll until complete (with timeout)
        max_attempts = 20
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            res = requests.get(polling_endpoint, headers=headers).json()
            status = res.get("status")
            
            if status == 'completed':
                return jsonify({
                    "text": res["text"],
                    "audio_file": filename
                })
            elif status == 'error':
                return jsonify({"error": f"Transcription failed: {res.get('error', 'Unknown error')}"}), 500
            
            time.sleep(2)
            
        # If we get here, it timed out
        return jsonify({"error": "Transcription timed out"}), 504
        
    except Exception as e:
        print(f"Error in transcribe_voice: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)