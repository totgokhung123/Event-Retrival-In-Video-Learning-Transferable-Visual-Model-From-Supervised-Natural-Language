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

# Import service initializer
from services import initialize_services

# Base directories
BASE_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\static\\processed_frames"
METADATA_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\metadata"
EMBEDDING_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\embedding"

# Create necessary directories if they don't exist
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Initialize all services
service_container = initialize_services(BASE_DIR, METADATA_DIR, EMBEDDING_DIR)

# Extract services from container
path_service = service_container['path_service']
cache_service = service_container['cache_service']
data_service = service_container['data_service']
embedding_service = service_container['embedding_service']
search_service = service_container['search_service']
model = service_container['model']
preprocess = service_container['preprocess']
device = service_container['device']
services_loaded = service_container['services_loaded']

# Import and initialize visualization service
from services.visualization_service import VisualizationService
visualization_service = VisualizationService(path_service, embedding_service, data_service, cache_service)

# For backwards compatibility, keep existing code
video_data_mapping = path_service.video_data_mapping

# Create a directory for voice uploads
VOICE_DIR = os.path.join(os.path.dirname(__file__), "static", "voice")
os.makedirs(VOICE_DIR, exist_ok=True)

# Forwarding functions to services for backward compatibility

def load_video_data_mapping():
    """Load mapping between videos and their metadata/embedding files."""
    return path_service.video_data_mapping

def save_video_data_mapping():
    """Save the current video data mapping to a file."""
    path_service.save_video_data_mapping()

# Load existing mapping at startup
video_data_mapping = load_video_data_mapping()

def get_default_metadata_path():
    """Get the default metadata path for backward compatibility."""
    return path_service.get_default_metadata_path()

def get_default_embeddings_path():
    """Get the default embeddings path for backward compatibility."""
    return path_service.get_default_embeddings_path()

def get_video_metadata_path(video_name=None):
    """Get the metadata file path for a specific video or the default one."""
    return path_service.get_metadata_path(video_name)

def get_video_embeddings_path(video_name=None):
    """Get the embeddings file path for a specific video or the default one."""
    return path_service.get_embeddings_path(video_name)

def load_frames_from_json(json_path, video_name=None):
    """Load danh sách tên file từ file JSON."""
    return data_service.load_frames_from_json(video_name)

def load_frames_mapping_from_json(json_path, video_name=None):
    """Load mapping from frame file names to full paths."""
    return data_service.load_frames_mapping_from_json(video_name)

FRAMES_MAPPING = load_frames_mapping_from_json(get_default_metadata_path())
app = Flask(__name__)
# Thêm CORS support
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Utility functions 
def get_video_duration(video_path):
    """Lấy thời lượng video từ file."""
    return data_service.get_video_duration(video_path)

def get_video_resolution(video_path):
    """Lấy độ phân giải video."""
    return data_service.get_video_resolution(video_path)

def format_event_for_frontend(frame_data):
    """Format dữ liệu event phù hợp với frontend React."""
    return data_service.format_event_for_frontend(frame_data)

# Các hàm core được sử dụng bởi cả template cũ và API mới
def search_frames_by_keyword(keyword, top_k, video_name=None):
    """
    Tìm kiếm frame theo từ khóa text.
    """
    return search_service.search_frames_by_keyword(keyword, top_k, video_name)

def extract_query_confidence(frame_path, query, video_name=None):
    """
    Trích xuất độ tương đồng giữa một khung hình và query text.
    """
    return embedding_service.extract_query_confidence(frame_path, query, video_name)

def filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence, video_name=None):
    """
    Lọc frame dựa trên từ khóa và độ tin cậy.
    """
    return search_service.filter_frame_by_keyword_and_confidence(
        frame, keyword, keyword_frames, min_confidence, video_name)

def search_top_frames(query, top_k, video_name=None):
    """
    Tìm kiếm top frames dựa trên query text sử dụng CLIP.
    """
    return embedding_service.search_top_frames(query, top_k, video_name)

def search_top_frames_by_image(image_features, top_k, video_name=None):
    """
    Tìm kiếm top frames dựa trên image features sử dụng CLIP.
    """
    return embedding_service.search_top_frames_by_image(image_features, top_k, video_name)

# Thêm 7 hàm xử lý truy vấn mới
def search_by_image(image_url, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm bằng hình ảnh.
    """
    return search_service.search_by_image(image_url, adaptive_threshold, top_k, video_name, preprocess)

def search_semantic_with_clip(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm ngữ nghĩa với CLIP.
    """
    return search_service.search_semantic_with_clip(query, adaptive_threshold, top_k, video_name)

def search_by_keyword(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm theo từ khóa.
    """
    return search_service.search_by_keyword(query, adaptive_threshold, top_k, video_name)

def search_by_object(query, adaptive_threshold, top_k, video_name=None):
    """
    Tìm kiếm theo object.
    """
    return search_service.search_by_object(query, adaptive_threshold, top_k, video_name)

def search_fallback(query, top_k, video_name=None):
    """
    Tìm kiếm dự phòng khi không có kết quả.
    """
    return search_service.search_fallback(query, top_k, video_name)

def deduplicate_results(all_results):
    """Loại bỏ các kết quả trùng lặp."""
    return search_service.deduplicate_results(all_results)

def sort_by_confidence(results):
    """Sắp xếp kết quả theo độ tin cậy."""
    return search_service.sort_by_confidence(results)

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
            response = send_file(frame_path, mimetype="image/jpeg")
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        
        # Kiểm tra trong frames mapping
        if frame_path in FRAMES_MAPPING:
            response = send_file(FRAMES_MAPPING[frame_path], mimetype="image/jpeg")
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
            
        # Try to find in video-specific directories
        frame_name = os.path.basename(frame_path)
        for video_name, video_info in video_data_mapping.items():
            frames_dir = video_info.get('frames_dir')
            if frames_dir and os.path.exists(frames_dir):
                potential_path = os.path.join(frames_dir, frame_name)
                if os.path.exists(potential_path):
                    response = send_file(potential_path, mimetype="image/jpeg")
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    return response
        
        # Fallback: Thử tìm trong output_samples.json
        with open(get_default_metadata_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for frame_data in data:
            filepath = frame_data.get('filepath')
            if filepath and (os.path.basename(filepath) == frame_path or filepath == frame_path):
                response = send_file(filepath, mimetype="image/jpeg")
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
        
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

@app.route('/api/visualization/umap', methods=['POST'])
def api_get_umap_visualization():
    """API trực quan hóa UMAP cho dữ liệu embedding."""
    try:
        data = request.json
        video_names = data.get('video_names')  # None = tất cả video
        n_neighbors = data.get('n_neighbors', 15)
        min_dist = data.get('min_dist', 0.1)
        n_components = data.get('n_components', 2)
        metric = data.get('metric', 'cosine')
        
        print(f"UMAP request with parameters: video_names={video_names}, n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        # Sử dụng visualization_service
        result = visualization_service.generate_umap_visualization(
            video_names=video_names,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        
        if result is None:
            return jsonify({"error": "No embeddings found"}), 404
        
        print(f"UMAP generation successful with {len(result['coordinates'])} points")
        return jsonify(result)
    except Exception as e:
        print(f"Error generating UMAP visualization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/videos/available', methods=['GET'])
def api_get_available_videos():
    """API lấy danh sách video có sẵn cho visualization."""
    try:
        # Lọc các video có sẵn embeddings
        available_videos = []
        for video_name, video_info in video_data_mapping.items():
            if not video_name.startswith("default"):
                embeddings_file = video_info.get('embeddings_file')
                if embeddings_file and os.path.exists(embeddings_file):
                    available_videos.append({
                        "name": video_name,
                        "embeddings_file": embeddings_file,
                        "video_path": video_info.get('video_path', '')
                    })
        
        return jsonify({
            "available_videos": available_videos,
            "count": len(available_videos)
        })
    except Exception as e:
        print(f"Error getting available videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)