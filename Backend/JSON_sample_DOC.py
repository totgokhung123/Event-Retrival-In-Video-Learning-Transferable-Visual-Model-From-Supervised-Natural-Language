import os
import easyocr
from PIL import Image, UnidentifiedImageError
import json
import uuid  # Để tạo id độc nhất
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from ultralytics import YOLO  

def get_image_metadata(image_path):
    """Lấy metadata cơ bản của ảnh."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return {
                "size_bytes": os.path.getsize(image_path),
                "mime_type": Image.MIME.get(img.format, "unknown"),
                "width": width,
                "height": height,
                "num_channels": len(img.getbands()),
            }
    except (UnidentifiedImageError, OSError) as e:
        print(f"Không thể đọc metadata từ file {image_path}: {e}")
        return None


def process_image(file_path, reader,path_video, model):
    """Xử lý một ảnh duy nhất và trả về sample JSON."""
    metadata = get_image_metadata(file_path)
    if not metadata:
        return None

    try:
        # Nhận diện văn bản bằng EasyOCR
        ocr_results = reader.readtext(file_path, detail=1)  
        text_detections = [  
            {  
                "label": text,  
                "bounding_box": [  
                    bbox[0][0] / metadata["width"],  # x0 (top-left)  
                    bbox[0][1] / metadata["height"], # y0 (top-left)  
                    (bbox[2][0] - bbox[0][0]) / metadata["width"],  # width  
                    (bbox[2][1] - bbox[0][1]) / metadata["height"], # height  
                ],  
                "confidence": prob,  
            }  
            for bbox, text, prob in ocr_results  
        ] 
        yolo_results = model(file_path)  
        object_detections = []  
        for result in yolo_results:  
            for box in result.boxes:  
                # Lấy tọa độ bounding box và nhãn  
                x1, y1, x2, y2 = box.xyxy[0]  
                confidence = box.conf  
                class_id = box.cls  
                label = model.names[int(class_id)] if int(class_id) in model.names else "unknown"  
                object_detections.append({  
                    "label": label,  
                    "bounding_box": [  
                        float(x1) / metadata["width"],  # x0 (top-left)  
                        float(y1) / metadata["height"], # y0 (top-left)  
                        (float(x2) - float(x1)) / metadata["width"],  # width  
                        (float(y2) - float(y1)) / metadata["height"], # height  
                    ],  
                    "confidence": float(confidence),  
                })
        # Escape đường dẫn path_video
        # escaped_path_video = path_video.replace("\\", "\\\\")
        return {
            "id": str(uuid.uuid4()), 
            "media_type": "image",
            "filepath": file_path,
            "tags": ["MainData"],
            "metadata": metadata,
            "video": path_video,
            "frameid": os.path.basename(file_path),
            "text_detections": {"detections": text_detections},  
            "object_detections": {"detections": object_detections},
            "frameidx": int(os.path.splitext(os.path.basename(file_path))[0]) 
                        if os.path.splitext(os.path.basename(file_path))[0].isdigit() else None,
        }
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None


def process_images_in_folder(folder_path, output_json_path,path_video, max_workers=4):
    reader = easyocr.Reader(['vi'], gpu=True)
    model = YOLO('yolov8x.pt') 
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []
    
    files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    # Dùng partial để đóng gói các tham số bổ sung
    process_with_params = partial(process_image, reader=reader, path_video=path_video,model=model)

    # Dùng ThreadPoolExecutor để xử lý đa luồng
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_with_params, files), 
                            total=len(files), desc="Processing Frames"))
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:

    #     results = list(tqdm(executor.map(lambda f: process_image(f, reader), files, path_video), 
    #                         total=len(files), desc="Processing Frames"))
    new_data = [res for res in results if res]
    existing_data.extend(new_data)
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    print(f"Updated JSON has been saved to {output_json_path}")
# def escape_path(path):
#     """Hàm escape đường dẫn video"""
#     return path.replace("\\", "\\\\")
# process_images_in_folder('D:\code\projects\git\19_12_2024\static\processed_frames\L01_V001',"output_samples.json", 'D:\code\projects\git\19_12_2024\static\video_frame\L01_V001\L01_V001.mp4')