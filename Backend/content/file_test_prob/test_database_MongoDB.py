import os
import sys
import easyocr
from PIL import Image, UnidentifiedImageError
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO  

# Đảm bảo console hỗ trợ Unicode
sys.stdout.reconfigure(encoding='utf-8')

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
        print(f"Không thể đọc metadata từ file {image_path}: {str(e).encode('utf-8', errors='ignore').decode()}")
        return None
def process_image(file_path, reader, model):  
    """Xử lý một ảnh duy nhất và trả về sample JSON."""  
    metadata = get_image_metadata(file_path)  
    if not metadata:  
        return None  

    frame_id = os.path.splitext(os.path.basename(file_path))[0]  
    video_name = os.path.basename(os.path.dirname(file_path))
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

        # Phát hiện đối tượng bằng YOLOv8x  
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

        return {  
            "media_type": "image",  
            "filepath": file_path,  
            "tags": ["MainData"],  
            "metadata": metadata,  
            "video": video_name,  
            "frameid": frame_id,  
            "text_detections": {"detections": text_detections},  
            "object_detections": {"detections": object_detections},  
            "frameidx": int(frame_id) if frame_id.isdigit() else None,  
        }  
    except Exception as e:  
        print(f"Lỗi khi xử lý file {file_path}: {e}")  
        return None  

def process_images_in_folder(folder_path, db_collection, max_workers=4):
    """Xử lý tất cả các ảnh trong thư mục và lưu thông tin vào MongoDB."""
    reader = easyocr.Reader(['vi'], gpu=True)
    model = YOLO('yolov8x.pt')
    files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda file: process_image(file, reader,model), files)
        for result in filter(None, results):
            db_collection.insert_one(result)  # Lưu từng tài liệu vào MongoDB

    print("Thông tin đã được lưu vào MongoDB")

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['testmongoDACN']  # Tên database
collection = db['frames']      # Tên collection
# Sử dụng hàm
folder_path = 'E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo'
process_images_in_folder(folder_path, collection, max_workers=8)
