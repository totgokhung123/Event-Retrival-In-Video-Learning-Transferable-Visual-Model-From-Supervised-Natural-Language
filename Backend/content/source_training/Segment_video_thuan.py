import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_frames(input_folder, output_folder, frames_per_video=10):
    """
    Trích xuất frame từ các video trong dataset
    
    Args:
        input_folder: Thư mục chứa dữ liệu video (có 2 folder con NonViolence và Violence)
        output_folder: Thư mục đầu ra để lưu các frame
        frames_per_video: Số lượng frame trích xuất từ mỗi video
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    
    # Lấy danh sách các folder con (NonViolence và Violence)
    classes = ["NonViolence", "Violence"]
    
    for class_name in classes:
        # Tạo thư mục cho mỗi lớp
        class_output_dir = os.path.join(output_folder, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Đường dẫn đến thư mục chứa video của lớp hiện tại
        class_input_dir = os.path.join(input_folder, class_name)
        
        # Lấy danh sách tất cả video trong thư mục
        video_files = [f for f in os.listdir(class_input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Đang xử lý {len(video_files)} video từ lớp {class_name}...")
        
        # Xử lý từng video
        for video_idx, video_file in enumerate(tqdm(video_files)):
            video_path = os.path.join(class_input_dir, video_file)
            
            # Tạo thư mục cho mỗi video
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(class_output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Đọc video
            cap = cv2.VideoCapture(video_path)
            
            # Kiểm tra xem video có mở được không
            if not cap.isOpened():
                print(f"Không thể mở video: {video_path}")
                continue
            
            # Lấy thông tin về video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                print(f"Video không có frame hoặc bị lỗi: {video_path}")
                continue
            
            # Tính toán các frame index cần trích xuất
            frame_indices = np.linspace(0, total_frames-1, frames_per_video, dtype=int)
            
            # Trích xuất frame tại các vị trí đã tính
            for i, frame_idx in enumerate(frame_indices):
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Lưu frame
                    frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
            # Giải phóng tài nguyên
            cap.release()
    
    print("Trích xuất frame hoàn tất!")

if __name__ == "__main__":
    # Đường dẫn thư mục
    input_dataset_folder = r"E:\Đồ án chuyên ngành\dataset\dataset"
    output_dataset_folder = r"E:\Đồ án chuyên ngành\out_dataset_2"
    
    # Trích xuất frame
    extract_frames(input_dataset_folder, output_dataset_folder, frames_per_video=15)
    
    # Thống kê số frame đã trích xuất
    violence_frames = 0
    nonviolence_frames = 0
    
    for class_name in ["Violence", "NonViolence"]:
        class_dir = os.path.join(output_dataset_folder, class_name)
        for video_dir in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_dir)
            if os.path.isdir(video_path):
                frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                if class_name == "Violence":
                    violence_frames += len(frames)
                else:
                    nonviolence_frames += len(frames)
    
    print(f"Tổng số frame Violence: {violence_frames}")
    print(f"Tổng số frame NonViolence: {nonviolence_frames}")
    print(f"Tổng số frame: {violence_frames + nonviolence_frames}")