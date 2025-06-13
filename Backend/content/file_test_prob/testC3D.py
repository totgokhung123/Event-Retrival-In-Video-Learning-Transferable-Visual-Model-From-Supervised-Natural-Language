import cv2
import numpy as np
import os

def simulate_content_detector(video_path, output_dir, threshold=30.0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Không thể đọc video.")
        return

    prev_yuv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)
    prev_y = prev_yuv[:, :, 0]  # Kênh độ sáng (Y)
    frame_id = 1
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret or saved:
            break

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0]

        # Tính sai khác tuyệt đối trung bình giữa kênh Y của hai khung hình
        diff = np.mean(cv2.absdiff(prev_y, y))

        print(f"Frame {frame_id} - Mean difference: {diff:.2f}")
        if diff > threshold:
            # Giả sử đây là chuyển cảnh → lưu frame giữa
            middle_frame = frame
            out_path = os.path.join(output_dir, f"scene_frame_{frame_id}.jpg")
            cv2.imwrite(out_path, middle_frame)
            print(f">>> Phát hiện cảnh! Đã lưu khung hình: {out_path}")
            saved = True

        prev_y = y
        frame_id += 1

    cap.release()
video_path = "E:\\THIHE\\testfitty one\\videotesst.mp4" 
output_dir = "E:\\Đồ án chuyên ngành\\19_12_2024"  
simulate_content_detector(video_path, output_dir, threshold=30.0)
print("Hoàn tất việc phát hiện cảnh trong video.")