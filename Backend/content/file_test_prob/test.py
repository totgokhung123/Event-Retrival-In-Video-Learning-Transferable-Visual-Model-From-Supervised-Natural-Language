import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

def extract_frames_from_video(video_path, output_dir, threshold=30.0):
    os.makedirs(output_dir, exist_ok=True)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        middle_frame = (start_frame + end_frame) // 2
        cap = cv2.VideoCapture(video_path)
        frames_to_capture = {middle_frame}  
        for frame_idx in frames_to_capture:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_dir, f"{frame_idx}.jpg")
                cv2.imwrite(frame_filename, frame)  
                print(f"Saved frame: {frame_filename}")
        cap.release()
    video_manager.release()
    print("Video extraction complete.")
video_path = "D:\\code\\projects\\git\\Data\\video\\L01_V001.mp4" 
output_dir = "D:\\code\\projects\\git\\Data\\segmnet_video_demovideo2"  
extract_frames_from_video(video_path,output_dir)