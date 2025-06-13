import torch  
import cv2  
import numpy as np  
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  
from PIL import Image  
from clipcap import ClipCaptionModel  # Đảm bảo rằng bạn đã tải mã nguồn ClipCap  

# Hàm để trích xuất khung hình từ video  
def extract_frames(video_path, num_frames=16):  
    cap = cv2.VideoCapture(video_path)  
    frames = []  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    step = max(1, total_frames // num_frames)  # Chọn khung hình cách đều  

    for i in range(num_frames):  
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)  
        ret, frame = cap.read()  
        if not ret:  
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)  

    cap.release()  
    return frames  

# Hàm để tiền xử lý khung hình  
def preprocess_frame(frame):  
    transform = Compose([  
        Resize((224, 224)),  
        CenterCrop((224, 224)),  
        ToTensor(),  
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])  
    return transform(Image.fromarray(frame)).unsqueeze(0)  

# Tải mô hình ClipCap  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
clipcap_model = ClipCaptionModel(prefix_length=10).to(device)  
clipcap_model.load_state_dict(torch.load("path_to_clipcap_weights.pt", map_location=device))  
clipcap_model.eval()  

# Hàm để tạo mô tả từ khung hình  
def generate_caption(frame):  
    frame_tensor = preprocess_frame(frame).to(device)  
    with torch.no_grad():  
        prefix = clipcap_model(frame_tensor)  
        # Tạo mô tả từ prefix  
        # (Thêm mã để tạo mô tả từ prefix ở đây)  
    return "Mô tả video sẽ được tạo ra ở đây."  

# Hàm để tạo mô tả từ video  
def generate_video_caption(video_path):  
    frames = extract_frames(video_path)  
    captions = []  
    for frame in frames:  
        caption = generate_caption(frame)  
        captions.append(caption)  
    return captions  

# Đường dẫn video  
video_path = "D:\\code\\projects\\git\\Data\\video\\L01_V001.mp4"  # Thay bằng đường dẫn video của bạn  
captions = generate_video_caption(video_path)  

# In kết quả  
print("Mô tả video:")  
for i, caption in enumerate(captions):  
    print(f"Khung hình {i + 1}: {caption}")