from roboflow import Roboflow
import os

# 1. Đăng nhập bằng API key
rf = Roboflow(api_key="7OZDukTvAp4kHtIMjpuk")

# 2. Vào workspace (thường là username), chọn project
project = rf.workspace("totgo").project("yolov8x-infer")

model_path = r"E:\Đồ án tôt nghiệp\Yolov8"

# 3. Chọn version đầu tiên
version = project.version(1)

# 4. Deploy mô hình yolov8
version.deploy(
    model_type="yolov8",
    model_path=model_path
)