# from deep_translator import GoogleTranslator

# translated = GoogleTranslator(source='vi', target='en').translate("Tôi rất thích xem các bức ảnh đẹp về thiên nhiên , hi.")
# print(translated)

# import json
# import requests

# # Run inference on an image
# url = "https://predict.ultralytics.com"
# headers = {"x-api-key": "01630c6df0cbd546054e37ea789e720ad805fd07fe"}
# data = {"model": "https://hub.ultralytics.com/models/R6nMlK6kQjSsQ76MPqQM", "imgsz": 640, "conf": 0.25, "iou": 0.45}
# with open("E:/Đồ án tôt nghiệp/source_code/Backend/static/processed_frames/video_test_3/22015.jpg", "rb") as f:
# 	response = requests.post(url, headers=headers, data=data, files={"file": f})

# # Check for successful response
# response.raise_for_status()

# # Print inference results
# print(json.dumps(response.json(), indent=2))

# import os
# import requests

# folder_path = "E:/Đồ án tôt nghiệp/source_code/Backend/static/processed_frames/video_test_3"
# url = "https://TienBinh-easyocr-api-test-1.hf.space/run/predict"

# for file_name in os.listdir(folder_path):
#     if file_name.lower().endswith(".jpg"):
#         image_path = os.path.join(folder_path, file_name)
#         with open(image_path, "rb") as f:
#             image_bytes = f.read()
#         response = requests.post(
#             url,
#             files={"data": (file_name, image_bytes, "image/jpeg")},
#         )
#         print(f"📄 {file_name} → {response.json()}")

# import os
# from gradio_client import Client
# from PIL import Image

# client = Client("TienBinh/easyocr-api-test-1")
# folder = "E:/Đồ án tôt nghiệp/source_code/Backend/static/processed_frames/video_test_3"

# for fname in os.listdir(folder):
#     if fname.lower().endswith((".jpg",".png")):
#         img = Image.open(os.path.join(folder, fname))
#         text = client.predict(img, api_name="/predict")
#         print(f"{fname} → {text}")


from gradio_client import Client

client = Client("TienBinh/easyocr-api-test-1")

# Truyền đường dẫn file, vì inputs=gr.Image(type="filepath")
image_path = "E:/Đồ án tôt nghiệp/source_code/Backend/static/processed_frames/video_test_3/22015.jpg"

# Đúng: truyền file path (string)
result = client.predict(image_path, api_name="/predict")

print(result)
