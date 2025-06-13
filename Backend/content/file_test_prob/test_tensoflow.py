import tensorflow as tf  
import numpy as np  
from PIL import Image  
import matplotlib.pyplot as plt  

# Tải mô hình từ TensorFlow Hub  
model = tf.saved_model.load("ssd_mobilenet_v2")  # Thay đổi đường dẫn cho mô hình của bạn  

# Đọc ảnh và chuyển đổi định dạng  
image_path = 'D:\\code\\projects\\git\\Data\\khung_hinh_1_1\\14043.jpg'  
image = Image.open(image_path)  
image_np = np.array(image)  

# Phát hiện đối tượng  
input_tensor = tf.convert_to_tensor(image_np[tf.newaxis, ...], dtype=tf.float32)  
detections = model(input_tensor)  

# Lấy các thông tin cần thiết từ kết quả phát hiện  
num_detections = int(detections.pop('num_detections'))  
detections = {key:value[0, :num_detections].numpy()   
              for key,value in detections.items()}  
detections['num_detections'] = num_detections  

# Lấy các độ chính xác (scores) và lớp (classes)  
scores = detections['detection_scores']  
boxes = detections['detection_boxes']  
classes = detections['detection_classes'].astype(int)  

# Thiết lập một ngưỡng để hiển thị các phát hiện (ví dụ: 0.5)  
threshold = 0.5  
valid_detections = scores >= threshold  

# Hiển thị hình ảnh với các bounding boxes  
plt.figure(figsize=(10, 10))  
plt.imshow(image_np)  

for i in range(num_detections):  
    if valid_detections[i]:  
        box = boxes[i]  
        ymin, xmin, ymax, xmax = (box[0] * image_np.shape[0], box[1] * image_np.shape[1],  
                                   box[2] * image_np.shape[0], box[3] * image_np.shape[1])  
        
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,  
                                            fill=False, color='red', linewidth=2))  
        plt.gca().text(xmin, ymin, f'Class: {classes[i]} Score: {scores[i]:.2f}',  
                       bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12)  

plt.axis('off')  
plt.show()