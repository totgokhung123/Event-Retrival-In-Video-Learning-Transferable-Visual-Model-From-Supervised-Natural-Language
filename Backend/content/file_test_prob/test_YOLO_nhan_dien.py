from ultralytics import YOLO
model = YOLO('D:/code/projects/git/yolov8x.pt') 

# Train the model
# train_results = model.train(
#     data="coco8.yaml",  
#     epochs=100, 
#     imgsz=640, 
#     device="cpu", 
# )

# metrics = model.val()

results = model('D:\\code\\projects\\git\\Data\\segmnet_video_demovideo2\\10343.jpg')
results[0].show()
path = model.export(format="onnx") 

print(results)