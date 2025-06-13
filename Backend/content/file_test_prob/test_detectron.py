import cv2
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Thiết lập cấu hình
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng phát hiện
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng phát hiện
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/cascade_mask_rcnn_R_50_FPN_3x.yaml")

# Thiết lập cấu hình  Cascade Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng phát hiện
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/cascade_mask_rcnn_R_50_FPN_3x/139257044/model_final_f5b7b0.pkl")

# Tạo predictor
predictor = DefaultPredictor(cfg)

# Đọc ảnh và phát hiện đối tượng
image_path = 'E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo\\3034.jpg'
image = cv2.imread(image_path)
outputs = predictor(image)

# Trích xuất kết quả
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.tolist()  # Bounding boxes (x1, y1, x2, y2)
scores = instances.scores.tolist()  # Điểm tin cậy
classes = instances.pred_classes.tolist()  # Nhãn (id)

# Chuyển nhãn từ id sang tên
class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
detections = [
    {
        "box": box,
        "score": score,
        "class_id": class_id,
        "class_name": class_names[class_id],
    }
    for box, score, class_id in zip(boxes, scores, classes)
]

# Ghi kết quả ra file JSON
output_json_path = "detected_objects.json"
with open(output_json_path, "w") as f:
    json.dump(detections, f, indent=4)

print(f"Ket qua da duoc lau vao {output_json_path}")

# Hiển thị kết quả trên ảnh
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(instances)
cv2.imshow("Detected Objects", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
