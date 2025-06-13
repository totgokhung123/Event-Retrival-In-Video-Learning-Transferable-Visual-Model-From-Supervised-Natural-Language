import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import gc

# Import thư viện cho Florence-2
from transformers import AutoProcessor, AutoModelForCausalLM

# Thiết lập đường dẫn
DATASET_ROOT = "/kaggle/input/dataset-rlvsd/dataset_RLVSD"  # Đường dẫn tới dataset

# Cấu hình các model và task prompts
MODEL_CONFIGS = [
    {
        "model_path": "microsoft/Florence-2-base",
        "model_name": "base",
        "tasks": [
            {
                "prompt": "<CAPTION>",
                "output_file": "/kaggle/working/florence2_captions_base.json"
            },
            {
                "prompt": "<DETAILED_CAPTION>",
                "output_file": "/kaggle/working/florence2_detail_captions_base.json"
            }
        ]
    },
    {
        "model_path": "microsoft/Florence-2-large",
        "model_name": "large",
        "tasks": [
            {
                "prompt": "<CAPTION>",
                "output_file": "/kaggle/working/florence2_captions_large.json"
            },
            {
                "prompt": "<DETAILED_CAPTION>",
                "output_file": "/kaggle/working/florence2_detail_captions_large.json"
            }
        ]
    }
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_florence2_model(model_path):
    """Tải model Florence-2"""
    print(f"Đang tải Florence-2 model: {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True
        # Sử dụng precision mặc định của model
    ).to(DEVICE).eval()
    
    return processor, model

def generate_caption(image_path, processor, model, task_prompt):
    """Tạo caption cho ảnh sử dụng Florence-2"""
    try:
        # Đọc ảnh
        image = Image.open(image_path).convert("RGB")
        
        # Chuẩn bị input cho model
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        # Sinh caption với model
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
                do_sample=False,
                num_beams=3,
                temperature=0.2,
                early_stopping=True,
            )
        
        # Giải mã output thành text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Giải phóng bộ nhớ
        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        
        return generated_text.strip()
    
    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {e}")
        # Giải phóng bộ nhớ nếu có lỗi
        torch.cuda.empty_cache()
        gc.collect()
        return f"Error: {str(e)}"

def process_dataset(processor, model, task_config):
    """Xử lý dataset với model và task prompt cụ thể"""
    task_prompt = task_config["prompt"]
    output_file = task_config["output_file"]
    
    results = []
    start_time = datetime.now()
    
    print(f"\nĐang xử lý với task prompt: {task_prompt}, output file: {output_file}")
    
    for class_name in ["Violence", "NonViolence"]:
        class_folder = os.path.join(DATASET_ROOT, class_name)
        
        print(f"\nĐang xử lý thư mục {class_name}...")
        subfolders = [f for f in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, f))]
        
        for subfolder in tqdm(subfolders, desc=f"Thư mục con trong {class_name}"):
            subfolder_path = os.path.join(class_folder, subfolder)
            
            # Lấy tất cả các file hình ảnh trong thư mục con
            image_files = [f for f in os.listdir(subfolder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Nếu có nhiều hình ảnh, chỉ xử lý một số hình đại diện để tiết kiệm thời gian
            if len(image_files) > 5:
                # Lấy 3 hình đầu, giữa và cuối
                selected_images = [image_files[0], 
                                 image_files[len(image_files) // 2], 
                                 image_files[-1]]
            else:
                selected_images = image_files
            
            for image_file in selected_images:
                image_path = os.path.join(subfolder_path, image_file)
                
                # Tạo caption với task prompt cụ thể
                caption = generate_caption(image_path, processor, model, task_prompt)
                
                # Lưu kết quả
                result = {
                    "image_path": image_path,
                    "class": class_name,
                    "subfolder": subfolder,
                    "caption": caption,
                    "task_prompt": task_prompt
                }
                results.append(result)
                
                # In kết quả ra console để theo dõi
                print(f"[{class_name}/{subfolder}] {image_file}: {caption[:100]}...")
                
                # Lưu file JSON sau mỗi 10 ảnh để đảm bảo không mất kết quả nếu script bị ngắt giữa chừng
                if len(results) % 10 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)

    # Lưu kết quả cuối cùng vào file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Hiển thị thống kê
    end_time = datetime.now()
    print(f"\nHoàn thành task {task_prompt}! Đã xử lý {len(results)} hình ảnh trong {(end_time - start_time).total_seconds() / 60:.2f} phút")
    print(f"Kết quả đã được lưu vào file: {output_file}")

def main():
    # Duyệt qua từng cấu hình model
    for model_config in MODEL_CONFIGS:
        model_path = model_config["model_path"]
        model_name = model_config["model_name"]
        
        print(f"\n===== Đang xử lý model: {model_name} =====")
        
        # Tải model
        processor, model = setup_florence2_model(model_path)
        
        # Xử lý tất cả các task cho model này
        for task_config in model_config["tasks"]:
            process_dataset(processor, model, task_config)
            
        # Giải phóng bộ nhớ sau khi xử lý xong model này
        del processor, model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Đã hoàn thành xử lý model {model_name}")

if __name__ == "__main__":
    main()
