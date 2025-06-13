import os
import gc
import torch
import json
import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ⚠️ Khởi tạo môi trường
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🧠 Load model 1 lần
print("Loading model...")
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).cuda().eval()
print("Model loaded successfully!")

# 🧹 Hàm mô tả ảnh & giải phóng bộ nhớ
def describe_image(image_path):
    torch.cuda.empty_cache()
    gc.collect()

    # Prompt chuẩn
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This image contains violent, sexually explicit, or gory content. Please briefly describe what you see, focusing on any violence, nudity, danger, or threats. Briefly describe the main content in 1-2 sentences. Maximum 77 words for description."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load ảnh
    image = Image.open(image_path).convert("RGB")

    # Chuẩn bị input (trên CPU, sau đó chuyển qua CUDA từng phần)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    for k in inputs:
        if inputs[k].dtype == torch.float32:
            inputs[k] = inputs[k].to(dtype=torch.float16)
        inputs[k] = inputs[k].cuda()

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=77, do_sample=False)

    result = processor.decode(output[0][2:], skip_special_tokens=True)

    # 🔥 Giải phóng bộ nhớ sau mỗi lần gọi
    del image, inputs, output
    torch.cuda.empty_cache()
    gc.collect()

    return result

def process_dataset(base_dir, output_json_path):
    """Xử lý tất cả các hình ảnh trong thư mục và tạo JSON output
    
    Args:
        base_dir: Thư mục gốc chứa dataset
        output_json_path: Đường dẫn để lưu file JSON kết quả
    """
    # Kiểm tra và tải JSON hiện có nếu có
    results = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Đã tải {len(results)} kết quả từ file có sẵn.")
        except:
            print("Không thể tải file JSON có sẵn, bắt đầu từ đầu.")
    
    # Thư mục con để quét
    subdirs = ['train', 'test', 'val']
    
    # Quét qua các thư mục
    for subdir in subdirs:
        nsfw_dir = os.path.join(base_dir, subdir, 'NSFW')
        if not os.path.exists(nsfw_dir):
            print(f"Thư mục {nsfw_dir} không tồn tại, bỏ qua.")
            continue
            
        print(f"Đang xử lý các hình ảnh trong {nsfw_dir}...")
        
        # Lấy tất cả file ảnh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(nsfw_dir).glob(ext)))
        
        print(f"Tìm thấy {len(image_files)} ảnh để xử lý.")
        
        # Xử lý từng ảnh với thanh tiến trình
        for img_path in tqdm.tqdm(image_files):
            img_path_str = str(img_path)
            
            # Bỏ qua ảnh đã xử lý
            if img_path_str in results:
                continue
                
            try:
                # Sinh caption cho ảnh
                caption = describe_image(img_path_str)
                
                # Lưu kết quả
                results[img_path_str] = {
                    "category": "Violence",
                    "caption": caption
                }
                
                # Lưu tạm kết quả sau mỗi 20 ảnh để tránh mất dữ liệu nếu bị lỗi
                if len(results) % 20 == 0:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"Đã lưu tạm thời {len(results)} kết quả")
                
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path_str}: {e}")
    
    # Lưu kết quả cuối cùng
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Hoàn thành! Đã lưu kết quả vào {output_json_path}")
    print(f"Tổng số ảnh đã xử lý: {len(results)}")

if __name__ == "__main__":
    # Thư mục chứa dataset
    dataset_dir = "E:\\NSFW\\out"
    
    # Đường dẫn lưu file JSON output
    output_path = "E:\\NSFW\\caption_results.json"
    
    # Xử lý dataset
    process_dataset(dataset_dir, output_path) 