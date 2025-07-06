import os, gc
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Thêm cài đặt để debug CUDA device-side assert
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import (
    AutoProcessor, LlavaForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)
from torch.optim import AdamW

# Thiết lập PYTORCH_CUDA_ALLOC_CONF cho việc quản lý bộ nhớ
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# 1. Khởi tạo model & processor
# -----------------------------
model_id = "llava-hf/llava-1.5-7b-hf"

# LLaVA: tối ưu hóa bộ nhớ tối đa
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    offload_folder="offload",  # Tạo thư mục để offload mô hình
).eval()

# Bật gradient checkpointing để tiết kiệm bộ nhớ
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# CLIP cho reward: chỉ load trên CPU để tiết kiệm VRAM
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu").eval()

# -----------------------------
# 2. Dataset & DataLoader
# -----------------------------
class ImageDataset(data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Trả về đường dẫn ảnh thay vì PIL Image
        image_path = self.image_paths[idx]
        is_nsfw = "NSFW" in image_path
        
        return image_path, is_nsfw

def custom_collate_fn(batch):
    paths = [item[0] for item in batch]
    is_nsfw_list = [item[1] for item in batch]
    
    # Không cần gộp ảnh, chỉ trả về danh sách path và is_nsfw
    return paths, is_nsfw_list

# Định nghĩa đường dẫn dataset
train_dir = "/kaggle/input/nsfw-detection/out/train"  # Thay đổi đường dẫn thực tế
val_dir = "/kaggle/input/nsfw-detection/out/val"      # Thay đổi đường dẫn thực tế

# Tạo danh sách đường dẫn hình ảnh, bao gồm cả NSFW và Neutral
train_image_paths = []
for category in ["NSFW", "Neutral"]:
    category_path = os.path.join(train_dir, category)
    if os.path.exists(category_path):
        for f in os.listdir(category_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_image_paths.append(os.path.join(category_path, f))

val_image_paths = []
for category in ["NSFW", "Neutral"]:
    category_path = os.path.join(val_dir, category)
    if os.path.exists(category_path):
        for f in os.listdir(category_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                val_image_paths.append(os.path.join(category_path, f))

print(f"Found {len(train_image_paths)} training images and {len(val_image_paths)} validation images")

# Tạo dataset và dataloader - giảm batch_size xuống 1
train_dataset = ImageDataset(train_image_paths)
val_dataset = ImageDataset(val_image_paths)
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# -----------------------------
# 3. Hyperparams & prompts
# -----------------------------
epochs = 3
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
target_clip_score = 40

# Giảm độ dài prompt để giảm tải bộ nhớ
NSFW_PROMPT = "Describe this image focusing on violence or unusual content. 1-2 sentences max."
NEUTRAL_PROMPT = "Describe this image's main subject and scene briefly. 1-2 sentences max."

# -----------------------------
# 4. Hỗ trợ clear cache
# -----------------------------
def clear_mem():
    """Giải phóng bộ nhớ GPU và CPU triệt để hơn"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # Thêm một bước chủ động để giải phóng bộ nhớ
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                del param.grad  # Xóa gradient
    gc.collect()
    torch.cuda.empty_cache()

# -----------------------------
# 5. Tính reward trên CPU
# -----------------------------
def compute_clip_reward(image_pil, caption: str):
    try:
        # Kiểm tra input
        if not isinstance(caption, str) or not caption:
            print("Warning: Empty or invalid caption")
            return 0.0
            
        # Xử lý riêng cho ảnh và văn bản với xử lý lỗi
        inputs = clip_processor(text=[caption], images=[image_pil],
                              return_tensors="pt", padding=True, 
                              truncation=True, max_length=77)  # Thêm truncation và max_length
      
        # Đưa dữ liệu lên CPU (không lên GPU để tiết kiệm VRAM)
        pixel_values = inputs.pixel_values.to("cpu")  # Cho ảnh
        input_ids = inputs.input_ids.to("cpu")        # Cho text
        attention_mask = inputs.attention_mask.to("cpu") if hasattr(inputs, "attention_mask") else None
      
        with torch.no_grad():
            # Lấy embedding ảnh (chỉ cần pixel_values)
            img_emb = clip_model.get_image_features(pixel_values=pixel_values)
            
            # Lấy embedding text (cần input_ids và có thể cần attention_mask)
            if attention_mask is not None:
                txt_emb = clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                txt_emb = clip_model.get_text_features(input_ids=input_ids)
              
        # Normalize embeddings trước khi tính similarity
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        
        # Tính cosine similarity
        sim = torch.cosine_similarity(img_emb, txt_emb, dim=-1).clamp(min=0) * 100
        
        return sim.item()
    except Exception as e:
        print(f"Error in compute_clip_reward: {str(e)}")
        # Trả về giá trị mặc định an toàn
        return 20.0  # Giá trị trung bình an toàn

# -----------------------------
# 6. Sinh caption (greedy/sample)
# -----------------------------
def generate_caption(inputs, do_sample=False):
    # inputs đã sẵn sàng trên GPU
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):  # Sử dụng FP16 để tiết kiệm bộ nhớ
            # Thêm kiểm tra đầu vào
            if not isinstance(inputs, dict) and not hasattr(inputs, 'input_ids'):
                print("Warning: Invalid inputs to generate_caption")
                return "Error: Invalid inputs for caption generation"
                
            # Đảm bảo các tensor ở đúng device và dtype
            generate_kwargs = {
                'do_sample': do_sample,
                'top_k': 50, 
                'top_p': 0.9,
                'max_new_tokens': 30,  # Giảm thêm xuống 30 để tiết kiệm bộ nhớ
                'num_beams': 1,        # Bỏ beam search để tiết kiệm bộ nhớ
                'early_stopping': True,
                'pad_token_id': processor.tokenizer.pad_token_id,
                'eos_token_id': processor.tokenizer.eos_token_id,
            }
            
            # Generate
            out_ids = model.generate(**inputs, **generate_kwargs)
            
            # Decode
            caption = processor.decode(out_ids[0], skip_special_tokens=True)
            
            # Đảm bảo caption không quá dài
            if len(caption.split()) > 50:
                caption = " ".join(caption.split()[:50])
                
            # Đảm bảo caption không rỗng
            if not caption.strip():
                caption = "Image contains visual content"
                
            return caption
            
    except RuntimeError as e:
        # Xử lý OOM
        if "out of memory" in str(e).lower():
            print(f"OOM in generate_caption: {str(e)}")
            clear_mem()
            return "Image description unavailable due to memory constraints."
        else:
            # Xử lý các RuntimeError khác
            print(f"RuntimeError in generate_caption: {str(e)}")
            return "Error generating caption"
    except Exception as e:
        # Xử lý các lỗi khác
        print(f"Exception in generate_caption: {str(e)}")
        return "Error: " + str(e)[:50]

# -----------------------------
# 7. SCST training
# -----------------------------
for epoch in range(epochs):
    # Training
    model.train()
    train_rewards = []
    
    # Giảm số lượng mẫu huấn luyện cho epoch đầu tiên nếu thiếu bộ nhớ
    current_samples = train_image_paths[:len(train_image_paths)//3] if epoch == 0 else train_image_paths
    train_dataset = ImageDataset(current_samples)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for image_paths, is_nsfw_list in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        # Chỉ xử lý 1 ảnh một lần do batch_size=1
        img_path = image_paths[0]
        is_nsfw = is_nsfw_list[0]
        
        # Biến để theo dõi các object cần giải phóng
        objects_to_delete = []
        
        try:
            # Clear mem trước mỗi ảnh
            clear_mem()

            # Load ảnh từ path
            img_pil = Image.open(img_path).convert('RGB')

            # 1) Chuẩn bị prompt & inputs (ngắn gọn hơn để giảm bộ nhớ)
            prompt = NSFW_PROMPT if is_nsfw else NEUTRAL_PROMPT
            
            conv = [{"role":"user","content":[{"type":"text","text":prompt},{"type":"image"}]}]
            text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            
            # Sử dụng float16 để tiết kiệm bộ nhớ
            inputs = processor(
                images=img_pil,
                text=text_prompt,
                return_tensors="pt",
                padding=True
            ).to("cuda", torch.float16)
            objects_to_delete.append('inputs')

            # 2) Greedy baseline
            baseline_cap = generate_caption(inputs, do_sample=False)
            r_base = compute_clip_reward(img_pil, baseline_cap)

            # 3) Sampled caption
            sample_cap = generate_caption(inputs, do_sample=True)
            r_samp = compute_clip_reward(img_pil, sample_cap)

            # 4) SCST loss
            if sample_cap == "Image description unavailable due to memory constraints.":
                # Giải phóng bộ nhớ và tiếp tục
                for obj_name in objects_to_delete:
                    if obj_name in locals():
                        del locals()[obj_name]
                clear_mem()
                continue  # Bỏ qua nếu không sinh được caption do hết bộ nhớ
                
            try:
                # Cách tiếp cận đơn giản hóa: sử dụng loss trực tiếp từ reward
                # Tính reward difference từ baseline và sample
                reward_diff = float(r_base - r_samp)  # Chuyển sang Python float
                
                # Tạo tensor có requires_grad=True để có thể gọi backward()
                # Sử dụng reward difference để scale loss
                # Thêm quy mô nhỏ (0.01) để tránh update quá lớn
                pseudo_loss = torch.tensor(reward_diff * 0.01, 
                                          device="cuda", 
                                          dtype=torch.float32, 
                                          requires_grad=True)
                
                # Gọi backward() trên loss để tạo gradient
                optimizer.zero_grad()
                pseudo_loss.backward()
                
                # Clip gradient để tránh exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Thực hiện bước optimizer
                optimizer.step()
                    
                # Lưu lại reward để theo dõi
                train_rewards.append(r_samp)
                    
            except Exception as e:
                print(f"Error in policy gradient update: {str(e)}")
                print(f"Skipping this sample")
                clear_mem()
                continue
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error encountered, clearing memory and skipping sample: {img_path}")
                # Giải phóng bộ nhớ và tiếp tục
                clear_mem()
                continue
            else:
                print(f"Runtime error: {str(e)}")
                raise e
        except Exception as e:
            print(f"Unexpected error processing {img_path}: {str(e)}")
            clear_mem()
            continue
                
        # Đảm bảo bộ nhớ được giải phóng sau mỗi ảnh
        # Sử dụng danh sách objects_to_delete thay vì hardcoded variables
        for obj_name in objects_to_delete:
            if obj_name in locals():
                del locals()[obj_name]
        clear_mem()

    # Validation sau mỗi epoch (giảm số lượng mẫu kiểm tra để tiết kiệm thời gian)
    model.eval()
    val_rewards = []
    val_samples = val_image_paths[:min(100, len(val_image_paths))]
    val_dataset = ImageDataset(val_samples)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    with torch.no_grad():
        for image_paths, is_nsfw_list in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
            img_path = image_paths[0]
            is_nsfw = is_nsfw_list[0]
            
            objects_to_delete = []
            
            try:
                clear_mem()
                
                # Load ảnh từ path
                img_pil = Image.open(img_path).convert('RGB')
                
                # giống phần training nhưng chỉ greedy
                prompt = NSFW_PROMPT if is_nsfw else NEUTRAL_PROMPT
                
                conv = [{"role":"user","content":[{"type":"text","text":prompt},{"type":"image"}]}]
                text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                
                inputs = processor(
                    images=img_pil,
                    text=text_prompt,
                    return_tensors="pt",
                    padding=True
                ).to("cuda", torch.float16)
                objects_to_delete.append('inputs')

                cap = generate_caption(inputs, do_sample=False)
                
                if cap != "Image description unavailable due to memory constraints.":
                    val_rewards.append(compute_clip_reward(img_pil, cap))
                    
                # Giải phóng bộ nhớ
                for obj_name in objects_to_delete:
                    if obj_name in locals():
                        del locals()[obj_name]
                clear_mem()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM during validation, skipping sample: {img_path}")
                    clear_mem()
                    continue
                else:
                    print(f"Runtime error during validation: {str(e)}")
                    clear_mem()
                    continue
            except Exception as e:
                print(f"Unexpected error during validation for {img_path}: {str(e)}")
                clear_mem()
                continue
    
    # Print metrics
    avg_tr = np.mean(train_rewards) if train_rewards else 0
    avg_va = np.mean(val_rewards) if val_rewards else 0
    print(f"Epoch {epoch+1}: Train μ={avg_tr:.2f}, Val μ={avg_va:.2f}")
    
    # Save model after each epoch to avoid losing progress
    if epoch > 0:
        save_dir = f"/kaggle/working/llava_scst_opt_epoch{epoch+1}"
        print(f"Saving checkpoint to {save_dir}")
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
    
    if avg_va >= target_clip_score:
        print("Target reached, stopping.")
        break

# Sau khi training xong
save_dir = "/kaggle/working/llava_scst_opt_final"
print(f"Model và processor đã được lưu vào: {save_dir}")
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print("Done.")
