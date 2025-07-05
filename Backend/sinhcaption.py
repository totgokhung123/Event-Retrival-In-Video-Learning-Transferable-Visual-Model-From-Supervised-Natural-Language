import os, gc
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    AutoProcessor, LlavaForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)
from torch.optim import AdamW

# -----------------------------
# 1. Khởi tạo model & processor
# -----------------------------
model_id = "llava-hf/llava-1.5-7b-hf"

# LLaVA: off‑load tự động, fp16
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
).eval()

# CLIP cho reward: chỉ load trên CPU để tiết kiệm VRAM
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel. from_pretrained("openai/clip-vit-base-patch32").to("cpu").eval()

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

# Tạo dataset và dataloader
train_dataset = ImageDataset(train_image_paths)
val_dataset = ImageDataset(val_image_paths)
train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
# DataLoader: shuffling chỉ cần ở train, batch_size có thể >1 vì SCST loop nội bộ vẫn 1

# -----------------------------
# 3. Hyperparams & prompts
# -----------------------------
epochs           = 3
optimizer        = AdamW(model.parameters(), lr=3e-5)
target_clip_score= 40

# NSFW_PROMPT    = "This image contains violence. Describe main actions in 1-2 sentences (≤77 words)."
# NEUTRAL_PROMPT = "Describe main subject, objects, scene in 1-2 sentences (≤77 words)."
# Prompt templates
NSFW_PROMPT = "This image contains violence. Please briefly describe what you saw, focusing on any signs of violence, unusual behavior, danger, or threats. Describe the main actions concisely in 1-2 sentences.Max 77 words to Describe it."
NEUTRAL_PROMPT = "Please describe the image concisely and naturally. Please briefly describe what you saw, Focus on the main subject, objects, scene, and actions. Briefly describe the main content in 1-2 sentences. Maximum 77 words for description."

# -----------------------------
# 4. Hỗ trợ clear cache
# -----------------------------
def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()

# -----------------------------
# 5. Tính reward trên CPU
# -----------------------------
def compute_clip_reward(image_pil, caption: str):
    # Xử lý riêng cho ảnh và văn bản
    inputs = clip_processor(text=[caption], images=[image_pil],
                            return_tensors="pt", padding=True)
    
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
            
    sim = torch.cosine_similarity(img_emb, txt_emb, dim=-1).clamp(min=0) * 100
    return sim.item()

# -----------------------------
# 6. Sinh caption (greedy/sample)
# -----------------------------
def generate_caption(inputs, do_sample=False):
    # inputs đã sẵn sàng trên GPU
    with torch.cuda.amp.autocast():  
        out_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            top_k=50, top_p=0.9,
            max_new_tokens=77,
            num_beams=1 if do_sample else 4,
            early_stopping=True
        )
    return processor.decode(out_ids[0], skip_special_tokens=True)

# -----------------------------
# 7. SCST training
# -----------------------------
for epoch in range(epochs):
    model.train()
    train_rewards = []

    for image_paths, is_nsfw_list in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        for img_path, is_nsfw in zip(image_paths, is_nsfw_list):
            # Clear mem trước mỗi ảnh
            clear_mem()

            # Load ảnh từ path
            img_pil = Image.open(img_path).convert('RGB')

            # 1) Chuẩn bị prompt & inputs
            prompt = NSFW_PROMPT if is_nsfw else NEUTRAL_PROMPT
            conv = [{"role":"user","content":[{"type":"text","text":prompt},{"type":"image"}]}]
            text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(images=img_pil, text=text_prompt,
                               return_tensors="pt", padding=True).to("cuda", torch.float16)

            # 2) Greedy baseline
            baseline_cap = generate_caption(inputs, do_sample=False)
            r_base = compute_clip_reward(img_pil, baseline_cap)

            # 3) Sampled caption
            sample_cap= generate_caption(inputs, do_sample=True)
            r_samp = compute_clip_reward(img_pil, sample_cap)
            train_rewards.append(r_samp)

            # 4) SCST loss
            loss = (r_base - r_samp) * model(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                labels=processor.tokenizer(sample_cap, return_tensors="pt").input_ids.to("cuda")
            ).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clear_mem()

    # Validation sau mỗi epoch
    model.eval()
    val_rewards = []
    with torch.no_grad():
        for image_paths, is_nsfw_list in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
            for img_path, is_nsfw in zip(image_paths, is_nsfw_list):
                clear_mem()
                
                # Load ảnh từ path
                img_pil = Image.open(img_path).convert('RGB')
                
                # giống phần training nhưng chỉ greedy
                prompt = NSFW_PROMPT if is_nsfw else NEUTRAL_PROMPT
                conv = [{"role":"user","content":[{"type":"text","text":prompt},{"type":"image"}]}]
                text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                inputs = processor(images=img_pil, text=text_prompt,
                                   return_tensors="pt", padding=True).to("cuda", torch.float16)

                cap = generate_caption(inputs, do_sample=False)
                val_rewards.append(compute_clip_reward(img_pil, cap))
                clear_mem()

    avg_tr = np.mean(train_rewards)
    avg_va = np.mean(val_rewards)
    print(f"Epoch {epoch+1}: Train μ={avg_tr:.2f}, Val μ={avg_va:.2f}")
    if avg_va >= target_clip_score:
        print("Target reached, stopping.")
        break

# Sau khi training xong, thay đổi đường dẫn lưu model:
save_dir = "/kaggle/working/llava_scst_opt"

model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print(f"Model và processor đã được lưu vào: {save_dir}")
print("Done.")
