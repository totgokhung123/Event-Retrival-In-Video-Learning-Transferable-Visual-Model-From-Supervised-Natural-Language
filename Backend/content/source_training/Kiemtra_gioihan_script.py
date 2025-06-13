from transformers import CLIPTokenizer
import json

# Load tokenizer của CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Load dữ liệu JSON
with open("Data_Script.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Kiểm tra caption nào vượt quá 77 tokens
limit = 77
over_token_limit = {}

for video_path, info in data.items():
    caption = info.get("caption", "")
    tokenized = tokenizer(caption, truncation=False, padding=False, return_tensors="pt")
    token_count = tokenized.input_ids.shape[1]
    
    if token_count > limit:
        over_token_limit[video_path] = {
            "token_count": token_count,
            "caption": caption
        }

# In kết quả
print(f"Có {len(over_token_limit)} video có caption vượt quá {limit} tokens:")
for path, detail in over_token_limit.items():
    print(f"- {path} ({detail['token_count']} tokens)\n  -> {detail['caption']}\n")
