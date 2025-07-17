import json
from PIL import Image
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch

def main():
    # 1. Đường dẫn đến file JSON của bạn
    json_path = 'caption_results train.json'
    # 2. Đọc dữ liệu
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Khởi tạo model CLIP (ViT-B/14)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    results = []
    # 4. Duyệt từng ảnh + caption
    for img_path, info in data.items():
        caption = info.get('caption', "")
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=caption, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            image_emb = outputs.image_embeds   # (1, D)
            text_emb  = outputs.text_embeds    # (1, D)
            # 5. Cosine similarity → CLIPScore
            cos_sim = torch.nn.functional.cosine_similarity(image_emb, text_emb)
            score = max(cos_sim.item() * 100, 0)  # scale to [0,100]
        except Exception as e:
            score = None
            print(f"[Warning] Không thể xử lý: {img_path} → {e}")

			
        results.append({
            "image_path": img_path,
            "caption": caption,
            "clip_score": score
        })

    # 6. Chuyển thành DataFrame
    df = pd.DataFrame(results)

    # 7. Tính thống kê μ, σ và threshold = μ – 2σ
    valid = df['clip_score'].dropna().astype(float)
    mu = valid.mean()
    sigma = valid.std()
    threshold = mu - 2 * sigma

    # 8. Gắn cờ các caption dưới threshold
    df['flag_for_review'] = df['clip_score'] < threshold

    # 9. Xuất ra CSV
    output_csv = 'clipscore_results.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')

    # 10. In tóm tắt ra console
    print(f"➤ Kết quả đã lưu: {output_csv}")
    print(f"→ Mean CLIPScore (μ): {mu:.2f}")
    print(f"→ Std Dev (σ): {sigma:.2f}")
    print(f"→ Threshold (μ - 2σ): {threshold:.2f}")
    print("\nTop 10 captions cần review (score thấp nhất):")
    flagged = df[df['flag_for_review']].sort_values('clip_score').head(10)
    print(flagged[['image_path', 'clip_score']].to_string(index=False))

if __name__ == "__main__":
    main()
