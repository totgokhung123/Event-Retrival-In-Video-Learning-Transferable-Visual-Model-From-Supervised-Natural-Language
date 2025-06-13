import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 1. Encode caption
caption = "A man is seen punching another man in the face, causing him to fall to the ground. The woman in the yellow shirt watches in shock."
text_token = clip.tokenize([caption]).to(device)
with torch.no_grad():
    text_feat = model.encode_text(text_token)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)


# 2. Load all frames
frame_folder = "E:/Đồ án chuyên ngành/testing/19_12_2024/dataset_RLVSD/Violence/V_39"  # all frames of V_1.mp4
frame_scores = []

for frame_file in sorted(os.listdir(frame_folder)):
    img_path = os.path.join(frame_folder, frame_file)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = model.encode_image(image)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)


    # 3. Compute similarity
    sim = (image_feat @ text_feat.T).item()
    frame_scores.append((img_path, sim))

# 4. Select best frame
best_frame = max(frame_scores, key=lambda x: x[1])
print("Best frame:", best_frame[0], "with similarity:", best_frame[1])