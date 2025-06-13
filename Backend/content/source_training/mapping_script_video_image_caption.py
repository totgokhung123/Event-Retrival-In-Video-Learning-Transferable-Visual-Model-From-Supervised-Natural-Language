# import json
# import os
# import torch
# import clip
# from PIL import Image

# # --- 1. Load CLIP ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # --- 2. Load JSON script gốc ---
# with open("Data_Script.json", "r", encoding="utf-8") as f:
#     video_data = json.load(f)

# # --- 3. Hàm chọn frame tốt nhất với CLIP ---
# def select_best_frame(frame_folder: str, caption: str):
#     # Encode caption
#     text_token = clip.tokenize([caption]).to(device)
#     with torch.no_grad():
#         text_feat = model.encode_text(text_token)
#         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

#     best = (None, -1.0)
#     # Duyệt frames
#     for fn in sorted(os.listdir(frame_folder)):
#         path = os.path.join(frame_folder, fn)
#         try:
#             img = preprocess(Image.open(path)).unsqueeze(0).to(device)
#         except:
#             continue
#         with torch.no_grad():
#             img_feat = model.encode_image(img)
#             img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
#         sim = (img_feat @ text_feat.T).item()
#         if sim > best[1]:
#             best = (path, sim)
#     return best[0]

# # --- 4. Duyệt videos, trích frame & build dict mới ---
# output = {}
# frames_root = r"E:\Đồ án chuyên ngành\dataset\outputdata"

# for vid_path, info in video_data.items():
#     # Ví dụ: vid_path = ".../Violence/V_39.mp4"
#     cat = info["category"]
#     cap = info["caption"]
#     vid_name = os.path.splitext(os.path.basename(vid_path))[0]  # "V_39"

#     # Thư mục chứa frames
#     frame_folder = os.path.join(frames_root, cat, vid_name)
#     if not os.path.isdir(frame_folder):
#         print(f"⚠️ Không tìm thấy thư mục frames: {frame_folder}")
#         continue

#     best_frame = select_best_frame(frame_folder, cap)
#     if best_frame is None:
#         print(f"❌ Không chọn được frame cho {vid_name}")
#         continue

#     # Ghi record mới
#     output[best_frame] = {
#         "category": cat,
#         "caption": cap
#     }
#     print(f"✅ {vid_name} → {best_frame}")

# # --- 5. Lưu JSON kết quả ---
# with open("Filtered_Frames_Script.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=2)

# print("🎉 Hoàn thành! Kết quả nằm ở Filtered_Frames_Script.json")


import json
import os
import shutil
import torch
import clip
from PIL import Image

# --- 1. Load CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- 2. Load JSON script gốc ---
with open("Data_Script.json", "r", encoding="utf-8") as f:
    video_data = json.load(f)

# --- Thư mục chứa frames ban đầu và đích ---
frames_root   = r"E:\Đồ án chuyên ngành\dataset\outputdata"
filtered_root = r"E:\Đồ án chuyên ngành\dataset\filtered_frames_mapping_caption"

# --- 3. Hàm chọn frame tốt nhất với CLIP ---
def select_best_frame(frame_folder: str, caption: str):
    text_token = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    best = (None, -1.0)
    for fn in sorted(os.listdir(frame_folder)):
        path = os.path.join(frame_folder, fn)
        try:
            img = preprocess(Image.open(path)).unsqueeze(0).to(device)
        except:
            continue
        with torch.no_grad():
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).item()
        if sim > best[1]:
            best = (path, sim)
    return best[0]

# --- 4. Duyệt videos, chọn & copy frame, build dict mới ---
output = {}

for vid_path, info in video_data.items():
    cat      = info["category"]
    cap      = info["caption"]
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]  # ex: "V_39"

    # Thư mục chứa frames gốc
    src_folder = os.path.join(frames_root, cat, vid_name)
    if not os.path.isdir(src_folder):
        print(f"⚠️ Không tìm thấy frames: {src_folder}")
        continue

    best_frame = select_best_frame(src_folder, cap)
    if best_frame is None:
        print(f"❌ Không chọn được frame cho {vid_name}")
        continue

    # --- Copy file sang thư mục đích ---
    dst_folder = os.path.join(filtered_root, cat, vid_name)
    os.makedirs(dst_folder, exist_ok=True)
    dst_path = os.path.join(dst_folder, os.path.basename(best_frame))
    shutil.copy2(best_frame, dst_path)

    # --- Ghi record mới với đường dẫn frame đã copy ---
    output[dst_path] = {
        "category": cat,
        "caption":  cap
    }
    print(f"✅ {vid_name}: {best_frame} → {dst_path}")

# --- 5. Lưu JSON kết quả ---
out_json = os.path.join(filtered_root, "Filtered_Frames_Script_v2.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"🎉 Hoàn thành! Kết quả JSON tại: {out_json}")
