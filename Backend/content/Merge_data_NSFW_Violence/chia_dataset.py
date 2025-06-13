import os
import json
import random
import shutil
from pathlib import Path
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_split.log"),
        logging.StreamHandler()
    ]
)

# Thiết lập seed để đảm bảo tính lặp lại
random.seed(42)

# Đường dẫn đến file input và thư mục output
input_json_path = "Violence_data2.json"
output_dir = r"E:\Đồ án chuyên ngành\dataset\Data_Merge_2"
train_json_path = os.path.join(output_dir, "caption_Violence_train.json")
val_json_path = os.path.join(output_dir, "caption_Violence_val.json")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "Violence"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "Violence"), exist_ok=True)

# Đọc file JSON input
logging.info(f"Đang đọc file {input_json_path}...")
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lọc chỉ lấy dữ liệu có category là "Violence"
violence_data = {}
for key, value in data.items():
    if value.get("category") == "Violence":
        violence_data[key] = value

# Lấy danh sách các khóa (đường dẫn ảnh) chỉ của dữ liệu Violence
all_keys = list(violence_data.keys())
total_images = len(all_keys)
logging.info(f"Tổng số ảnh Violence: {total_images}")

# Xáo trộn danh sách để đảm bảo tính ngẫu nhiên
random.shuffle(all_keys)

# Chia thành tập train (80%) và val (20%)
split_index = int(0.8 * total_images)
train_keys = all_keys[:split_index]
val_keys = all_keys[split_index:]

logging.info(f"Số ảnh trong tập train: {len(train_keys)}")
logging.info(f"Số ảnh trong tập val: {len(val_keys)}")

# Tạo dict cho tập train và val
train_data = {}
val_data = {}

# Hàm để sao chép ảnh và cập nhật đường dẫn trong JSON
def process_images(keys, target_dict, subset_name):
    success_count = 0
    missing_files = []
    error_files = []
    
    # Đánh số từ 1 cho mỗi tập
    for i, key in enumerate(keys):
        # Lấy thông tin ảnh
        image_info = violence_data[key]
        
        # Lấy phần mở rộng từ file gốc
        original_path = Path(key)
        file_extension = original_path.suffix
        
        # Tạo tên file mới theo dãy số tự nhiên (đảm bảo không trùng)
        new_filename = f"violence_{subset_name}_{i+1:04d}{file_extension}"
        
        # Tạo đường dẫn mới
        new_relative_path = f"{subset_name}/Violence/{new_filename}"
        new_absolute_path = os.path.join(output_dir, new_relative_path)
        
        # Sao chép file ảnh nếu tồn tại
        if os.path.exists(key):
            try:
                shutil.copy2(key, new_absolute_path)
                # Cập nhật đường dẫn trong dict
                target_dict[new_relative_path] = image_info
                success_count += 1
                
                if (i + 1) % 100 == 0 or i == len(keys) - 1:
                    logging.info(f"Đã xử lý {i+1}/{len(keys)} ảnh cho tập {subset_name}")
            except Exception as e:
                error_message = f"Lỗi khi sao chép {key} -> {new_filename}: {str(e)}"
                logging.error(error_message)
                error_files.append((key, str(e)))
        else:
            missing_message = f"Không tìm thấy file: {key}"
            logging.warning(missing_message)
            missing_files.append(key)
    
    # Thống kê kết quả
    logging.info(f"Tập {subset_name} - Tổng số ảnh: {len(keys)}")
    logging.info(f"Tập {subset_name} - Số ảnh thành công: {success_count}")
    logging.info(f"Tập {subset_name} - Số ảnh không tìm thấy: {len(missing_files)}")
    logging.info(f"Tập {subset_name} - Số ảnh lỗi khi sao chép: {len(error_files)}")
    
    # Ghi log chi tiết các file bị lỗi
    if missing_files:
        logging.info("Danh sách file không tìm thấy:")
        for file in missing_files[:10]:  # Chỉ hiển thị 10 file đầu tiên để tránh log quá dài
            logging.info(f"  - {file}")
        if len(missing_files) > 10:
            logging.info(f"  ... và {len(missing_files) - 10} file khác")
    
    return success_count, missing_files, error_files

# Xử lý ảnh cho tập train và val
logging.info("\nĐang xử lý tập train...")
train_success, train_missing, train_errors = process_images(train_keys, train_data, "train")

logging.info("\nĐang xử lý tập val...")
val_success, val_missing, val_errors = process_images(val_keys, val_data, "val")

# Lưu file JSON cho tập train và val
logging.info("\nĐang lưu file JSON...")
with open(train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

# Tạo file ánh xạ giữa tên file cũ và tên file mới
mapping_path = os.path.join(output_dir, "filename_mapping.json")
filename_mapping = {}

# Xây dựng ánh xạ từ các dict train và val
for new_path, info in train_data.items():
    original_path = next((k for k, v in violence_data.items() if v == info), None)
    if original_path:
        filename_mapping[os.path.basename(original_path)] = os.path.basename(new_path)

for new_path, info in val_data.items():
    original_path = next((k for k, v in violence_data.items() if v == info), None)
    if original_path:
        filename_mapping[os.path.basename(original_path)] = os.path.basename(new_path)

# Lưu file ánh xạ
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(filename_mapping, f, ensure_ascii=False, indent=2)

# Tạo file báo cáo chi tiết
report_path = os.path.join(output_dir, "dataset_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("BÁO CÁO CHIA DATASET\n")
    f.write("===================\n\n")
    f.write(f"Tổng số ảnh Violence: {total_images}\n")
    f.write(f"Số ảnh phân chia cho train: {len(train_keys)}\n")
    f.write(f"Số ảnh phân chia cho val: {len(val_keys)}\n\n")
    
    f.write("KẾT QUẢ XỬ LÝ\n")
    f.write("============\n\n")
    f.write(f"Tập train - Thành công: {train_success}, Không tìm thấy: {len(train_missing)}, Lỗi: {len(train_errors)}\n")
    f.write(f"Tập val - Thành công: {val_success}, Không tìm thấy: {len(val_missing)}, Lỗi: {len(val_errors)}\n\n")
    
    if train_missing or val_missing:
        f.write("DANH SÁCH FILE KHÔNG TÌM THẤY\n")
        f.write("===========================\n\n")
        for file in train_missing + val_missing:
            f.write(f"{file}\n")

logging.info(f"\nHoàn thành! Đã lưu:")
logging.info(f"- Tập train: {train_json_path} ({len(train_data)} ảnh)")
logging.info(f"- Tập val: {val_json_path} ({len(val_data)} ảnh)")
logging.info(f"- Ánh xạ tên file: {mapping_path}")
logging.info(f"- Báo cáo chi tiết: {report_path}")
logging.info(f"- Các ảnh đã được sao chép vào: {output_dir}")

# Kiểm tra sự chênh lệch giữa số ảnh phân chia và số ảnh thực tế lưu được
if len(train_data) != len(train_keys) or len(val_data) != len(val_keys):
    logging.warning(f"Cảnh báo: Có sự chênh lệch giữa số ảnh phân chia và số ảnh thực tế lưu được!")
    logging.warning(f"Train: {len(train_keys)} ảnh phân chia, {len(train_data)} ảnh lưu được")
    logging.warning(f"Val: {len(val_keys)} ảnh phân chia, {len(val_data)} ảnh lưu được") 