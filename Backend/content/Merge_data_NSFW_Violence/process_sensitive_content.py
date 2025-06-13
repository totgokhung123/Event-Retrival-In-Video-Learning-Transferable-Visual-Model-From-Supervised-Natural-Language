import os
import json
import shutil
from pathlib import Path
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sensitive_content_process.log"),
        logging.StreamHandler()
    ]
)

# Đường dẫn đến các file input và thư mục output
train_json_path = "caption_NonViolence_train.json"
val_json_path = "caption_NonViolence_val.json"
output_dir = r"E:\Đồ án chuyên ngành\dataset\Data_Merge_2"

# Đường dẫn đến file JSON output
output_train_json_path = os.path.join(output_dir, "caption_NonViolence_train.json")
output_val_json_path = os.path.join(output_dir, "caption_NonViolence_val.json")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "NonViolence"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "NonViolence"), exist_ok=True)

# Hàm để xử lý và sao chép ảnh từ file JSON
def process_images(json_path, target_folder, output_json_path):
    logging.info(f"Đang đọc file {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Lỗi khi đọc file {json_path}: {str(e)}")
        return
    
    logging.info(f"Tổng số ảnh trong {json_path}: {len(data)}")
    
    # Dictionary để lưu dữ liệu mới
    new_data = {}
    
    # Đếm số lượng
    success_count = 0
    missing_count = 0
    error_count = 0
    
    # Danh sách các file không tìm thấy hoặc lỗi
    missing_files = []
    error_files = []
    
    # Xử lý từng ảnh
    for i, (img_path, img_info) in enumerate(data.items()):
        # Lấy phần mở rộng từ file gốc
        original_path = Path(img_path)
        file_extension = original_path.suffix
        
        # Tạo tên file mới theo dãy số tự nhiên (đảm bảo không trùng)
        folder_name = "NonViolence"  # Tên folder là "Sensitive"
        new_filename = f"NonViolence_{i+1:06d}{file_extension}"
        
        # Tạo đường dẫn mới
        new_relative_path = f"{target_folder}/{folder_name}/{new_filename}"
        new_absolute_path = os.path.join(output_dir, new_relative_path)
        
        # Sao chép file ảnh nếu tồn tại
        if os.path.exists(img_path):
            try:
                shutil.copy2(img_path, new_absolute_path)
                # Cập nhật đường dẫn trong dict mới
                new_data[new_relative_path] = img_info
                success_count += 1
                
                # In tiến độ mỗi 100 ảnh
                if (i + 1) % 100 == 0 or i == len(data) - 1:
                    logging.info(f"Đã xử lý {i+1}/{len(data)} ảnh cho {target_folder}")
            except Exception as e:
                error_message = f"Lỗi khi sao chép {img_path} -> {new_filename}: {str(e)}"
                logging.error(error_message)
                error_files.append((img_path, str(e)))
                error_count += 1
        else:
            missing_message = f"Không tìm thấy file: {img_path}"
            logging.warning(missing_message)
            missing_files.append(img_path)
            missing_count += 1
    
    # Lưu file JSON mới
    logging.info(f"Đang lưu file JSON {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    # Thống kê kết quả
    logging.info(f"Kết quả xử lý {json_path}:")
    logging.info(f"- Tổng số ảnh: {len(data)}")
    logging.info(f"- Số ảnh thành công: {success_count}")
    logging.info(f"- Số ảnh không tìm thấy: {missing_count}")
    logging.info(f"- Số ảnh lỗi khi sao chép: {error_count}")
    
    # Ghi log chi tiết các file bị lỗi
    if missing_files:
        logging.info(f"Danh sách file không tìm thấy (hiển thị tối đa 10 file):")
        for file in missing_files[:10]:
            logging.info(f"  - {file}")
        if len(missing_files) > 10:
            logging.info(f"  ... và {len(missing_files) - 10} file khác")
    
    return success_count, missing_count, error_count, len(new_data)

# Tạo file báo cáo
report_path = os.path.join(output_dir, "sensitive_content_report.txt")

# Xử lý tập train
logging.info("\n===== XỬ LÝ TẬP TRAIN =====")
train_success, train_missing, train_errors, train_output_count = process_images(
    train_json_path, "train", output_train_json_path
)

# Xử lý tập val
logging.info("\n===== XỬ LÝ TẬP VAL =====")
val_success, val_missing, val_errors, val_output_count = process_images(
    val_json_path, "val", output_val_json_path
)

# Tạo báo cáo chi tiết
with open(report_path, "w", encoding="utf-8") as f:
    f.write("BÁO CÁO XỬ LÝ DỮ LIỆU SENSITIVE CONTENT\n")
    f.write("=====================================\n\n")
    
    f.write("TẬP TRAIN\n")
    f.write("---------\n")
    f.write(f"File nguồn: {train_json_path}\n")
    f.write(f"Tổng số ảnh: {train_success + train_missing + train_errors}\n")
    f.write(f"Số ảnh thành công: {train_success}\n")
    f.write(f"Số ảnh không tìm thấy: {train_missing}\n")
    f.write(f"Số ảnh lỗi khi sao chép: {train_errors}\n")
    f.write(f"Số ảnh trong file output: {train_output_count}\n\n")
    
    f.write("TẬP VAL\n")
    f.write("-------\n")
    f.write(f"File nguồn: {val_json_path}\n")
    f.write(f"Tổng số ảnh: {val_success + val_missing + val_errors}\n")
    f.write(f"Số ảnh thành công: {val_success}\n")
    f.write(f"Số ảnh không tìm thấy: {val_missing}\n")
    f.write(f"Số ảnh lỗi khi sao chép: {val_errors}\n")
    f.write(f"Số ảnh trong file output: {val_output_count}\n\n")
    
    f.write("TỔNG KẾT\n")
    f.write("--------\n")
    f.write(f"Tổng số ảnh đã xử lý: {train_success + train_missing + train_errors + val_success + val_missing + val_errors}\n")
    f.write(f"Tổng số ảnh thành công: {train_success + val_success}\n")
    f.write(f"Tổng số ảnh trong file output: {train_output_count + val_output_count}\n")

logging.info(f"\nHoàn thành! Đã lưu:")
logging.info(f"- Tập train: {output_train_json_path} ({train_output_count} ảnh)")
logging.info(f"- Tập val: {output_val_json_path} ({val_output_count} ảnh)")
logging.info(f"- Báo cáo chi tiết: {report_path}")
logging.info(f"- Các ảnh đã được sao chép vào: {output_dir}")

# Kiểm tra sự chênh lệch giữa số ảnh thành công và số ảnh trong file output
if train_success != train_output_count or val_success != val_output_count:
    logging.warning(f"Cảnh báo: Có sự chênh lệch giữa số ảnh thành công và số ảnh trong file output!")
    logging.warning(f"Train: {train_success} ảnh thành công, {train_output_count} ảnh trong file output")
    logging.warning(f"Val: {val_success} ảnh thành công, {val_output_count} ảnh trong file output")
    logging.warning(f"Nguyên nhân có thể là do trùng tên file sau khi đổi tên") 