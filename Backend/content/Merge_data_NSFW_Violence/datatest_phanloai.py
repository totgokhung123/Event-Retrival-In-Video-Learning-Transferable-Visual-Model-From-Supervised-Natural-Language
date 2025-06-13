import os
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# ==== Cấu hình đường dẫn ====
images_dir         = Path(r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\rgb\images")
xml_dir            = Path(r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\rgb\xml")
violence_img_dir   = Path(r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\violence_images")
violence_xml_dir   = Path(r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\violence_xml")
report_file        = Path(r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\report.xlsx")

# Tạo các thư mục đích nếu chưa tồn tại
violence_img_dir.mkdir(parents=True, exist_ok=True)
violence_xml_dir.mkdir(parents=True, exist_ok=True)

# Danh sách để lưu báo cáo
rows = []

# Duyệt qua từng file XML
for xml_path in xml_dir.glob("*.xml"):
    img_name = xml_path.stem + ".jpg"  # điều chỉnh thành .png nếu cần
    img_path = images_dir / img_name

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Lấy danh sách tất cả <name> trong file
    names = [obj.findtext("name", default="").strip() for obj in root.findall("object")]

    # Nối các tên bằng dấu phẩy để làm giá trị cột label
    label_str = ", ".join(names) if names else ""

    # Kiểm tra xem có bất kỳ label nào khác "normal"
    is_violence = any(n.lower() != "normal" for n in names)

    # Nếu là violence, copy cả ảnh và xml sang thư mục tương ứng
    if is_violence:
        if img_path.exists():
            shutil.copy2(img_path, violence_img_dir / img_name)
        else:
            print(f"[WARN] Ảnh không tồn tại: {img_path}")
        shutil.copy2(xml_path, violence_xml_dir / xml_path.name)

    # Thêm thông tin vào báo cáo
    rows.append({
        "image_path": str(img_path),
        "type": "Violence" if is_violence else "NonViolence",
        "label": label_str
    })

# Tạo DataFrame và ghi ra Excel
df = pd.DataFrame(rows)
df.to_excel(report_file, index=False)

print("✅ Hoàn thành!")
print("Báo cáo:", report_file)
print("Thư mục ảnh Violence:", violence_img_dir)
print("Thư mục XML Violence:", violence_xml_dir)
