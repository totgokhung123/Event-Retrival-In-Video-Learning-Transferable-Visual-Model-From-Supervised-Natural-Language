# Setup for model finetune

Dự án này sử dụng mô hình LLaVA (Large Language and Vision Assistant) để tạo caption chi tiết cho bộ dataset của bạn, phân loại theo 2 nhóm "Violence" và "NonViolence".

## 1. Cài đặt

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài đặt LLaVA

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..
```

## 2. Tạo Caption

Chạy tập lệnh `llava_caption_generator.py` để tạo caption cho dataset của bạn:

```bash
python llava_caption_generator.py
```

Mặc định, script này sẽ:
- Xử lý các hình ảnh trong thư mục "E:/Đồ án chuyên ngành/dataset_RLVSD"
- Sử dụng 2 prompt khác nhau cho folder Violence và NonViolence
- Đối với mỗi thư mục con, chỉ chọn 3 hình đại diện để tiết kiệm thời gian
- Lưu kết quả vào file `llava_captions.json`

### Tùy chỉnh

Bạn có thể chỉnh sửa các biến sau trong file `llava_caption_generator.py`:
- `DATASET_ROOT`: Đường dẫn tới thư mục chứa 2 thư mục "Violence" và "NonViolence"
- `MODEL_PATH`: Đường dẫn hoặc tên model LLaVA (mặc định "liuhaotian/llava-v1.5-7b")
- `PROMPT_DICT`: Các prompt được sử dụng cho từng lớp

## 3. Phân tích Caption

Sau khi đã sinh caption, bạn có thể phân tích chúng bằng cách chạy:

```bash
python caption_analyzer.py
```

Script này sẽ:
- Đọc file `llava_captions.json`
- Tạo các biểu đồ phân tích (word cloud, phân phối từ, đánh giá cảm xúc)
- So sánh từ khóa phổ biến giữa 2 lớp
- Tạo báo cáo chi tiết về sự khác biệt giữa caption của hình ảnh Violence và NonViolence

Kết quả phân tích sẽ được lưu trong thư mục `caption_analysis/`.

## 4. So sánh với BLIP-2

Nếu muốn so sánh kết quả với mô hình BLIP-2 bạn đã sử dụng trước đó, có thể chạy:

```bash
python testing_caption_autogen.py
```

## 5. Lời khuyên

1. **Điều chỉnh prompt**: Prompt là yếu tố quan trọng nhất ảnh hưởng đến chất lượng caption. Bạn có thể cần điều chỉnh prompt để có được kết quả tốt nhất.

2. **Thử nghiệm model khác nhau**:
   - LLaVA-1.5-7B: Nhanh, phù hợp với máy có ít VRAM
   - LLaVA-1.5-13B: Chính xác hơn, nhưng cần GPU mạnh hơn

3. **Quantization**: Mã đã bật chế độ 4-bit quantization để tiết kiệm VRAM. Nếu bạn có GPU mạnh, bạn có thể tắt nó bằng cách đổi `load_4bit=False`.

4. **Xử lý nhiều hình ảnh**: Nếu một thư mục có nhiều hình, script sẽ chỉ chọn 3 hình đại diện. Bạn có thể thay đổi điều này nếu muốn xử lý tất cả.

## 6. Cấu trúc thư mục

```
Violence/
  ├── subfolder1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── subfolder2/
  │   └── ...
  └── ...
NonViolence/
  ├── subfolder1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── subfolder2/
  │   └── ...
  └── ...
```

## 7. Tối ưu Prompt

Với LLaVA, prompt đóng vai trò quan trọng trong việc định hướng mô hình sinh caption. Prompt tốt nên:

1. Mô tả rõ loại nội dung bạn đang tìm kiếm
2. Đưa ra hướng dẫn cụ thể về mức độ chi tiết
3. Tránh gợi ý quá nhiều, để mô hình mô tả khách quan

Ví dụ:
- Prompt cải tiến cho Violence: "This image may contain violent, abnormal, or sensitive content. Please describe specifically what actions are taking place, the setting, and any concerning behaviors visible. Focus on objective details."

- Prompt cải tiến cho NonViolence: "This image contains normal, everyday activities. Please describe in detail what people are doing, the setting, and the general atmosphere. Focus on objective details of the scene."

# So sánh các model truy vấn sự kiện nhạy cảm

Tool này cho phép so sánh hiệu suất của các model khác nhau trong việc truy vấn sự kiện nhạy cảm trên một tập test.

## Các model hỗ trợ

- **CLIP Fine-tuned**: Model CLIP đã được fine-tune của bạn
- **CLIP gốc**: Model CLIP gốc từ OpenAI
- **BLIP-2**: Model BLIP-2 từ Salesforce
- **ViT**: Model ViT base từ Google

## Cài đặt

```bash
pip install torch torchvision transformers pandas openpyxl matplotlib tqdm pillow clip
```

Đảm bảo đã cài CLIP từ GitHub:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Sử dụng

Chạy lệnh sau để so sánh các model:

```bash
python compare_models.py --test_file test_caption_image_translated.xlsx --finetuned_model_path /path/to/your/model.pt --output_dir results --use_finetuned --use_clip --use_blip2 --use_vit
```

### Các tham số:

- `--test_file`: Đường dẫn đến file excel test (bắt buộc)
- `--finetuned_model_path`: Đường dẫn đến model CLIP fine-tuned của bạn
- `--output_dir`: Thư mục để lưu kết quả (mặc định: `compare_results`)
- `--use_finetuned`: Sử dụng model CLIP fine-tuned
- `--use_clip`: Sử dụng model CLIP gốc
- `--use_blip2`: Sử dụng model BLIP-2
- `--use_vit`: Sử dụng model ViT
- `--batch_size`: Batch size khi xử lý frames (mặc định: 16)

Nếu bạn chỉ muốn so sánh một số model, chỉ cần chọn các flag tương ứng. Ví dụ, để so sánh CLIP fine-tuned và CLIP gốc:

```bash
python compare_models.py --test_file test_caption_image_translated.xlsx --finetuned_model_path /path/to/your/model.pt --use_finetuned --use_clip
```

## Cấu trúc file test Excel

File test yêu cầu có 3 cột:
- `folder`: Đường dẫn tới thư mục chứa các frame để test
- `caption`: Đoạn mô tả (text query) dùng để truy vấn
- `image`: Đường dẫn tới frame được coi là đáp án đúng (có thể có nhiều frame, phân cách bằng dấu `;`)

## Kết quả

Script sẽ tạo ra:
1. File JSON chứa tất cả metrics cho từng model
2. Biểu đồ so sánh các metrics chính (R@1, R@5, R@10, MRR)
3. Biểu đồ so sánh Median Rank và thời gian xử lý

## Metrics được tính toán

- **R@K**: Tỉ lệ truy vấn mà ground-truth nằm trong top K kết quả (K = 1, 5, 10)
- **MRR (Mean Reciprocal Rank)**: Trung bình của nghịch đảo rank của ground-truth
- **Median Rank**: Trung vị của rank ground-truth trong kết quả
- **Mean Rank**: Trung bình rank của ground-truth
- **P@K (Precision@K)**: Độ chính xác ở top K kết quả (K = 1, 5, 10)
- **Avg_Processing_Time**: Thời gian xử lý trung bình mỗi truy vấn 

# CLIP Finetune cho Truy Vấn Nội Dung NSFW

Đây là chương trình giúp finetune mô hình CLIP để đạt hiệu quả tốt hơn trong việc truy vấn hình ảnh-văn bản với bộ dataset NSFW.

## Cấu trúc dữ liệu yêu cầu

```
Data_Merge/
├── train/
│   ├── Violence/
│   │   └── [hình ảnh Violence]
│   └── Sensitive/
│       └── [hình ảnh Sensitive]
└── val/
    ├── Violence/
    │   └── [hình ảnh Violence]
    └── Sensitive/
        └── [hình ảnh Sensitive]
```

Các file caption (đặt trong cùng thư mục với mã):
- `caption_Violence_train.json`
- `caption_Violence_val.json`
- `caption_Sensitive_train.json`
- `caption_Sensitive_val.json`

Mỗi file caption có cấu trúc JSON dạng: 
```json
{
  "image_filename.jpg": "Mô tả văn bản",
  ...
}
```

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 1. Fine-tuning

### Chạy chương trình:

```bash
python train_clip_nsfw.py
```

### Cấu hình tham số:

Thay vì sử dụng command line arguments, tất cả các tham số được hardcode trực tiếp trong file `train_clip_nsfw.py`. Để thay đổi tham số, mở file và chỉnh sửa phần sau trong hàm `main()`:

```python
# Khai báo hardcode các tham số
args = Args()
args.data_dir = "/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge"
args.model_name = "ViT-B-32"
args.pretrained = "openai"
args.batch_size = 64
args.learning_rate = 2e-4
args.weight_decay = 0.2
args.num_epochs = 10
args.temperature = 0.07
args.use_amp = True
args.hard_negatives = True
args.num_workers = 4
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.output_dir = "./output"
```

### Các tham số chính:

| Tham số | Giá trị mặc định | Mô tả |
|---------|---------|-------|
| `data_dir` | `/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge` | Đường dẫn đến thư mục dữ liệu |
| `model_name` | `ViT-B-32` | Tên mô hình CLIP (ViT-B-16, ViT-B-32, ViT-L-14...) |
| `pretrained` | `openai` | Nguồn pretrained model |
| `batch_size` | `64` | Kích thước batch |
| `learning_rate` | `2e-4` | Tốc độ học |
| `weight_decay` | `0.2` | Hệ số weight decay |
| `num_epochs` | `10` | Số epoch huấn luyện |
| `temperature` | `0.07` | Hệ số nhiệt độ cho contrastive loss |
| `use_amp` | `True` | Sử dụng mixed precision training |
| `hard_negatives` | `True` | Sử dụng hard negative mining |

## 2. Đánh giá mô hình

Sau khi finetune, bạn có thể đánh giá hiệu quả của mô hình:

```bash
python evaluate_clip.py
```

### Cấu hình tham số đánh giá:

Tương tự như training, các tham số đánh giá được hardcode trong file `evaluate_clip.py`:

```python
# Khai báo hardcode các tham số
args = Args()
args.data_dir = "/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge"
args.model_name = "ViT-B-16"
args.pretrained = "openai"
args.checkpoint = "./output/best_model_epoch_5.pt"
args.batch_size = 32
args.save_analysis = True
args.test_queries = True
args.num_workers = 4
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.output_dir = "./evaluation"
```

### Các tham số đánh giá chính:

| Tham số | Giá trị mặc định | Mô tả |
|---------|---------|-------|
| `data_dir` | `/kaggle/input/data-merge-nsfw-rlvsd/Data_Merge` | Đường dẫn đến thư mục dữ liệu |
| `model_name` | `ViT-B-16` | Tên mô hình CLIP |
| `checkpoint` | `./output/best_model_epoch_5.pt` | Đường dẫn đến mô hình đã finetune |
| `save_analysis` | `True` | Lưu phân tích các ví dụ tệ nhất |
| `test_queries` | `True` | Chạy thử với các truy vấn cụ thể |

## 3. Sử dụng mô hình cho video

Sử dụng mô hình đã được finetune để phát hiện nội dung nhạy cảm trong video:

```bash
python infer_clip.py
```

### Cấu hình tham số video:

Để xử lý video của bạn, hãy mở file `infer_clip.py` và thay đổi đường dẫn video:

```python
# Khai báo hardcode các tham số
args = Args()
args.video = "path/to/your/video.mp4"  # Thay đổi đường dẫn này theo video của bạn
args.output_dir = "./results"
args.model_name = "ViT-B-16"
args.pretrained = "openai"
args.checkpoint = "./output/best_model_epoch_5.pt"
args.fps = 1.0
args.batch_size = 32
args.threshold = 30.0
args.top_k = 20
args.queries = [
    "Violence: A person being physically attacked",
    "Violence: A scene showing a gun",
    "Violence: Blood and injury",
    "Sensitive: Nude adult content",
    "Sensitive: Inappropriate touching",
    "Sensitive: Revealing clothing"
]
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.save_frames = True
```

### Các tham số inference chính:

| Tham số | Giá trị mặc định | Mô tả |
|---------|---------|-------|
| `video` | `"path/to/your/video.mp4"` | Đường dẫn đến video cần phân tích (cần thay đổi) |
| `checkpoint` | `./output/best_model_epoch_5.pt` | Đường dẫn đến mô hình đã finetune |
| `fps` | `1.0` | Số khung hình mỗi giây để trích xuất từ video |
| `threshold` | `30.0` | Ngưỡng tương đồng (0-100) để xác định nội dung phù hợp |
| `top_k` | `20` | Số kết quả tối đa trả về cho mỗi truy vấn |
| `save_frames` | `True` | Lưu riêng lẻ từng khung hình phù hợp |
| `queries` | `[...]` | Danh sách các truy vấn text để tìm kiếm trong video |

### Output của phân tích video:

Mỗi video sẽ được phân tích và tạo ra:

1. **Báo cáo tổng quan**: File text liệt kê các khung hình phù hợp với mỗi truy vấn
2. **Hình ảnh tổng hợp**: Hiển thị top-k khung hình phù hợp nhất cho mỗi truy vấn
3. **Khung hình riêng lẻ**: Lưu từng khung hình phù hợp với timestamp

## Chiến lược tối ưu

1. **Finetune toàn bộ model**: Không freeze bất kỳ layer nào để mô hình thích nghi tốt nhất với domain NSFW
2. **Batch size lớn**: Sử dụng batch size lớn (64) để tăng hiệu quả contrastive learning
3. **Hard negative mining**: Bật tính năng hard negatives để tập trung vào các cặp khó phân biệt
4. **Mixed precision**: Sử dụng mixed precision để tăng tốc độ training
5. **Đánh giá thường xuyên**: Mô hình sẽ tự động đánh giá trên tập validation sau mỗi epoch

## Demo sử dụng

Quy trình đầy đủ từ finetune đến inference:

```bash
# 1. Đầu tiên sửa các tham số trong các file theo nhu cầu

# 2. Finetune mô hình
python train_clip_nsfw.py

# 3. Đánh giá mô hình
python evaluate_clip.py

# 4. Phân tích video (đảm bảo đã sửa đường dẫn video)
python infer_clip.py
``` 
