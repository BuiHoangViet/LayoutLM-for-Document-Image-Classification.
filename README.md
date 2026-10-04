# LayoutLM for Document Image Classification — SmartDoc AI (Hoá đơn bán lẻ)

Researched and fine-tuned LayoutLM for document image classification and information extraction. Built a custom
dataset by scraping document images, followed by data preprocessing, OCR, and bounding-box annotation with text
coordinates to preserve both textual and spatial information.

**SmartDoc AI** quét ảnh **hoá đơn bán lẻ** (siêu thị, cửa hàng, nhà sách, quán ăn...) chụp bằng điện thoại và
trích xuất **đầy đủ** thông tin — chạy **hoàn toàn offline**, không gửi dữ liệu ra ngoài:

- Cửa hàng, địa chỉ, điện thoại, mã số thuế, số hoá đơn, ngày giờ, thu ngân, hình thức thanh toán
- **Danh sách món**: tên hàng, số lượng, đơn vị tính, đơn giá, thành tiền, giảm giá
- Tổng thanh toán, giảm giá, khách trả, tiền thừa
- **Tự kiểm tra đủ món**: đối chiếu tổng các món với tổng hoá đơn, cảnh báo nếu lệch để sửa ngay trên bảng
- **Hỏi đáp**: *"tôi muốn lấy danh sách hàng đã mua"*, *"tổng tiền bao nhiêu"*, *"món nào đắt nhất"*...
- Xuất Excel, lưu kho hoá đơn, thống kê chi tiêu, REST API cho app mobile

## Kết quả

| Thành phần | Đánh giá | Kết quả |
|---|---|---|
| Kiểm tra "có phải hoá đơn" (LayoutLMv3) | 241 ảnh test (111 hoá đơn, 130 giấy tờ khác) | **100%** accuracy |
| Trích xuất — tổng tiền | 8 hoá đơn gán tay, chưa dùng khi phát triển | **100%** |
| Trích xuất — danh sách món | như trên | 86% món đúng thành tiền, 91% đúng số lượng |
| Hoá đơn lấy **đủ & đúng toàn bộ** | như trên | 6/8 |

Lỗi còn lại chủ yếu do OCR đọc nhầm chữ số (5↔6, 8↔0); khi đó hệ thống báo *"có thể thiếu món: lệch X đ"* để
người dùng sửa dòng tương ứng.

## Pipeline

```
ảnh → xoay đúng chiều → tăng chất lượng (CLAHE, khử nhiễu, làm nét)
    → OCR: EasyOCR (CRAFT) phát hiện vùng chữ + VietOCR đọc tiếng Việt
    → gom dòng (chịu được ảnh nghiêng / giấy cong)
    → LayoutLMv3: ảnh có phải hoá đơn bán lẻ không?
    → invoice.py: trích xuất trường + danh sách món → đối chiếu tổng → lưu SQLite
```

## Yêu cầu

- Python 3.10+ (đã chạy thử trên **Python 3.13**, Windows 11), khuyến nghị RAM ≥ 8 GB, ~3 GB dung lượng trống
- Chạy được trên **CPU** (mỗi ảnh ~20–40 giây); có GPU NVIDIA sẽ nhanh hơn nhiều
- [Git LFS](https://git-lfs.com) để tải file model

## Cài đặt

### 1. Clone repo (kèm model qua Git LFS)

```bash
git lfs install
```

```bash
git clone https://github.com/BuiHoangViet/LayoutLM-for-Document-Image-Classification..git smartdoc
```

```bash
cd smartdoc
```

Đảm bảo file model đã tải đủ (`model_output/model.safetensors` ~485 MB, không phải file text vài trăm byte):

```bash
git lfs pull
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

VietOCR phải cài riêng với `--no-deps` (bản trên PyPI ghim Pillow cũ, không build được trên Python mới):

```bash
pip install --no-deps vietocr
```

> Có GPU NVIDIA: cài PyTorch bản CUDA theo hướng dẫn tại https://pytorch.org trước khi chạy các lệnh trên.

### 3. Tải trọng số VietOCR (~150 MB, một lần)

Linux / macOS / Git Bash:

```bash
mkdir -p ~/.cache/vietocr && curl -L -o ~/.cache/vietocr/vgg_transformer.pth https://vocr.vn/data/vietocr/vgg_transformer.pth
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.cache\vietocr"; Invoke-WebRequest https://vocr.vn/data/vietocr/vgg_transformer.pth -OutFile "$HOME\.cache\vietocr\vgg_transformer.pth"
```

Không có file này hệ thống tự chuyển sang EasyOCR (kém chính xác hơn với tiếng Việt). Model phát hiện chữ của
EasyOCR (~95 MB) tự tải về `~/.EasyOCR/` ở lần chạy đầu tiên.

Trên Linux / Colab có thể chạy gộp bước 2 + 3:

```bash
bash setup.sh
```

## Chạy

### Giao diện web

```bash
streamlit run app.py
```

Mở http://localhost:8501. Lần quét đầu tiên mất thêm ~30 giây để nạp model.

| Trang | Chức năng |
|---|---|
| 🏠 Tổng quan | Số hoá đơn, tổng chi tiêu, chi tiêu theo cửa hàng, món mua nhiều nhất |
| 📷 Quét hoá đơn | Tải ảnh lên (nhiều ảnh cùng lúc) → xem & **sửa trực tiếp** thông tin, bảng món → lưu, tải Excel |
| 💬 Hỏi đáp | Hỏi bằng tiếng Việt trên từng hoá đơn, có sẵn các câu hỏi gợi ý |
| 📋 Kho hoá đơn | Tìm kiếm, sửa, xoá, xuất Excel toàn bộ (sheet hoá đơn + sheet chi tiết món) |
| ⚙️ Cài đặt | Ngưỡng nhận diện hoá đơn, kết quả test của model |

Chia sẻ ra Internet qua ngrok (cần biến môi trường `NGROK_AUTHTOKEN`):

```bash
python run_app.py --ngrok
```

### REST API (cho app mobile)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Tài liệu API tự sinh tại http://localhost:8000/docs.

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Kiểm tra server |
| POST | `/scan` | Gửi ảnh (form field `file`) → kiểm tra hoá đơn + trích xuất đầy đủ, lưu vào kho |
| GET | `/invoices?limit=30` | Danh sách hoá đơn đã lưu |
| GET | `/invoices/{id}` | Chi tiết 1 hoá đơn |
| POST | `/invoices/{id}/ask` | Hỏi đáp (form field `query`) |

Ví dụ:

```bash
curl -F "file=@hoa_don.jpg" http://localhost:8000/scan
```

```bash
curl -F "query=liệt kê danh sách hàng đã mua" http://localhost:8000/invoices/1/ask
```

Kết quả `/scan` thật trên 1 hoá đơn MC-OCR (rút gọn; trường không đọc được trả về `"Không tìm thấy"`):

```json
{
  "id": 1, "is_invoice": true, "invoice_prob": 0.9997,
  "invoice": {
    "store": "MINIMARTANAN", "date": "15/08/2020", "time": "Không tìm thấy",
    "items": [{"name": "Dầu simply 5L", "qty": 1, "unit": "chai", "unit_price": 214000, "amount": 214000, "discount": 0}],
    "total": 442000, "paid": 502000, "change": 60000,
    "checks": {"status": "ok", "message": "Đủ: tổng 6 món = 442.000 đ khớp tổng hoá đơn."}
  }
}
```

Dùng trực tiếp trong Python:

```python
from engine import scan_invoice

out = scan_invoice("hoa_don.jpg")
print(out["invoice"]["items"], out["invoice"]["total"], out["invoice"]["checks"]["message"])
```

## Đánh giá lại trích xuất

`eval/` chứa đáp án gán tay của 23 hoá đơn thuộc bộ dữ liệu MC-OCR 2021 (15 hoá đơn dùng khi phát triển luật,
8 hoá đơn giữ riêng để kiểm tra) cùng kết quả OCR đã cache, nên chấm điểm lại được ngay không cần ảnh:

```bash
python eval/evaluate_extraction.py --split holdout -v
```

Chạy lại OCR (thêm `--ocr`) cần ảnh gốc trong `dataset/HOA_DON/`.

## Train lại bộ kiểm tra hoá đơn

Model đã train sẵn trong `model_output/`. Muốn train lại cần 2 loại ảnh (không đưa lên git vì dung lượng lớn):

```
dataset/
├── HOA_DON/     ảnh hoá đơn bán lẻ (vd bộ MC-OCR 2021)
└── KHAC/        ảnh KHÔNG phải hoá đơn: hợp đồng, CV, giấy tờ, ảnh chụp thường...
```

```bash
python preprocess.py
```

```bash
python train.py
```

`preprocess.py` chạy OCR cho ảnh mới (dừng giữa chừng thì chạy lại sẽ tiếp tục). `train.py` lưu model vào
`model_output_new/` kèm `metrics.json`; app tự dùng bản model mới nhất. Thêm `--resume` để train tiếp khi bị ngắt.

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py`, `static/style.css` | Giao diện web Streamlit |
| `api.py` | REST API FastAPI |
| `engine.py` | `scan_invoice(ảnh)`: chạy toàn bộ pipeline |
| `invoice.py` | Trích xuất hoá đơn, kiểm tra đủ món, trả lời câu hỏi |
| `preprocess.py` | Xoay ảnh, tăng chất lượng, OCR (EasyOCR + VietOCR), gom dòng |
| `train.py`, `data_processing.py` | Fine-tune LayoutLMv3 2 lớp `HOA_DON` / `KHAC` |
| `config.py` | Đường dẫn, siêu tham số, ngưỡng |
| `model_output/` | Model LayoutLMv3 đã fine-tune (Git LFS) |
| `eval/` | Bộ đáp án + script chấm điểm trích xuất |
| `run_app.py`, `run_api.py`, `setup.sh` | Chạy kèm ngrok / cài môi trường |

Dữ liệu tạo ra khi chạy: `documents.db` (kho hoá đơn SQLite), `uploads/` (ảnh đã quét).

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SMARTDOC_OCR` | `vietocr` nếu có trọng số | `vietocr` hoặc `easyocr` |
| `SMARTDOC_VIETOCR_WEIGHTS` | `~/.cache/vietocr/vgg_transformer.pth` | Đường dẫn trọng số VietOCR |
| `SMARTDOC_CONF_THRESHOLD` | `0.6` | Ngưỡng xác suất "là hoá đơn" |
| `SMARTDOC_PROJECT_DIR` | thư mục chứa `config.py` | Đổi khi chạy trên Colab / Google Drive |
| `NGROK_AUTHTOKEN` | — | Token ngrok cho `--ngrok` |

## Xử lý sự cố

- **`model.safetensors` chỉ vài trăm byte / lỗi "Error while deserializing header"**: chưa tải file LFS →
  chạy `git lfs install` rồi `git lfs pull`.
- **`pip install vietocr` lỗi build Pillow**: dùng `pip install --no-deps vietocr` như hướng dẫn.
- **Quét chậm**: bình thường trên CPU (~20–40 giây/ảnh); lần đầu nạp model thêm ~30 giây.
- **Windows: train báo lỗi ghi `model.safetensors`**: train luôn lưu vào `model_output_new/` nên không xung đột
  với app đang chạy; nếu vẫn lỗi hãy tắt app rồi chạy lại.
- **Ảnh không phải hoá đơn**: app cảnh báo và không tự lưu; bấm *"Vẫn trích xuất như hoá đơn"* nếu muốn.
