# ============================================================
# config.py — Cấu hình dùng chung (chạy được cả local lẫn Colab)
# ============================================================
# Mọi đường dẫn đều tính tương đối theo thư mục chứa file này.
# Có thể ghi đè bằng biến môi trường SMARTDOC_PROJECT_DIR
# (ví dụ trên Colab: /content/drive/MyDrive/Project_LayoutLM_Demo).
import os
import sys

# Console Windows mặc định cp1252 → print tiếng Việt bị UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_DIR = os.environ.get("SMARTDOC_PROJECT_DIR",
                             os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_DIR, "dataset")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "dataset_processed")
MODEL_DIR     = os.path.join(PROJECT_DIR, "model_output")
# train.py lưu vào thư mục riêng: trên Windows không ghi đè được model.safetensors
# khi app đang mở (file bị memory-map) → không bao giờ làm hỏng model đang chạy.
TRAIN_OUT_DIR = os.path.join(PROJECT_DIR, "model_output_new")
DB_PATH       = os.path.join(PROJECT_DIR, "documents.db")

# Metadata OCR (words + bboxes) do preprocess.py sinh ra
PROCESSED_JSON = os.path.join(PROCESSED_DIR, "dataset_processed.json")
# Cache ảnh đã crop (đầu vào nhánh ảnh của LayoutLMv3)
IMAGE_CACHE_DIR = os.path.join(PROCESSED_DIR, "image_cache")

# Các đường dẫn Colab cũ còn nằm trong metadata → map về PROJECT_DIR
LEGACY_PREFIXES = [
    "/content/drive/MyDrive/Project_LayoutLM_Demo",
    "/content/drive/My Drive/Project_LayoutLM_Demo",
]

PRETRAINED_MODEL = "microsoft/layoutlmv3-base"

# Phân loại 2 lớp: "có phải hoá đơn bán lẻ không".
# dataset/HOA_DON = hoá đơn; MỌI thư mục khác trong dataset/ (vd dataset/KHAC) = mẫu âm.
LABEL_NAMES = ["HOA_DON", "KHAC"]
LABEL2ID    = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL    = {i: name for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS  = len(LABEL_NAMES)


def source_to_label(folder: str) -> str:
    """Thư mục con trong dataset/ → nhãn train."""
    return "HOA_DON" if folder == "HOA_DON" else "KHAC"

# ── Train ────────────────────────────────────────────────────
MAX_LENGTH    = int(os.environ.get("SMARTDOC_MAX_LENGTH", 256))
BATCH_SIZE    = int(os.environ.get("SMARTDOC_BATCH_SIZE", 8))
GRAD_ACCUM    = int(os.environ.get("SMARTDOC_GRAD_ACCUM", 2))
EPOCHS        = int(os.environ.get("SMARTDOC_EPOCHS", 2))
LEARNING_RATE = 3e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
PATIENCE      = 2
# Giới hạn số mẫu mỗi lớp khi train (cân bằng lớp + giảm thời gian trên CPU)
MAX_PER_CLASS = int(os.environ.get("SMARTDOC_MAX_PER_CLASS", 1300))

VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
RANDOM_SEED = 42

# ── Inference ────────────────────────────────────────────────
# Dưới ngưỡng này → trả về UNKNOWN thay vì đoán bừa
CONFIDENCE_THRESHOLD = float(os.environ.get("SMARTDOC_CONF_THRESHOLD", 0.60))
OCR_LANGUAGES = ["vi", "en"]
OCR_CONF_THR  = 0.25

def active_model_dir() -> str:
    """Model dùng để chạy: bản train mới nhất (có metrics.json) trong TRAIN_OUT_DIR hoặc MODEL_DIR."""
    cands = [d for d in (TRAIN_OUT_DIR, MODEL_DIR) if os.path.exists(os.path.join(d, "metrics.json"))]
    if cands:
        return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "metrics.json")))
    return MODEL_DIR


def local_path(path: str) -> str:
    """Chuyển đường dẫn Colab cũ trong metadata thành đường dẫn hiện tại."""
    if not path:
        return path
    p = path.replace("\\", "/")
    for prefix in LEGACY_PREFIXES:
        if p.startswith(prefix):
            return os.path.join(PROJECT_DIR, p[len(prefix):].lstrip("/"))
    return path


if __name__ == "__main__":
    print(f"PROJECT_DIR : {PROJECT_DIR}")
    print(f"LABEL2ID    : {LABEL2ID}")
    print(f"MODEL_DIR   : {MODEL_DIR}")
