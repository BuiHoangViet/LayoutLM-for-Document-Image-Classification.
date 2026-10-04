# ==============================================================================
# preprocess.py — Tiền xử lý ảnh + OCR dùng CHUNG cho train và inference
#
# Gộp từ image_preprocess_pipeline_v3.py, sửa các lỗi:
#   - Chọn chế độ crop theo đặc điểm ảnh (ảnh chụp / ảnh scan) thay vì theo
#     nhãn lớp → inference dùng đúng pipeline như lúc tạo dữ liệu train.
#   - Ảnh RGBA / P / L luôn được chuyển về RGB (lỗi "cannot write mode RGBA as JPEG").
#   - EasyOCR chỉ khởi tạo 1 lần, tự dùng GPU nếu có.
#   - Đường dẫn lấy từ config.py (không còn hard-code Colab).
#
# Chạy toàn bộ dataset (có resume):
#   python preprocess.py               # xử lý file chưa có trong metadata
#   python preprocess.py --only HOA_DON  # chỉ 1 thư mục
# ==============================================================================
import os
import io
import json
import argparse

import cv2
import numpy as np
from PIL import Image

import config

UPSCALE_MIN   = 1400
UPSCALE_SCALE = 2.0
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)
SHARPEN_AMOUNT = 1.4
SHARPEN_SIGMA  = 1.5
DENOISE_H      = 5
PAD_RATIO      = 0.010
PDF_DPI        = 200

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
PDF_EXTS = ('.pdf',)


# ==============================================================================
#  EasyOCR singleton
# ==============================================================================
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        print(f"[OCR] Khởi tạo EasyOCR (gpu={gpu})...")
        _reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=gpu, verbose=False)
    return _reader


# ==============================================================================
#  Đọc ảnh từ mọi kiểu đầu vào
# ==============================================================================
def to_pil_rgb(raw) -> Image.Image:
    """Nhận bytes / đường dẫn / numpy (RGB) / PIL → PIL RGB."""
    if isinstance(raw, (bytes, bytearray)):
        img = Image.open(io.BytesIO(raw))
    elif isinstance(raw, str):
        img = Image.open(raw)
    elif isinstance(raw, np.ndarray):
        img = Image.fromarray(raw)
    else:
        img = raw
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ==============================================================================
#  Crop
# ==============================================================================
def _pad_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad = max(4, int(max(h, w) * PAD_RATIO))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def _warp_min_area_rect(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    box = np.array(cv2.boxPoints(cv2.minAreaRect(contour)), dtype="float32")
    s, diff = box.sum(axis=1), np.diff(box, axis=1)
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(diff)], box[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype="float32")
    dst_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    dst_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if dst_w < 10 or dst_h < 10:
        return img
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _fallback_white_crop(img_bgr, gray, ih, iw):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(mask == 255) > 0.7 * mask.size:
        mask = cv2.bitwise_not(mask)
    k = max(15, int(min(ih, iw) * 0.03))
    k = k + (1 - k % 2)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=3)
    coords = cv2.findNonZero(closed)
    if coords is None:
        return img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(8, int(max(ih, iw) * PAD_RATIO * 2))
    return img_bgr[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]


def smart_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Cắt tờ giấy ra khỏi nền (dành cho ảnh chụp điện thoại)."""
    ih, iw = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lo, hi = int(max(0, 0.67 * median)), int(min(255, 1.33 * median))
    edges = cv2.Canny(blur, lo, hi)
    k = max(9, int(min(ih, iw) * 0.015))
    k = k + (1 - k % 2)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_white_crop(img_bgr, gray, ih, iw)
    valid = [c for c in contours if cv2.contourArea(c) > 0.05 * ih * iw] or \
            [max(contours, key=cv2.contourArea)]
    warped = _warp_min_area_rect(img_bgr, max(valid, key=cv2.contourArea))
    if warped.shape[0] * warped.shape[1] < 0.08 * ih * iw:
        return _fallback_white_crop(img_bgr, gray, ih, iw)
    return _pad_image(warped)


def crop_white_border(img_bgr: np.ndarray) -> np.ndarray:
    """Cắt lề trắng (dành cho ảnh scan / ảnh render từ file số)."""
    ih, iw = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 245, 255, cv2.THRESH_BINARY_INV)
    k = max(15, int(min(ih, iw) * 0.02))
    k = k + (1 - k % 2)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2)
    coords = cv2.findNonZero(closed)
    if coords is None:
        return img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(6, int(max(ih, iw) * PAD_RATIO * 3))
    return img_bgr[max(0, y - pad):min(ih, y + h + pad), max(0, x - pad):min(iw, x + w + pad)]


def is_scanned(img_bgr: np.ndarray) -> bool:
    """True nếu viền ảnh gần như trắng tinh → ảnh scan / render số.
    False → ảnh chụp có nền (bàn, tay, bóng...)."""
    h, w = img_bgr.shape[:2]
    b = max(4, int(min(h, w) * 0.03))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(),
                             gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return float(np.mean(border > 235)) > 0.85


def prepare_document(img_bgr: np.ndarray):
    """Bước crop chung. Trả về (ảnh đã crop, mode)."""
    if is_scanned(img_bgr):
        return crop_white_border(img_bgr), "scan"
    img = smart_crop(img_bgr)
    h, w = img.shape[:2]
    if w > h * 1.15:            # hoá đơn chụp ngang → xoay dọc
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img, "photo"


# ==============================================================================
#  Enhance (chỉ dùng cho OCR)
# ==============================================================================
def _deskew(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bin_ > 0))
    if len(coords) < 200:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if not (0.3 < abs(angle) < 10):
        return img_bgr
    H, W = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_image(img_bgr: np.ndarray, do_deskew: bool = False) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if min(h, w) < UPSCALE_MIN:
        img_bgr = cv2.resize(img_bgr, None, fx=UPSCALE_SCALE, fy=UPSCALE_SCALE,
                             interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID).apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=DENOISE_H, hColor=DENOISE_H,
                                              templateWindowSize=7, searchWindowSize=15)
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), SHARPEN_SIGMA)
    img_bgr = cv2.addWeighted(img_bgr, SHARPEN_AMOUNT, blurred, -(SHARPEN_AMOUNT - 1.0), 0)
    if do_deskew:
        img_bgr = _deskew(img_bgr)
    return img_bgr


# ==============================================================================
#  OCR
# ==============================================================================
VIETOCR_WEIGHTS = os.environ.get(
    "SMARTDOC_VIETOCR_WEIGHTS", os.path.join(os.path.expanduser("~"), ".cache", "vietocr", "vgg_transformer.pth"))
_vietocr = None


def ocr_engine() -> str:
    """'vietocr' (mặc định nếu có trọng số) hoặc 'easyocr'. Ghi đè bằng SMARTDOC_OCR."""
    eng = os.environ.get("SMARTDOC_OCR")
    if eng:
        return eng
    return "vietocr" if os.path.exists(VIETOCR_WEIGHTS) else "easyocr"


def get_vietocr():
    global _vietocr
    if _vietocr is None:
        import torch
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        name = "vgg_seq2seq" if "seq2seq" in os.path.basename(VIETOCR_WEIGHTS) else "vgg_transformer"
        cfg = Cfg.load_config_from_name(name)
        cfg["weights"] = VIETOCR_WEIGHTS
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["predictor"]["beamsearch"] = False
        cfg["cnn"]["pretrained"] = False          # trọng số đầy đủ nạp từ VIETOCR_WEIGHTS, không cần ImageNet
        print(f"[OCR] Khởi tạo VietOCR ({name}, {cfg['device']})...")
        _vietocr = Predictor(cfg)
    return _vietocr


def _run_vietocr(img_bgr: np.ndarray, conf_thr: float):
    """EasyOCR (CRAFT) phát hiện vùng chữ → VietOCR đọc từng vùng (tốt hơn hẳn với dấu tiếng Việt)."""
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    horiz, free = get_reader().detect(rgb, width_ths=0.5, add_margin=0.05)
    boxes = [(int(b[0]), int(b[2]), int(b[1]), int(b[3])) for b in horiz[0]]          # x1, y1, x2, y2
    for poly in free[0]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    crops, kept = [], []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
        kept.append((x1, y1, x2, y2))
    if not crops:
        return []
    texts, probs = get_vietocr().predict_batch(crops, return_prob=True)
    tokens = []
    for (x1, y1, x2, y2), text, prob in zip(kept, texts, probs):
        text = text.strip()
        if not text or prob < conf_thr:
            continue
        tokens.append({"text": text,
                       "box": [int(1000 * x1 / W), int(1000 * y1 / H), int(1000 * x2 / W), int(1000 * y2 / H)],
                       "conf": float(prob)})
    return tokens


def run_ocr(img_bgr: np.ndarray, conf_thr: float = config.OCR_CONF_THR, engine: str = None):
    """Trả về list token: {text, box (0-1000), conf}."""
    if (engine or ocr_engine()) == "vietocr":
        return _run_vietocr(img_bgr, conf_thr)
    H, W = img_bgr.shape[:2]
    results = get_reader().readtext(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                    detail=1, paragraph=False)
    tokens = []
    for polygon, text, conf in results:
        text = text.strip()
        if conf < conf_thr or not text:
            continue
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(W, int(max(xs))), min(H, int(max(ys)))
        if x2 <= x1 or y2 <= y1:
            continue
        tokens.append({
            "text": text,
            "box": [int(1000 * x1 / W), int(1000 * y1 / H),
                    int(1000 * x2 / W), int(1000 * y2 / H)],
            "conf": float(conf),
        })
    return tokens


def group_lines(tokens):
    """Gom token thành dòng, chịu được ảnh nghiêng / giấy cong.
    Duyệt token từ trái sang phải; mỗi token nối vào dòng có token CUỐI chồng lấn
    theo chiều dọc nhiều nhất (so với hàng xóm gần nhất chứ không so toạ độ y toàn cục),
    nên một dòng nghiêng vẫn được giữ liền, không bị trộn với dòng kế bên."""
    if not tokens:
        return []
    lines = []                                   # mỗi dòng: list token (đã theo thứ tự x)
    for t in sorted(tokens, key=lambda t: (t["box"][0], t["box"][1])):
        x1, y1, x2, y2 = t["box"]
        h = max(1, y2 - y1)
        best, best_ov = None, 0.0
        for line in lines:
            last = line[-1]
            lx1, ly1, lx2, ly2 = last["box"]
            if x1 < lx1:                         # chỉ nối về bên phải
                continue
            ov = min(y2, ly2) - max(y1, ly1)
            ov /= max(1, min(h, ly2 - ly1))
            # phạt nhẹ khoảng cách ngang lớn để ưu tiên hàng xóm gần
            ov -= 0.0005 * max(0, x1 - lx2)
            if ov > best_ov:
                best, best_ov = line, ov
        if best is not None and best_ov >= 0.45:
            best.append(t)
        else:
            lines.append([t])
    lines.sort(key=lambda l: sum((t["box"][1] + t["box"][3]) / 2 for t in l) / len(l))
    return [" ".join(t["text"] for t in line) for line in lines]


def fix_orientation(img_bgr: np.ndarray) -> np.ndarray:
    """Ảnh chụp xoay ngang (chữ chạy dọc) → xoay về đúng chiều.
    Chiều xoay được chọn bằng cách thử cả 2 hướng và lấy hướng OCR đọc tự tin hơn."""
    small = img_bgr.copy()
    scale = 960 / max(small.shape[:2])
    if scale < 1:
        small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reader = get_reader()
    horiz, free = reader.detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), width_ths=0.5)
    boxes = [(b[1] - b[0], b[3] - b[2]) for b in horiz[0]]
    boxes += [(max(p[0] for p in q) - min(p[0] for p in q), max(p[1] for p in q) - min(p[1] for p in q))
              for q in free[0]]
    boxes = [(w, h) for w, h in boxes if w > 5 and h > 5]
    if len(boxes) < 3:
        return img_bgr
    vertical = sum(1 for w, h in boxes if h > 1.5 * w)
    if vertical < 0.5 * len(boxes):
        return img_bgr

    def score(rot):
        res = reader.readtext(cv2.cvtColor(cv2.rotate(small, rot), cv2.COLOR_BGR2RGB), detail=1)
        return sum(conf * len(text) for _, text, conf in res)

    best = max((cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE), key=score)
    return cv2.rotate(img_bgr, best)


def process_image(raw, engine: str = None):
    """Pipeline cho 1 ảnh: xoay đúng chiều → tăng chất lượng → OCR trên TOÀN ảnh.
    (Không crop: crop tự động hay cắt mất phần đầu hoá đơn — tên cửa hàng, địa chỉ.)
    Trả về dict {image (PIL, đã xoay đúng chiều), tokens, lines, mode}."""
    bgr = fix_orientation(pil_to_bgr(to_pil_rgb(raw)))
    mode = "scan" if is_scanned(bgr) else "photo"
    enhanced = enhance_image(bgr, do_deskew=(mode == "scan"))
    tokens = run_ocr(enhanced, engine=engine)
    return {
        "image": bgr_to_pil(bgr),
        "tokens": tokens,
        "lines": group_lines(tokens),
        "mode": mode,
    }


# ==============================================================================
#  Chạy cho cả dataset (resume được)
# ==============================================================================
def _save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        backup = path + ".broken_backup"
        os.replace(path, backup)
        print(f"⚠️  JSON lỗi ({e}) → đã backup sang {backup}")
        return []


def list_images(class_dir: str):
    """Liệt kê ảnh trong 1 lớp. File .docx bị bỏ qua vì mỗi .docx trong
    resume_database đã có bản .png tương ứng."""
    out = []
    for root, _, files in os.walk(class_dir):
        for fname in sorted(files):
            if fname.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fname))
    return out


def preprocess_dataset(only=None, json_out=config.PROCESSED_JSON, data_dir=config.DATA_DIR):
    from tqdm import tqdm
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    records = _load_json(json_out)
    done = {os.path.normcase(config.local_path(r["original_path"])) for r in records}
    print(f"[Resume] {len(done)} file đã có OCR.")

    classes = sorted(c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c)))
    if only:
        classes = [c for c in classes if c in only]
    try:
        for cls in classes:
            files = [f for f in list_images(os.path.join(data_dir, cls))
                     if os.path.normcase(f) not in done]
            print(f"[{cls}] {len(files)} file cần OCR")
            for fpath in tqdm(files, desc=cls, unit="img"):
                try:
                    res = process_image(fpath)
                except Exception as e:
                    print(f"  [ERR] {fpath}: {e}")
                    continue
                records.append({
                    "original_path": os.path.relpath(fpath, config.PROJECT_DIR).replace("\\", "/"),
                    "words":  [t["text"] for t in res["tokens"]],
                    "bboxes": [t["box"] for t in res["tokens"]],
                    "mode":   res["mode"],
                    "label":  cls,
                })
                if len(records) % 25 == 0:
                    _save_json(json_out, records)
    except KeyboardInterrupt:
        print("\n[!] Dừng thủ công, đang lưu...")
    finally:
        _save_json(json_out, records)
        print(f"[Done] {len(records)} bản ghi → {json_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="Chỉ xử lý các lớp này")
    args = ap.parse_args()
    preprocess_dataset(only=args.only)
