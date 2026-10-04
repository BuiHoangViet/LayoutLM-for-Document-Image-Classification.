# ============================================================
# data_processing.py — Nạp metadata OCR, chia tập, Dataset + augmentation
# ============================================================
import os
import json
import random
import hashlib
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

import config
import preprocess as P

IMAGE_CACHE_MAX_SIDE = 448   # model chỉ dùng 224x224, giữ 448 là đủ


# ------------------------------------------------------------------
# Nạp & làm sạch bản ghi
# ------------------------------------------------------------------
def _resolve(path):
    p = config.local_path(path)
    return p if os.path.isabs(p) else os.path.join(config.PROJECT_DIR, p)


def load_records(json_path=config.PROCESSED_JSON, min_words=3):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records, seen_text = [], set()
    dropped = Counter()
    for r in raw:
        source = r.get("label")
        if not source:
            dropped["label"] += 1
            continue
        src = _resolve(r["original_path"])
        if not os.path.exists(src):
            dropped["missing_file"] += 1
            continue
        words = [str(w) for w in r["words"]]
        boxes = [[max(0, min(1000, int(v))) for v in b] for b in r["bboxes"]]
        if len(words) < min_words:
            dropped["too_few_words"] += 1
            continue
        # Bỏ trùng lặp nội dung (một số CV trùng nhau) để val/test không bị rò rỉ
        key = hashlib.md5(" ".join(words[:200]).lower().encode("utf-8")).hexdigest()
        if key in seen_text:
            dropped["duplicate"] += 1
            continue
        seen_text.add(key)
        # Bản ghi cũ (pipeline v3) không có "mode": v3 crop theo lớp
        mode = r.get("mode") or ("photo" if source == "HOA_DON" else "scan")
        records.append({"src": src, "words": words, "bboxes": boxes, "source": source,
                        "label": config.source_to_label(source), "mode": mode})
    print(f"📥 Nạp {len(records)} bản ghi | bỏ: {dict(dropped)}")
    print(f"   Phân bố: {dict(Counter((r['label'], r['source']) for r in records))}")
    return records


def split_records(records, val=config.VAL_SPLIT, test=config.TEST_SPLIT,
                  seed=config.RANDOM_SEED, max_per_class=config.MAX_PER_CLASS):
    """Chia stratified theo lớp. Lớp quá lớn được lấy mẫu xuống để cân bằng."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    train, val_set, test_set = [], [], []
    for label, items in by.items():
        rng.shuffle(items)
        if max_per_class and len(items) > max_per_class:
            # Lấy đều từ mỗi nguồn (vd KHAC = hợp đồng + CV) để mẫu âm đa dạng
            by_src = defaultdict(list)
            for r in items:
                by_src[r["source"]].append(r)
            quota = max_per_class // len(by_src)
            items = [r for src_items in by_src.values() for r in src_items[:quota]]
            rng.shuffle(items)
        n = len(items)
        n_test, n_val = int(n * test), int(n * val)
        test_set += items[:n_test]
        val_set  += items[n_test:n_test + n_val]
        train    += items[n_test + n_val:]
    rng.shuffle(train)
    for name, s in (("train", train), ("val", val_set), ("test", test_set)):
        print(f"   {name:5s}: {len(s):5d} {dict(Counter(r['label'] for r in s))}")
    return train, val_set, test_set


# ------------------------------------------------------------------
# Cache ảnh đã crop (nhánh ảnh của LayoutLMv3)
# ------------------------------------------------------------------
def cache_path(src):
    h = hashlib.md5(os.path.normcase(src).encode("utf-8")).hexdigest()
    return os.path.join(config.IMAGE_CACHE_DIR, h + ".jpg")


def _build_one(args):
    src, mode = args
    dst = cache_path(src)
    if os.path.exists(dst):
        return True
    try:
        import cv2
        bgr = P.pil_to_bgr(P.to_pil_rgb(src))
        if mode == "photo":
            img = P.smart_crop(bgr)
            h, w = img.shape[:2]
            if w > h * 1.15:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            img = P.crop_white_border(bgr)
        pil = P.bgr_to_pil(img)
        pil.thumbnail((IMAGE_CACHE_MAX_SIDE, IMAGE_CACHE_MAX_SIDE))
        pil.save(dst, quality=90)
        return True
    except Exception as e:
        print(f"[cache] lỗi {src}: {e}")
        return False


def build_image_cache(records, workers=6):
    os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
    todo = [(r["src"], r["mode"]) for r in records if not os.path.exists(cache_path(r["src"]))]
    if not todo:
        return
    print(f"🖼️  Tạo cache ảnh cho {len(todo)} file...")
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    with ProcessPoolExecutor(workers) as ex:
        list(tqdm(ex.map(_build_one, todo, chunksize=16), total=len(todo)))


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------
def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Làm ảnh scan trông giống ảnh chụp (và ngược lại) để model không học
    'ảnh chụp = hoá đơn'. Biến đổi hình học nhỏ nên bbox vẫn khớp."""
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-4, 4), resample=Image.BICUBIC,
                         fillcolor=tuple(rng.randint(150, 255) for _ in range(3)))
    if rng.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.25))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.4:
        tint = Image.new("RGB", img.size, tuple(rng.randint(170, 255) for _ in range(3)))
        img = Image.blend(img, tint, rng.uniform(0.1, 0.35))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    if rng.random() < 0.2:
        img = ImageOps.grayscale(img).convert("RGB")
    if rng.random() < 0.15:
        img = ImageOps.autocontrast(img)
    return img


def augment_text(words, boxes, rng: random.Random):
    n = len(words)
    # 1) Chỉ giữ một đoạn liên tiếp (giả lập ảnh chụp thiếu / OCR thiếu)
    if n > 20 and rng.random() < 0.3:
        keep = rng.randint(int(n * 0.5), n)
        start = rng.randint(0, n - keep)
        words, boxes = words[start:start + keep], boxes[start:start + keep]
    # 2) Bỏ ngẫu nhiên ~10% token
    if rng.random() < 0.5:
        pairs = [(w, b) for w, b in zip(words, boxes) if rng.random() > 0.1]
        if pairs:
            words, boxes = map(list, zip(*pairs))
    # 3) Rung bbox nhẹ
    if rng.random() < 0.5:
        j = lambda v: max(0, min(1000, v + rng.randint(-8, 8)))
        boxes = [[j(b[0]), j(b[1]), max(j(b[0]), j(b[2])), max(j(b[1]), j(b[3]))] for b in boxes]
    return list(words), list(boxes)


# ------------------------------------------------------------------
# Dataset + collate
# ------------------------------------------------------------------
class DocumentDataset(Dataset):
    def __init__(self, records, train=False, seed=config.RANDOM_SEED):
        self.records = records
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            image = Image.open(cache_path(r["src"])).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        words, boxes = r["words"], r["bboxes"]
        if self.train:
            image = augment_image(image, self.rng)
            words, boxes = augment_text(words, boxes, self.rng)
        if not words:
            words, boxes = ["[EMPTY]"], [[0, 0, 0, 0]]
        return image, words, boxes, config.LABEL2ID[r["label"]]


class Collator:
    def __init__(self, processor, max_length=config.MAX_LENGTH):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        images, words, boxes, labels = zip(*batch)
        enc = self.processor(list(images), list(words), boxes=list(boxes),
                             truncation=True, max_length=self.max_length,
                             padding="longest", return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
