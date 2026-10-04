"""
engine.py — SmartDoc AI (hoá đơn bán lẻ)

  scan_invoice(raw_image) → dict:
      is_invoice, invoice_prob   (LayoutLMv3: có phải hoá đơn không)
      invoice                    (invoice.extract_invoice: cửa hàng, ngày, danh sách món, tổng...)
      lines, tokens, ocr_text, image

  Model + OCR chỉ nạp 1 lần (lazy).
"""
import os
import threading

import torch

import config
import preprocess as P
import invoice as INV

_lock = threading.Lock()
_model = None
_processor = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_available() -> bool:
    return os.path.exists(os.path.join(config.active_model_dir(), "config.json"))


def load_model():
    global _model, _processor
    with _lock:
        if _model is None:
            from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
            src = config.active_model_dir()
            try:
                _processor = LayoutLMv3Processor.from_pretrained(src, apply_ocr=False)
            except Exception:
                _processor = LayoutLMv3Processor.from_pretrained(config.PRETRAINED_MODEL, apply_ocr=False)
            _model = LayoutLMv3ForSequenceClassification.from_pretrained(src).to(_device).eval()
            print(f"[engine] ✅ Model nạp từ {src} ({_device}) | nhãn: {_model.config.id2label}")
    return _model, _processor


def invoice_probability(image, words, boxes) -> float:
    """Xác suất ảnh là hoá đơn bán lẻ (nhãn HOA_DON)."""
    model, processor = load_model()
    if not words:
        words, boxes = ["[EMPTY]"], [[0, 0, 0, 0]]
    enc = processor(image, words, boxes=boxes, truncation=True,
                    max_length=config.MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu()
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    return float(probs[label2id["HOA_DON"]])


def scan_invoice(raw_image, check_invoice: bool = True) -> dict:
    """raw_image: bytes / đường dẫn / PIL / numpy."""
    doc = P.process_image(raw_image)
    words = [t["text"] for t in doc["tokens"]]
    boxes = [t["box"] for t in doc["tokens"]]
    prob = None
    if check_invoice and model_available():
        prob = invoice_probability(doc["image"], words, boxes)
    inv = INV.extract_invoice(doc["lines"], doc["tokens"])
    is_invoice = (prob is None or prob >= config.CONFIDENCE_THRESHOLD) and len(words) >= 3
    print(f"[engine] invoice_prob={prob} | {len(inv['items'])} món | total={inv['total']} | "
          f"{inv['checks']['status']}")
    return {
        "is_invoice": is_invoice,
        "invoice_prob": prob,
        "invoice": inv,
        "lines": doc["lines"],
        "tokens": doc["tokens"],
        "ocr_text": "\n".join(doc["lines"]),
        "image": doc["image"],
        "mode": doc["mode"],
    }
