# ============================================================
# train.py — Fine-tune LayoutLMv3 kiểm tra "có phải hoá đơn bán lẻ" (HOA_DON / KHAC)
#
#   python train.py                 # train từ microsoft/layoutlmv3-base
#   python train.py --resume        # tiếp tục từ checkpoint cuối epoch
#   python train.py --eval-only     # chỉ đánh giá model_output trên tập test
#
# Cải tiến so với bản cũ:
#   - Dữ liệu OCR sạch (dataset_processed.json) thay cho dataset.json bị lỗi OCR
#   - Ảnh lấy từ file gốc (không còn ảnh "annotated" có khung vẽ → rò rỉ nhãn)
#   - Chia train/val/test stratified, bỏ trùng lặp, cân bằng lớp
#   - Augmentation ảnh + văn bản, dynamic padding (nhanh hơn nhiều trên CPU)
#   - Chọn model theo macro-F1, xuất báo cáo + confusion matrix ra metrics.json
# ============================================================
import os
import json
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (LayoutLMv3ForSequenceClassification, LayoutLMv3Processor,
                          get_cosine_schedule_with_warmup)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import config
from data_processing import (load_records, split_records, build_image_cache,
                             DocumentDataset, Collator)

OUT_DIR = config.TRAIN_OUT_DIR
CKPT_DIR = os.path.join(OUT_DIR, "last_checkpoint")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, total_loss = [], [], 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        preds += out.logits.argmax(-1).cpu().tolist()
        labels += batch["labels"].cpu().tolist()
    model.train()
    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds, "labels": labels,
    }


def report(res, split):
    names = config.LABEL_NAMES
    rep = classification_report(res["labels"], res["preds"], labels=list(range(len(names))),
                                target_names=names, digits=4, zero_division=0)
    cm = confusion_matrix(res["labels"], res["preds"], labels=list(range(len(names))))
    log(f"=== {split}: acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f}\n{rep}\n"
        f"Confusion matrix (hàng = thật, cột = dự đoán) {names}:\n{cm}")
    return {"acc": res["acc"], "macro_f1": res["macro_f1"], "loss": res["loss"],
            "report": classification_report(res["labels"], res["preds"],
                                            labels=list(range(len(names))), target_names=names,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | threads={torch.get_num_threads()}")

    records = load_records()
    missing = [n for n in config.LABEL_NAMES if not any(r["label"] == n for r in records)]
    if missing:
        log(f"❌ Thiếu dữ liệu cho lớp {missing}. Cần ảnh hoá đơn trong dataset/HOA_DON và ảnh KHÔNG phải "
            "hoá đơn (giấy tờ, ảnh khác...) trong một thư mục khác, vd dataset/KHAC, rồi chạy "
            "`python preprocess.py` trước khi train. Model hiện tại trong model_output/ vẫn dùng bình thường.")
        return
    train_recs, val_recs, test_recs = split_records(records)
    build_image_cache(train_recs + val_recs + test_recs)

    processor = LayoutLMv3Processor.from_pretrained(config.PRETRAINED_MODEL, apply_ocr=False)
    collate = Collator(processor)
    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(DocumentDataset(train_recs, train=True), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate, num_workers=workers)
    val_loader = DataLoader(DocumentDataset(val_recs), batch_size=config.BATCH_SIZE * 2,
                            collate_fn=collate, num_workers=workers)
    test_loader = DataLoader(DocumentDataset(test_recs), batch_size=config.BATCH_SIZE * 2,
                             collate_fn=collate, num_workers=workers)

    if args.eval_only:
        model = LayoutLMv3ForSequenceClassification.from_pretrained(config.active_model_dir()).to(device)
        report(evaluate(model, test_loader, device), "TEST")
        return

    start_epoch, best_f1, bad_epochs, history = 0, -1.0, 0, []
    init_from = config.PRETRAINED_MODEL
    state_path = os.path.join(CKPT_DIR, "train_state.pt")
    if args.resume and os.path.exists(state_path):
        init_from = CKPT_DIR
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        init_from, num_labels=config.NUM_LABELS, id2label=config.ID2LABEL,
        label2id=config.LABEL2ID, ignore_mismatched_sizes=True).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader) // config.GRAD_ACCUM)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO),
                                                total_steps)
    if init_from == CKPT_DIR:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        start_epoch, best_f1 = st["epoch"], st["best_f1"]
        bad_epochs, history = st["bad_epochs"], st["history"]
        log(f"↻ Resume từ epoch {start_epoch} (best macro-F1={best_f1:.4f})")

    log(f"🚀 Train {len(train_recs)} mẫu | {args.epochs} epoch | batch {config.BATCH_SIZE}"
        f"x{config.GRAD_ACCUM} | max_len {config.MAX_LENGTH} | {steps_per_epoch} step/epoch")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0, run_loss = time.time(), 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / config.GRAD_ACCUM).backward()
            run_loss += loss.item()
            if (i + 1) % config.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(train_loader) - i - 1)
                log(f"  epoch {epoch + 1} batch {i + 1}/{len(train_loader)} "
                    f"loss={run_loss / (i + 1):.4f} | ETA epoch {eta / 60:.1f} phút")

        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": run_loss / len(train_loader),
                        "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]})
        log(f"Epoch {epoch + 1}/{args.epochs} | train_loss={run_loss / len(train_loader):.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f} | val_F1={val['macro_f1']:.4f} | "
            f"{(time.time() - t0) / 60:.1f} phút")

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            model.save_pretrained(OUT_DIR)
            processor.save_pretrained(OUT_DIR)
            log(f"  ✅ Lưu model tốt nhất (val macro-F1={best_f1:.4f}) → {OUT_DIR}")
        else:
            bad_epochs += 1
            log(f"  ⚠️ Không cải thiện ({bad_epochs}/{config.PATIENCE})")

        # Checkpoint để --resume nếu máy tắt giữa chừng
        model.save_pretrained(CKPT_DIR)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "best_f1": best_f1, "bad_epochs": bad_epochs,
                    "history": history}, state_path)
        if bad_epochs >= config.PATIENCE:
            log("🛑 Early stopping")
            break

    # Đánh giá model tốt nhất trên tập test (chưa từng thấy khi train / chọn model)
    best = LayoutLMv3ForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    metrics = {"val": report(evaluate(best, val_loader, device), "VAL"),
               "test": report(evaluate(best, test_loader, device), "TEST"),
               "history": history,
               "config": {"max_length": config.MAX_LENGTH, "batch_size": config.BATCH_SIZE,
                          "grad_accum": config.GRAD_ACCUM, "lr": config.LEARNING_RATE,
                          "epochs": args.epochs, "n_train": len(train_recs),
                          "n_val": len(val_recs), "n_test": len(test_recs)}}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"🏁 Xong. Test acc={metrics['test']['acc']:.4f} | macro-F1={metrics['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
