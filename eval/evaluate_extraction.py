"""
Đánh giá trích xuất hoá đơn trên bộ đáp án gán tay (eval/receipts_gt.json).

  python eval/evaluate_extraction.py --ocr        # chạy OCR (lưu cache) rồi chấm điểm
  python eval/evaluate_extraction.py --engine easyocr   # so sánh engine OCR
  python eval/evaluate_extraction.py              # chỉ chấm điểm lại từ cache OCR (nhanh)
  python eval/evaluate_extraction.py --verbose    # in chi tiết từng hoá đơn

Chỉ số:
  total / paid / change / date / time : đúng tuyệt đối (tiền sai số ≤ 1đ)
  items recall    : % món trong đáp án được tìm thấy (khớp theo thành tiền + tên gần đúng)
  items precision : % món trích xuất ra là món thật
  name similarity : độ giống tên món (0-1) trên các món khớp
  qty accuracy    : % món khớp có số lượng đúng
  complete bills  : % hoá đơn lấy ĐỦ & ĐÚNG toàn bộ món + tổng tiền
"""
import os
import sys
import json
import time
import argparse
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import invoice  # noqa: E402

GT_PATH = os.path.join(ROOT, "eval", "receipts_gt.json")


def cache_path(engine, split="dev"):
    return os.path.join(ROOT, "eval", f"ocr_cache_{engine}{'' if split == 'dev' else '_' + split}.json")


def run_ocr(gt, engine, split="dev"):
    import preprocess as P
    cache = {}
    for g in gt:
        t = time.time()
        res = P.process_image(os.path.join(config.DATA_DIR, "HOA_DON", g["file"]), engine=engine)
        cache[g["file"]] = {"tokens": res["tokens"], "lines": res["lines"], "mode": res["mode"]}
        print(f"  OCR {g['file']}: {len(res['tokens'])} token, {time.time() - t:.1f}s", flush=True)
    with open(cache_path(engine, split), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    return cache


def norm(s):
    return invoice.strip_accents(str(s or "")).replace(" ", "")


def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def money_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= 1


def match_items(gt_items, pred_items):
    """Ghép món dự đoán ↔ đáp án: ưu tiên cùng thành tiền, rồi tên giống nhất."""
    used, pairs = set(), []
    for gi, g in enumerate(gt_items):
        best, best_score = None, 0.0
        for pi, p in enumerate(pred_items):
            if pi in used:
                continue
            s = sim(g[0], p["name"])
            score = s + (1.0 if money_eq(g[3], p["amount"]) else 0.0)
            if score > best_score and (money_eq(g[3], p["amount"]) or s >= 0.6):
                best, best_score = pi, score
        if best is not None:
            used.add(best)
            pairs.append((gi, best))
    return pairs


def evaluate(gt, cache, verbose=False):
    agg = {k: [0, 0] for k in ("total", "paid", "change", "date", "time")}
    n_gt_items = n_pred_items = n_match = n_amount_ok = n_qty_ok = 0
    name_sims, complete = [], 0
    for g in gt:
        c = cache[g["file"]]
        inv = invoice.extract_invoice(c["lines"], c["tokens"])
        for k in ("total", "paid", "change"):
            if g[k] is not None:
                agg[k][1] += 1
                agg[k][0] += money_eq(g[k], inv.get(k))
        for k in ("date", "time"):
            if g[k] is not None:
                agg[k][1] += 1
                agg[k][0] += (inv.get(k) or "")[:len(g[k])] == g[k]
        items = inv["items"]
        pairs = match_items(g["items"], items)
        n_gt_items += len(g["items"])
        n_pred_items += len(items)
        n_match += len(pairs)
        ok_items = 0
        for gi, pi in pairs:
            gname, gqty, _, gamt = g["items"][gi]
            p = items[pi]
            name_sims.append(sim(gname, p["name"]))
            a_ok = money_eq(gamt, p["amount"])
            q_ok = p["qty"] is not None and abs(float(p["qty"]) - float(gqty)) < 1e-3
            n_amount_ok += a_ok
            n_qty_ok += q_ok
            ok_items += a_ok
        is_complete = (ok_items == len(g["items"]) == len(items)
                       and (g["total"] is None or money_eq(g["total"], inv.get("total"))))
        complete += is_complete
        if verbose:
            print(f"\n=== {g['file']}  {'✅ ĐỦ' if is_complete else '❌'}")
            print(f"  total {inv.get('total')} (gt {g['total']}) | paid {inv.get('paid')} (gt {g['paid']}) | "
                  f"change {inv.get('change')} (gt {g['change']}) | {inv.get('date')} {inv.get('time')} | "
                  f"store: {inv.get('store')}")
            for p in items:
                print(f"   pred: {p['name'][:45]:45s} qty={p['qty']} unit={p['unit_price']} amt={p['amount']}")
            missed = [g['items'][i] for i in range(len(g['items'])) if i not in {gi for gi, _ in pairs}]
            for m in missed:
                print(f"   MISS: {m}")
    pct = lambda a, b: f"{100 * a / b:5.1f}% ({a}/{b})" if b else "  n/a"
    print("\n================ KẾT QUẢ ================")
    for k, (a, b) in agg.items():
        print(f"{k:16s}: {pct(a, b)}")
    print(f"{'items recall':16s}: {pct(n_amount_ok, n_gt_items)}   (khớp đúng thành tiền)")
    print(f"{'items precision':16s}: {pct(n_amount_ok, n_pred_items)}")
    print(f"{'qty accuracy':16s}: {pct(n_qty_ok, n_match)}")
    print(f"{'name similarity':16s}: {sum(name_sims) / max(1, len(name_sims)):.3f}")
    print(f"{'complete bills':16s}: {pct(complete, len(gt))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true", help="chạy lại OCR thay vì dùng cache")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--engine", default="vietocr", choices=["vietocr", "easyocr"])
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"],
                    help="dev = 15 hoá đơn dùng để tinh chỉnh luật; holdout = 8 hoá đơn chưa từng dùng")
    args = ap.parse_args()
    gt_path = GT_PATH if args.split == "dev" else os.path.join(ROOT, "eval", "receipts_holdout.json")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    if args.ocr or not os.path.exists(cache_path(args.engine, args.split)):
        cache = run_ocr(gt, args.engine, args.split)
    else:
        with open(cache_path(args.engine, args.split), encoding="utf-8") as f:
            cache = json.load(f)
    evaluate(gt, cache, args.verbose)
