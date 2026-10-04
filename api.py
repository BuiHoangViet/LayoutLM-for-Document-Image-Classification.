"""
api.py — REST API hoá đơn cho app mobile (offline)
  GET  /health
  POST /scan                     (file ảnh)   → kiểm tra hoá đơn + trích xuất đầy đủ, lưu kho
  GET  /invoices?limit=30                     → danh sách hoá đơn đã lưu
  GET  /invoices/{id}                         → chi tiết 1 hoá đơn
  POST /invoices/{id}/ask        (query)      → hỏi đáp, vd "liệt kê danh sách hàng đã mua"

Chạy: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
import invoice as INV
from engine import scan_invoice

app = FastAPI(title="SmartDoc AI — Invoice API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR = os.path.join(config.PROJECT_DIR, "uploads")


def _db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')), store TEXT, inv_date TEXT, total INTEGER,
        n_items INTEGER, status TEXT, invoice_prob REAL, edited INTEGER DEFAULT 0, data_json TEXT,
        ocr_text TEXT, image_path TEXT)""")
    return conn


def _summary(r):
    return {"id": r["id"], "filename": r["filename"], "created_at": r["created_at"], "store": r["store"],
            "date": r["inv_date"], "total": r["total"], "n_items": r["n_items"], "status": r["status"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(file: UploadFile = File(...), save: bool = True):
    try:
        data = await file.read()
        out = scan_invoice(data)
        inv = out["invoice"]
        inv_id = None
        if save and out["is_invoice"]:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            img_path = os.path.join(UPLOAD_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{hashlib.md5(data).hexdigest()[:12]}.jpg")
            out["image"].save(img_path, quality=92)
            with _db() as conn:
                cur = conn.execute(
                    "INSERT INTO invoices (filename, store, inv_date, total, n_items, status, invoice_prob, data_json,"
                    " ocr_text, image_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (file.filename, inv["store"], inv["date"], inv["total"], len(inv["items"]),
                     inv["checks"]["status"], out["invoice_prob"], json.dumps(inv, ensure_ascii=False),
                     out["ocr_text"], img_path))
                inv_id = cur.lastrowid
        return {"id": inv_id, "is_invoice": out["is_invoice"], "invoice_prob": out["invoice_prob"], "invoice": inv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/invoices")
def list_invoices(limit: int = 30):
    rows = _db().execute("SELECT * FROM invoices ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_summary(r) for r in rows]


@app.get("/invoices/{inv_id}")
def get_invoice(inv_id: int):
    r = _db().execute("SELECT * FROM invoices WHERE id=?", (inv_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hoá đơn")
    return {**_summary(r), "invoice": json.loads(r["data_json"])}


@app.post("/invoices/{inv_id}/ask")
def ask(inv_id: int, query: str = Form(...)):
    r = _db().execute("SELECT data_json FROM invoices WHERE id=?", (inv_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hoá đơn")
    return INV.answer_query(query, json.loads(r["data_json"]))
