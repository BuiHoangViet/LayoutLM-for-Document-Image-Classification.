"""
SmartDoc AI — Quét & trích xuất HOÁ ĐƠN BÁN LẺ (chạy offline)
  • Kiểm tra ảnh có phải hoá đơn (LayoutLMv3)
  • OCR tiếng Việt (EasyOCR phát hiện + VietOCR nhận dạng)
  • Trích xuất đầy đủ: cửa hàng, địa chỉ, ngày giờ, danh sách món, tổng tiền, khách trả, tiền thừa...
  • Tự kiểm tra ĐỦ (tổng các món = tổng hoá đơn), sửa trực tiếp trên bảng, xuất Excel
  • Hỏi đáp: "liệt kê danh sách hàng đã mua", "tổng tiền bao nhiêu"...

Chạy: streamlit run app.py
"""
import io
import os
import json
import sqlite3
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

import config
import invoice as INV

UPLOAD_DIR = os.path.join(config.PROJECT_DIR, "uploads")
PAGES = ['🏠 Tổng quan', '📷 Quét hoá đơn', '💬 Hỏi đáp', '📋 Kho hoá đơn', '⚙️ Cài đặt']
QUICK_PROMPTS = ["Liệt kê danh sách hàng đã mua", "Tổng tiền phải trả và ngày mua",
                 "Món nào đắt nhất?", "Hoá đơn có đủ món không?", "Tên cửa hàng, địa chỉ, số điện thoại"]
STATUS_BADGE = {"ok": ("✅ Đủ món", "b-hop"), "missing": ("⚠️ Có thể thiếu món", "b-unk"),
                "excess": ("⚠️ Thừa / đọc nhầm", "b-unk"), "unknown": ("❔ Chưa đối chiếu được", "b-ho")}
ITEM_COLS = ["Tên hàng", "SL", "ĐVT", "Đơn giá", "Thành tiền", "Giảm giá"]
HEADER_FIELDS = [("store", "Cửa hàng"), ("address", "Địa chỉ"), ("phone", "Điện thoại"),
                 ("invoice_no", "Số HĐ / GD"), ("date", "Ngày"), ("time", "Giờ"), ("cashier", "Thu ngân"),
                 ("payment_method", "Thanh toán"), ("tax_code", "Mã số thuế")]
MONEY_FIELDS = [("total", "Tổng thanh toán"), ("discount", "Giảm giá"), ("paid", "Khách trả"),
                ("change", "Tiền thừa")]


# ══════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT,
            created_at   TEXT DEFAULT (datetime('now','localtime')),
            store        TEXT,
            inv_date     TEXT,
            total        INTEGER,
            n_items      INTEGER,
            status       TEXT,
            invoice_prob REAL,
            edited       INTEGER DEFAULT 0,
            data_json    TEXT,
            ocr_text     TEXT,
            image_path   TEXT
        )""")
    conn.commit()
    return conn


def db_save(inv, filename=None, prob=None, ocr_text=None, image_path=None, inv_id=None, edited=False):
    vals = (inv.get("store"), inv.get("date"), inv.get("total"), len(inv["items"]), inv["checks"]["status"],
            json.dumps(inv, ensure_ascii=False))
    with get_db() as conn:
        if inv_id:
            conn.execute("UPDATE invoices SET store=?, inv_date=?, total=?, n_items=?, status=?, data_json=?,"
                         " edited=? WHERE id=?", vals + (int(edited), inv_id))
            return inv_id
        cur = conn.execute("INSERT INTO invoices (store, inv_date, total, n_items, status, data_json, filename,"
                           " invoice_prob, ocr_text, image_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                           vals + (filename, prob, ocr_text, image_path))
        return cur.lastrowid


def db_get(inv_id):
    r = get_db().execute("SELECT * FROM invoices WHERE id=?", (inv_id,)).fetchone()
    return (r, json.loads(r["data_json"])) if r else (None, None)


def db_all(search=None, limit=None):
    q, p = "SELECT * FROM invoices WHERE 1=1", []
    if search:
        q += " AND (store LIKE ? OR data_json LIKE ? OR filename LIKE ?)"
        p += [f"%{search}%"] * 3
    q += " ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
    return get_db().execute(q, p).fetchall()


def db_delete(inv_id):
    with get_db() as conn:
        r = conn.execute("SELECT image_path FROM invoices WHERE id=?", (inv_id,)).fetchone()
        conn.execute("DELETE FROM invoices WHERE id=?", (inv_id,))
    if r and r["image_path"] and os.path.exists(r["image_path"]):
        os.remove(r["image_path"])


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Đang nạp model + OCR (lần đầu ~30 giây)...")
def get_engine():
    import engine
    if engine.model_available():
        engine.load_model()
    import preprocess
    preprocess.get_reader()
    if preprocess.ocr_engine() == "vietocr":
        preprocess.get_vietocr()
    return engine


# ══════════════════════════════════════════════════════════════════════
# Chuyển đổi hoá đơn ↔ bảng
# ══════════════════════════════════════════════════════════════════════
def items_df(inv):
    df = pd.DataFrame([{"Tên hàng": it["name"], "SL": it["qty"], "ĐVT": it.get("unit") or "",
                        "Đơn giá": it["unit_price"], "Thành tiền": it["amount"],
                        "Giảm giá": it.get("discount") or 0} for it in inv["items"]], columns=ITEM_COLS)
    for c in ("SL", "Đơn giá", "Thành tiền", "Giảm giá"):
        df[c] = pd.to_numeric(df[c], errors="coerce")       # None → ô trống thay vì chữ "None"
    df["Tên hàng"] = df["Tên hàng"].astype(str)
    df["ĐVT"] = df["ĐVT"].astype(str)
    return df


def df_to_items(df):
    items = []
    for _, r in df.iterrows():
        name = str(r["Tên hàng"] or "").strip()
        if not name or name == "nan":
            continue
        num = lambda v: None if pd.isna(v) or v == "" else float(v)
        amount = num(r["Thành tiền"]) or 0
        unit_price = num(r["Đơn giá"]) or amount
        qty = num(r["SL"])
        items.append({"name": name, "qty": int(qty) if qty is not None and qty == int(qty) else qty,
                      "unit": str(r["ĐVT"] or "") if str(r["ĐVT"]) != "nan" else "",
                      "unit_price": int(unit_price), "amount": int(amount),
                      "discount": int(num(r["Giảm giá"]) or 0)})
    return items


def excel_bytes(rows):
    """rows: list (id, inv). Sheet 1: hoá đơn, sheet 2: chi tiết món."""
    heads, details = [], []
    for inv_id, inv in rows:
        heads.append({"ID": inv_id, **{label: inv.get(k) for k, label in HEADER_FIELDS},
                      **{label: inv.get(k) for k, label in MONEY_FIELDS},
                      "Số món": len(inv["items"]), "Kiểm tra": inv["checks"]["message"]})
        for it in inv["items"]:
            details.append({"ID hoá đơn": inv_id, "Cửa hàng": inv.get("store"), "Ngày": inv.get("date"),
                            "Tên hàng": it["name"], "SL": it["qty"], "ĐVT": it.get("unit"),
                            "Đơn giá": it["unit_price"], "Thành tiền": it["amount"],
                            "Giảm giá": it.get("discount") or 0})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(heads).to_excel(xw, sheet_name="Hoa don", index=False)
        pd.DataFrame(details, columns=["ID hoá đơn", "Cửa hàng", "Ngày", "Tên hàng", "SL", "ĐVT", "Đơn giá",
                                       "Thành tiền", "Giảm giá"]).to_excel(xw, sheet_name="Chi tiet mon", index=False)
    return buf.getvalue()


def money(v):
    return f"{INV.fmt_money(v)} đ" if isinstance(v, (int, float)) and v is not None else "—"


# ══════════════════════════════════════════════════════════════════════
# UI chung
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(layout='wide', page_title='SmartDoc AI — Hoá đơn', page_icon='🧾',
                   initial_sidebar_state='expanded')
with open(os.path.join(config.PROJECT_DIR, "static", "style.css"), encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

for k, d in [('page', PAGES[0]), ('cache', {}), ('conf_threshold', config.CONFIDENCE_THRESHOLD),
             ('chat_inv', None), ('chats', {})]:
    if k not in st.session_state:
        st.session_state[k] = d


def go(page, **kw):
    st.session_state.page = page
    for k, v in kw.items():
        st.session_state[k] = v
    st.rerun()


def stats():
    rows = db_all()
    invs = [json.loads(r["data_json"]) for r in rows]
    total = sum(i.get("total") or 0 for i in invs)
    n_items = sum(len(i["items"]) for i in invs)
    return rows, invs, total, n_items


with st.sidebar:
    st.markdown("""<div style='padding:.5rem 0 .8rem'>
      <div style='font-size:1.6rem'>🧾</div>
      <div style='font-size:1.05rem;font-weight:700;color:#e2e8f0'>SmartDoc AI</div>
      <div style='font-size:.7rem;color:#475569;font-family:monospace'>Trích xuất hoá đơn bán lẻ · offline</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='status-live'><span class='pulse'></span>Chạy offline — không gửi dữ liệu ra ngoài"
                "</div><br>", unsafe_allow_html=True)
    for p in PAGES:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            go(p)
    st.divider()


# ══════════════════════════════════════════════════════════════════════
# Trang
# ══════════════════════════════════════════════════════════════════════
def page_overview():
    st.markdown("<div class='hero-title'>Invoice Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>// Quét hoá đơn · trích xuất đầy đủ · kiểm tra đủ món</div>",
                unsafe_allow_html=True)
    rows, invs, total, n_items = stats()
    ok = sum(1 for i in invs if i["checks"]["status"] == "ok")
    cols = st.columns(4)
    for col, (num, icon, lbl, c) in zip(cols, [
            (len(rows), '🧾', 'Hoá đơn', '#60a5fa'), (money(total), '💰', 'Tổng chi tiêu', '#34d399'),
            (n_items, '🛒', 'Số món', '#a78bfa'), (f"{ok}/{len(rows)}", '✅', 'Đủ món (đã đối chiếu)', '#f59e0b')]):
        col.markdown(f"<div class='stat-card'><div style='font-size:1.5rem'>{icon}</div>"
                     f"<div class='stat-num' style='font-size:1.5rem;background:linear-gradient(135deg,{c},{c}99);"
                     f"-webkit-background-clip:text;background-clip:text'>{num}</div>"
                     f"<div class='stat-lbl'>{lbl}</div></div>", unsafe_allow_html=True)
    if not rows:
        st.info("Chưa có hoá đơn nào. Vào **📷 Quét hoá đơn** để bắt đầu.")
        return
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec-hdr'>🏪 Chi tiêu theo cửa hàng</div>", unsafe_allow_html=True)
        df = pd.DataFrame([{"Cửa hàng": i.get("store") or "?", "Tổng": i.get("total") or 0} for i in invs])
        st.bar_chart(df.groupby("Cửa hàng")["Tổng"].sum().sort_values(ascending=False).head(10),
                     color="#60a5fa", height=260)
    with c2:
        st.markdown("<div class='sec-hdr'>🛒 Món mua nhiều nhất</div>", unsafe_allow_html=True)
        it = pd.DataFrame([{"Món": x["name"], "Thành tiền": x["amount"]} for i in invs for x in i["items"]])
        if not it.empty:
            top = it.groupby("Món")["Thành tiền"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(8)
            top.columns = ["Số lần", "Tổng tiền"]
            st.dataframe(top, use_container_width=True)
    st.markdown("<div class='sec-hdr'>🕐 Hoá đơn gần đây</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"ID": r["id"], "Ngày": r["inv_date"], "Cửa hàng": r["store"],
                                "Số món": r["n_items"], "Tổng": money(r["total"]),
                                "Kiểm tra": STATUS_BADGE.get(r["status"], ("", ""))[0]} for r in rows[:10]]),
                 hide_index=True, use_container_width=True)


def scan_upload(f):
    data = f.getvalue()
    digest = hashlib.md5(data).hexdigest()[:12]
    if digest in st.session_state.cache:
        return st.session_state.cache[digest]
    eng = get_engine()
    out = eng.scan_invoice(data)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    img_path = os.path.join(UPLOAD_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{digest}.jpg")
    out["image"].save(img_path, quality=92)
    prob = out["invoice_prob"]
    is_inv = (prob is None or prob >= st.session_state.conf_threshold) and len(out["tokens"]) >= 3
    inv_id = db_save(out["invoice"], f.name, prob, out["ocr_text"], img_path) if is_inv else None
    res = {"id": inv_id, "is_invoice": is_inv, "prob": prob, "invoice": out["invoice"],
           "ocr_text": out["ocr_text"], "image_path": img_path, "filename": f.name}
    st.session_state.cache[digest] = res
    return res


def invoice_editor(inv_id, inv, key):
    """Hiển thị + cho phép sửa hoá đơn. Trả về True nếu đã lưu thay đổi."""
    chk = inv["checks"]
    label, cls = STATUS_BADGE.get(chk["status"], ("", "b-unk"))
    st.markdown(f"<span class='badge {cls}'>{label}</span> <span class='hint'>{chk['message']}</span>",
                unsafe_allow_html=True)
    with st.form(f"form_{key}", border=False):
        hc = st.columns(3)
        new_head = {}
        for i, (k, lbl) in enumerate(HEADER_FIELDS):
            v = inv.get(k)
            new_head[k] = hc[i % 3].text_input(lbl, "" if v in (None, INV.NOT_FOUND) else str(v), key=f"{key}_{k}")
        st.markdown("**🛒 Danh sách hàng** — sửa trực tiếp trong bảng, thêm/xoá dòng nếu thiếu/thừa")
        df = st.data_editor(items_df(inv), num_rows="dynamic", use_container_width=True, key=f"{key}_items",
                            column_config={
                                "SL": st.column_config.NumberColumn(format="%g"),
                                "Đơn giá": st.column_config.NumberColumn(format="%d"),
                                "Thành tiền": st.column_config.NumberColumn(format="%d"),
                                "Giảm giá": st.column_config.NumberColumn(format="%d")})
        mc = st.columns(4)
        new_money = {}
        for i, (k, lbl) in enumerate(MONEY_FIELDS):
            v = inv.get(k)
            new_money[k] = mc[i].number_input(lbl, value=int(v) if isinstance(v, (int, float)) else 0,
                                              step=1000, key=f"{key}_{k}")
        saved = st.form_submit_button("💾 Lưu chỉnh sửa", use_container_width=True)
    if saved and inv_id:
        for k, v in new_head.items():
            inv[k] = v.strip() or INV.NOT_FOUND
        for k, v in new_money.items():
            inv[k] = int(v) if v else (0 if k == "discount" else None)
        inv["items"] = df_to_items(df)
        inv["checks"] = INV.check_invoice(inv)
        db_save(inv, inv_id=inv_id, edited=True)
        st.success("Đã lưu. " + inv["checks"]["message"])
        return True
    return False


def page_scan():
    st.markdown("<div class='sec-hdr'>📷 Quét hoá đơn</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Kéo thả hoặc chọn ảnh hoá đơn (chụp điện thoại / scan) — tối đa 10 ảnh",
                                accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp', 'bmp'])
    if not uploaded:
        st.markdown("<div class='up-zone'><div style='font-size:2.5rem;margin-bottom:.5rem'>🧾</div>"
                    "<div style='color:#94a3b8;font-size:.88rem;font-weight:500'>Kéo thả ảnh hoá đơn vào đây</div>"
                    "<div style='color:#475569;font-size:.75rem;margin-top:.3rem;font-family:monospace'>"
                    "Ảnh nghiêng / xoay ngang vẫn đọc được · mỗi ảnh ~20–40 giây trên CPU</div></div>",
                    unsafe_allow_html=True)
        return
    for f in uploaded[:10]:
        st.markdown(f"<div class='sec-hdr'>📎 {f.name}</div>", unsafe_allow_html=True)
        try:
            with st.spinner(f"Đang đọc {f.name}..."):
                d = scan_upload(f)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            continue
        c_img, c_data = st.columns([2, 5])
        with c_img:
            st.image(d["image_path"], use_container_width=True)
            if d["prob"] is not None:
                st.caption(f"Khả năng là hoá đơn: {d['prob']:.0%}")
            with st.expander("Văn bản OCR"):
                st.text(d["ocr_text"])
        with c_data:
            if not d["is_invoice"]:
                st.warning("Ảnh này có vẻ **không phải hoá đơn bán lẻ** nên chưa được lưu.")
                if st.button("Vẫn trích xuất như hoá đơn", key=f"force_{f.name}"):
                    d["id"] = db_save(d["invoice"], d["filename"], d["prob"], d["ocr_text"], d["image_path"])
                    d["is_invoice"] = True
                    st.rerun()
                continue
            row, inv = db_get(d["id"])
            if inv is None:
                st.error("Không tìm thấy hoá đơn trong kho (có thể đã bị xoá).")
                continue
            invoice_editor(d["id"], inv, f"scan_{d['id']}")
            b1, b2 = st.columns(2)
            b1.download_button("⬇ Tải Excel", excel_bytes([(d["id"], inv)]), file_name=f"hoa_don_{d['id']}.xlsx",
                               key=f"xl_{d['id']}", use_container_width=True)
            if b2.button("💬 Hỏi về hoá đơn này", key=f"ask_{d['id']}", use_container_width=True):
                go(PAGES[2], chat_inv=d["id"])


def page_chat():
    st.markdown("<div class='sec-hdr'>💬 Hỏi đáp trên hoá đơn</div>", unsafe_allow_html=True)
    rows = db_all(limit=300)
    if not rows:
        st.info("Chưa có hoá đơn. Vào **📷 Quét hoá đơn** trước.")
        return
    ids = [r["id"] for r in rows]
    if st.session_state.chat_inv not in ids:
        st.session_state.chat_inv = ids[0]
    labels = {r["id"]: f"#{r['id']} · {r['store']} · {r['inv_date']} · {money(r['total'])}" for r in rows}
    inv_id = st.selectbox("Chọn hoá đơn", ids, index=ids.index(st.session_state.chat_inv), format_func=labels.get)
    st.session_state.chat_inv = inv_id
    row, inv = db_get(inv_id)
    c_img, c_chat = st.columns([2, 3])
    with c_img:
        if row["image_path"] and os.path.exists(row["image_path"]):
            st.image(row["image_path"], use_container_width=True)
    with c_chat:
        history = st.session_state.chats.setdefault(inv_id, [])
        qcols = st.columns(len(QUICK_PROMPTS))
        clicked = None
        for qc, qp in zip(qcols, QUICK_PROMPTS):
            if qc.button(qp, key=f"qp_{inv_id}_{qp}", use_container_width=True):
                clicked = qp
        query = st.chat_input("Bạn muốn biết gì? Ví dụ: tôi muốn lấy danh sách hàng đã mua") or clicked
        if query:
            history.append({"query": query, **INV.answer_query(query, inv)})
        for turn in history:
            with st.chat_message("user"):
                st.markdown(turn["query"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                for t in turn["tables"]:
                    st.dataframe(pd.DataFrame(t["rows"], columns=t["columns"]), hide_index=True,
                                 use_container_width=True)
                if turn["fields"]:
                    st.markdown("".join(f"<div class='field-row'><span class='k'>{f['name']}</span>"
                                        f"<span class='v{' miss' if f['value'] == INV.NOT_FOUND else ''}'>"
                                        f"{f['value']}</span></div>" for f in turn["fields"]),
                                unsafe_allow_html=True)


def page_archive():
    st.markdown("<div class='sec-hdr'>📋 Kho hoá đơn</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    search = c1.text_input("Tìm", placeholder="Tên cửa hàng, tên món...", label_visibility="collapsed")
    rows = db_all(search=search or None)
    all_inv = [(r["id"], json.loads(r["data_json"])) for r in rows]
    if rows:
        c2.download_button("⬇ Excel tất cả", excel_bytes(all_inv), file_name="kho_hoa_don.xlsx",
                           use_container_width=True)
    st.caption(f"{len(rows)} hoá đơn")
    for r, (inv_id, inv) in zip(rows, all_inv):
        badge = STATUS_BADGE.get(r["status"], ("", ""))[0]
        with st.expander(f"#{r['id']} · {r['inv_date']} · {r['store']} · {money(r['total'])} · "
                         f"{r['n_items']} món · {badge}{' · ✏️ đã sửa' if r['edited'] else ''}"):
            ci, cd = st.columns([1, 3])
            with ci:
                if r["image_path"] and os.path.exists(r["image_path"]):
                    st.image(r["image_path"], use_container_width=True)
                if st.button("💬 Hỏi", key=f"askarc_{inv_id}", use_container_width=True):
                    go(PAGES[2], chat_inv=inv_id)
                if st.button("🗑 Xoá", key=f"del_{inv_id}", use_container_width=True):
                    db_delete(inv_id)
                    st.rerun()
            with cd:
                if invoice_editor(inv_id, inv, f"arc_{inv_id}"):
                    st.rerun()


def page_settings():
    st.markdown("<div class='sec-hdr'>⚙️ Cài đặt</div>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Ngưỡng nhận diện hoá đơn")
    t = st.slider("Ảnh có xác suất 'là hoá đơn' dưới ngưỡng này sẽ bị cảnh báo và không tự lưu",
                  0.0, 1.0, float(st.session_state.conf_threshold), 0.05)
    if t != st.session_state.conf_threshold:
        st.session_state.conf_threshold = t
        st.session_state.cache.clear()
    import preprocess
    st.markdown(f"#### 🔤 OCR: `{preprocess.ocr_engine()}`")
    st.caption("vietocr = EasyOCR phát hiện vùng chữ + VietOCR đọc chữ (chính xác hơn với tiếng Việt). "
               "Đổi bằng biến môi trường SMARTDOC_OCR=easyocr.")
    st.markdown("#### 🤖 Model kiểm tra hoá đơn (LayoutLMv3)")
    mpath = os.path.join(config.active_model_dir(), "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        names = list(m["test"]["report"].keys())[:len(m["test"]["confusion_matrix"])]
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{m['test']['acc'] * 100:.2f}%")
        c2.metric("Test macro-F1", f"{m['test']['macro_f1'] * 100:.2f}%")
        st.table(pd.DataFrame(m["test"]["confusion_matrix"], index=names, columns=names))
    else:
        st.info("Chưa có metrics.json — chạy `python train.py`.")
    st.markdown("#### 🗑️ Dữ liệu")
    if st.button("Xoá toàn bộ kho hoá đơn"):
        for r in db_all():
            db_delete(r["id"])
        st.session_state.cache.clear()
        st.session_state.chats.clear()
        st.rerun()


{PAGES[0]: page_overview, PAGES[1]: page_scan, PAGES[2]: page_chat,
 PAGES[3]: page_archive, PAGES[4]: page_settings}.get(st.session_state.page, page_overview)()

# Thống kê nhanh ở sidebar — vẽ SAU trang để phản ánh hoá đơn vừa lưu
with st.sidebar:
    _rows, _invs, _total, _n = stats()
    st.markdown(f"<div class='hint'>Đã lưu <b style='color:#e2e8f0'>{len(_rows)}</b> hoá đơn · "
                f"<b style='color:#e2e8f0'>{_n}</b> món<br>Tổng chi tiêu "
                f"<b style='color:#34d399'>{money(_total)}</b></div>", unsafe_allow_html=True)
