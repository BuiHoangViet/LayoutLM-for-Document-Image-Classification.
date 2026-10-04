# ============================================================
# invoice.py — Trích xuất ĐẦY ĐỦ thông tin hoá đơn bán lẻ (offline)
#
#   extract_invoice(lines, tokens) → dict:
#     store, address, phone, tax_code, invoice_no, date, time, cashier,
#     items[{name, qty, unit, unit_price, amount, discount}],
#     subtotal, discount, total, paid, change, vat, payment_method,
#     checks{items_sum, matches_total, missing_amount, ...}
#
#   answer_query(query, inv) → câu trả lời cho chat ("danh sách hàng đã mua"...)
#
# Mọi so khớp từ khoá đều chạy trên văn bản đã BỎ DẤU và cho phép sai 1 ký tự,
# vì OCR hay đọc nhầm dấu (TỔNG → TỠNG / TŨNG, TIỀN → TIẼN ...).
# ============================================================
import re
import unicodedata

NOT_FOUND = "Không tìm thấy"


# ------------------------------------------------------------------
# Tiện ích
# ------------------------------------------------------------------
def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


# 1.234.000 | 1,234,000 | 48,139.00 | 6,783.20 | 15000
MONEY_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d{4,9})(?:[.,](\d{1,2}))?")
OCR_DIGIT_FIX = str.maketrans({"o": "0", "O": "0", "u": "0", "U": "0", "l": "1", "I": "1", "|": "1",
                                "T": "7", "S": "5", "B": "8"})


def parse_money(tok: str):
    """'105.806' / '145,000' / '48,139.00' / '63750' / '15,0o0' → số nguyên VND, None nếu không phải tiền."""
    s = tok.strip().strip("đdĐ;:()~*").strip()
    s = s.lstrip("-").strip()
    if not s or not s[0].isdigit():
        return None
    if re.match(r"0[.,]", s):                     # '0,460' (kg) là số lượng
        return None
    if re.search(r"[A-Za-z]", s):                 # sửa lỗi OCR phổ biến trong số: 15,0o0 → 15,000
        if len(re.findall(r"[A-Za-z]", s)) > 2:
            return None
        s = s.translate(OCR_DIGIT_FIX)
    m = MONEY_RE.fullmatch(s)
    if not m:
        m2 = re.fullmatch(r"(\d{1,3}(?:[.,]\d{3})+)[4d]", s)   # '75.0004' = '75.000đ'
        m = MONEY_RE.fullmatch(m2.group(1)) if m2 else None
    if not m:
        return None
    v = int(re.sub(r"[.,]", "", m.group(1)))
    if m.group(2):
        v = int(round(v + int(m.group(2)) / 10 ** len(m.group(2))))
    return v


def parse_qty(tok: str):
    s = tok.strip().strip(";:()*x").strip()
    if not re.fullmatch(r"\d{1,3}(?:[.,]\d{1,3})?", s):
        return None
    # '0,260' / '0.598' → số thập phân (kg); '1.000'(4 chữ số) không vào đây
    return float(s.replace(",", ".")) if re.search(r"[.,]", s) else int(s)


def fmt_money(v):
    return f"{v:,}".replace(",", ".") if isinstance(v, (int, float)) and v is not None else ""


def _f(pat):
    return re.compile(pat)


# ------------------------------------------------------------------
# Nhận diện loại dòng (trên văn bản bỏ dấu)
# ------------------------------------------------------------------
TOTAL_STRONG = _f(r"phai\s*t\S*\s*t\S?\S?an|tong\s*so\s*tien\s*thanh|tong\s*s\S?\s*thanh\s*to|t\S?ng\s*thanh\s*to")
TOTAL_RE  = _f(r"\bt\S?ng\s*(?:c\S{0,2}ng|ti\S{0,3}n)|thanh\s*to\S?n|\bt\.\s*to\S?n|phai\s*t|"
               r"\bc\S?ng\s*ti\S{0,2}n?\s*h|\btotal\b|\bt\S?ng\s*s\S?\s*ti")
COUNT_RE  = _f(r"\bt\S?ng\s*s\S?\s*(?::|\d|l\S{0,3}ng|m\S?t|sp\b)|\bso\s*luong\b|luong\s*hang")
DISC_RE   = _f(r"gi\S?m\s*gi\S?|\b\S?\s*gi\S?m\b|chi\S?t\s*kh\S?u|\bck\b|khuyen\s*mai")
PAID_RE   = _f(r"kh\S?ch\s*(?:tr\S?|d\S?a)\b|ti\S{0,2}n\s*[mn]\S?t\b|\bcash\b|\bvisa\b|thanh\s*toan\s*the")
CHANGE_RE = _f(r"tr\S?\s*l\S?i\b|ti\S{0,2}n\s*th\S?a\b|th\S?i\s*l\S?i|\bchange\b")
VAT_RE    = _f(r"\bvat\b|gtgt|thue\s*suat")
META_RE   = _f(r"hot\s*l?ine|\btel\b|\bdt\b|dien\s*thoai|\bmst\b|ma\s*so\s*thue|dia\s*chi|website|"
               r"\bngay\b|thu\s*ngan|nhan\s*vien|\bquay\b|\bso\s*gd\b|h\S?a\s*d\S?n|ban\s*hang|"
               r"cam\s*on|c\S?m\s*\S?n|chiet\s*khau|\bdiem\b|the\s*khach|ma\s*the|hang\s*hien|"
               r"tham\s*chi|invoice|\bgia\s*da\s*bao|gia\s*dang\s*k|tich\s*luy|email|phuong\s*thuc|"
               r"\bnvbh\b|\bso\s*hd\b|\bma\s*hd\b|\bso\s*the\b|phe\s*duyet|giao\s*dich|"
               r"\b0\d{9,10}\b|\b0\d{2,3}[.\s]\d{6,8}|\bchk\b|\bpos\d|store\s*id|\bclosed\b|\bstoreid")
HEADER_RE = _f(r"mat\s*hang|ten\s*hang|san\s*pham|don\s*gia|\S?n\s*gi\S?\b|thanh\s*ti|\bsl\b|\bdvt\b|t\.\s*ti")
UNIT_WORDS = {"goi", "hop", "h@p", "cai", "cha", "chai", "lon", "tui", "tuy", "kg", "g", "bo", "thung", "ly",
              "coc", "km", "dvt", "sl", "cay", "vi", "lo", "bich", "hu", "can", "suat", "phan", "ea", "pcs",
              "x", "r", "t", "nt", "gia", "ban", "ban:", "goi.", "dia", "to", "chiec", "doi", "hop."}
UNIT_NAMES = {"goi": "gói", "hop": "hộp", "cai": "cái", "cha": "chai", "chai": "chai", "lon": "lon",
              "tui": "túi", "tuy": "túi", "kg": "kg", "vi": "vỉ", "bich": "bịch", "thung": "thùng",
              "chiec": "chiếc", "doi": "đôi", "ly": "ly", "coc": "cốc"}
BARCODE_RE = re.compile(r"^\d{8,14}$")
VATTAG_RE = re.compile(r"^vat[o0]?\d*[%z2]?$", re.I)


def _na(line):
    return strip_accents(line)


def normalize_line(line: str) -> str:
    """Sửa các lỗi OCR hay gặp trong số tiền: '000. 00' → '000.00', '100 000.00' → '100,000.00'."""
    line = re.sub(r"(\d)([.,])\s+(\d{2,3})\b", r"\1\2\3", line)
    line = re.sub(r"\b(\d{1,3})\s(\d{3}[.,]\d{2})\b", r"\1,\2", line)
    na = strip_accents(line)
    if TOTAL_RE.search(na) or PAID_RE.search(na) or CHANGE_RE.search(na):
        line = re.sub(r"([A-Za-zÀ-ỹ:.]\s+)(\d{1,3}) (\d{3})$", r"\1\2.\3", line)
    return line


DATE_RE = re.compile(r"\b\d{1,2}\s*/\s*\d{1,2}\s*[/\-.\s]\s*(?:19|20)\d{2}\b")


def _is_meta(na):
    return bool(DATE_RE.search(na) or TOTAL_RE.search(na) or PAID_RE.search(na) or CHANGE_RE.search(na) or META_RE.search(na)
                or COUNT_RE.search(na) or (VAT_RE.search(na) and not re.search(r"vat\d", na)))


def _analyze_line(line):
    """Tách 1 dòng thành: tên hàng, các số tiền, số lượng, đơn vị, mã vạch."""
    name_toks, money, qtys, unit, barcode = [], [], [], "", ""
    toks = line.split()
    has_money = any(parse_money(t.strip(";:,()~*")) is not None for t in toks)
    for i, tok in enumerate(toks):
        clean = tok.strip(";:,()~*").strip()
        low = strip_accents(clean).strip(".:")
        if not clean:
            continue
        if BARCODE_RE.match(clean):
            barcode = barcode or clean
            continue
        if VATTAG_RE.match(clean) or clean.startswith("-"):
            continue                                # VAT05% / số âm (giảm giá) không phải tên hay giá
        nxt = strip_accents(toks[i + 1]).strip(".,:;") if i + 1 < len(toks) else ""
        v = parse_money(clean)
        if v is not None and nxt in ("kg", "g", "gr"):
            v = None                                 # '10,878 KG' = khối lượng (OCR đọc nhầm 0,878)
            continue
        if v is not None:
            money.append(v)
            continue
        q = parse_qty(clean)
        if q is not None and (money or name_toks or i + 1 < len(toks)):
            qtys.append(q)
            continue
        if low in UNIT_WORDS and (has_money or low in ("km", "x", "nt", "r", "t", "dvt", "sl")):
            if low in UNIT_NAMES and not unit:
                unit = UNIT_NAMES[low]
            continue
        if not money:
            name_toks.append(tok)
    while name_toks and (len(name_toks[0].strip(".,:;'\"")) <= 1 or name_toks[0].isdigit()):
        name_toks.pop(0)                            # rác OCR ở lề trái: '5', 'j', 'g', '"'
    name = " ".join(name_toks).strip(" -:*_.,'\"")
    name = re.sub(r"^(?:gi[aá]\s*b[aá]n|km)\s*:?\s*", "", name, flags=re.I)
    letters = len(re.findall(r"[A-Za-zÀ-ỹ]", name))
    if len(name_toks) == 1 and letters < 0.6 * len(name):
        letters = 0                                  # 'GO.S82' → không phải tên hàng
    return {"name": name if letters >= 3 else "", "money": money, "qty": qtys, "unit": unit,
            "barcode": barcode, "letters": letters}


# ------------------------------------------------------------------
# Danh sách hàng
# ------------------------------------------------------------------
def _pick_qty(cands, unit_price, amount):
    """Chọn số lượng sao cho SL × đơn giá ≈ thành tiền (bền với số rác OCR trên dòng)."""
    cands = [q for q in cands if q]
    if unit_price:
        for q in cands:
            if abs(q * unit_price - amount) <= max(2, 0.002 * amount):
                return q
        if amount % unit_price == 0 and 0 < amount // unit_price <= 1000:
            return amount // unit_price
        floats = [q for q in cands if isinstance(q, float)]
        if floats:                                   # hàng cân ký: 0,878 KG
            return floats[0]
    if unit_price is None or unit_price == amount:
        floats = [q for q in cands if isinstance(q, float)]
        return floats[0] if floats else 1
    if cands:
        return cands[0]
    return 1 if 0.8 <= amount / unit_price <= 1.25 else None


def parse_items(lines):
    """Tách danh sách món. Hỗ trợ các bố cục phổ biến:
       A) 'Tên ... SL ĐVT đơn giá thành tiền' trên 1 dòng
       B) 'mã vạch Tên' dòng 1  →  'SL ĐVT đơn giá thành tiền' dòng 2  (Minimart, Co.op)
       C) 'Tên' dòng 1  →  'mã vạch đơn giá SL thành tiền' dòng 2         (VinMart)
       D) 'Giá đăng ký: x' / 'Giá bán: SL ĐVT đơn giá thành tiền'
       E) dòng 'KM -5.850' ngay sau 1 món → giảm giá của món đó."""
    lines = [normalize_line(l.strip()) for l in lines if l and l.strip()]
    items, pending, header_seen = [], None, False
    for i, line in enumerate(lines):
        na = _na(line)
        if (TOTAL_RE.search(na) or COUNT_RE.search(na)) and items and not re.search(r"gia\s*ban", na):
            break                                    # tới phần tổng → hết danh sách hàng
        # Giảm giá của món ngay trước (KM -5.850 / ~-50,000 / Chiết khấu)
        if items and (re.match(r"^\W*(?:km|ck|giam)\b", na) or re.match(r"^[~\-]\s*\d", na)
                      or (re.search(r"-\s*\d[\d.,]{3,}\s*$", line) and not _analyze_line(line)["barcode"])):
            vals = [parse_money(t) for t in re.findall(r"[\d][\d.,]*", line)]
            vals = [v for v in vals if v]
            if vals:
                items[-1]["discount"] = (items[-1].get("discount") or 0) + vals[-1]
                continue
        if re.search(r"gia\s*dang\s*k", na):
            continue                                 # giá niêm yết, không phải dòng hàng
        a = _analyze_line(line)
        if HEADER_RE.search(na) and not a["money"]:
            if not header_seen:
                items.clear()                        # mọi thứ trước tiêu đề bảng là thông tin cửa hàng
            header_seen, pending = True, None
            continue
        if _is_meta(na) and not re.search(r"gia\s*ban", na):
            pending = None
            continue
        if not a["money"]:
            if a["name"]:
                # Dòng tiếp nối tên (vd '0,878 KG' / 'KG') chỉ cập nhật số lượng
                pending = {"name": a["name"], "qty": a["qty"][0] if a["qty"] else None}
            elif pending and a["qty"]:
                pending["qty"] = a["qty"][0]
            elif items and a["qty"] and isinstance(a["qty"][0], float):
                items[-1]["qty"] = a["qty"][0]       # '0.598 KG' nằm dưới dòng giá (Co.op)
                items[-1]["unit"] = items[-1]["unit"] or "kg"
            continue
        # Dòng tên có 1 số (vd '300g' đọc thành 3009) + dòng sau chỉ toàn số tiền → ghép làm tên
        if a["name"] and len(a["money"]) == 1 and i + 1 < len(lines):
            nxt = _analyze_line(lines[i + 1])
            if not nxt["name"] and len(nxt["money"]) >= 1 and not _is_meta(_na(lines[i + 1])):
                pending = {"name": line, "qty": None}
                continue
        weak = a["name"] and len(a["name"]) <= 4 and " " not in a["name"]
        name = ((pending or {}).get("name") if weak and pending else None) or a["name"] \
            or (pending or {}).get("name") or (f"Mã {a['barcode']}" if a["barcode"] else "")
        if not name:
            continue
        amount = a["money"][-1]
        unit_price = a["money"][-2] if len(a["money"]) >= 2 else None
        int_q = [q for q in a["qty"] if isinstance(q, int) and q > 1]
        if unit_price is None and int_q and re.search(r"gia\s*ban", na):
            unit_price, amount = amount, amount * int_q[0]
        if amount < 500:
            continue
        cands = list(a["qty"]) + ([pending["qty"]] if pending and pending.get("qty") else [])
        qty = _pick_qty(cands, unit_price, amount)
        # Hàng cân ký: 'Cà rốt 30.000/KG x 0,260 KG' → SL 1 gói theo bill, giữ nguyên
        items.append({"name": name, "qty": qty, "unit": a["unit"], "unit_price": unit_price or amount,
                      "amount": amount, "discount": 0})
        pending = None
    seen, out = set(), []
    for it in items:                                 # bỏ dòng OCR lặp
        k = (it["name"], it["amount"])
        if k in seen:
            continue
        if out and it["name"].startswith("Mã ") and out[-1]["amount"] == it["amount"] \
                and not out[-1]["name"].startswith("Mã "):
            continue                                 # dòng mã vạch lặp lại tiền của món ngay trên
        seen.add(k)
        out.append(it)
    return out


# ------------------------------------------------------------------
# Tổng tiền & các trường đầu / cuối hoá đơn
# ------------------------------------------------------------------
def _line_money(line):
    vals = []
    for t in line.split():
        v = parse_money(t)
        if v is not None and not BARCODE_RE.match(t.strip()):
            vals.append(v)
    return vals


def _money_for(lines, pattern, exclude=None, look_around=True, info=None):
    """Số tiền trên dòng khớp nhãn; nếu dòng nhãn không có số thì lấy dòng kế bên chỉ có số."""
    info = info if info is not None else {}
    for i, line in enumerate(lines):
        na = _na(line)
        if pattern.search(na) and not (exclude and exclude.search(na)):
            vals = _line_money(line)
            if vals:
                return vals[-1]
            if look_around:
                for j in (i + 1, i - 1):
                    if 0 <= j < len(lines):
                        nb, nb_na = lines[j], _na(lines[j])
                        other_label = (PAID_RE.search(nb_na) or CHANGE_RE.search(nb_na) or DISC_RE.search(nb_na)
                                       or (TOTAL_RE.search(nb_na) and pattern is not TOTAL_RE))
                        vals = [v for v in _line_money(nb) if v >= 1000]
                        if vals and (len(re.findall(r"[A-Za-zÀ-ỹ]", nb)) <= 4 or not other_label):
                            info["neighbor"] = True
                            return vals[-1]
    return None


KNOWN_STORES = [
    (r"vin\s*commerce|vinmart|vm\+", "VinMart / VinCommerce"),
    (r"co\.?\s*op\s*food|saigon\s*co\.?\s*op|co\.?op\s*mart", "Co.op"),
    (r"bach\s*hoa\s*xanh", "Bách Hoá Xanh"),
    (r"circle\s*k", "Circle K"), (r"family\s*mart", "FamilyMart"), (r"ministop", "Ministop"),
    (r"lotte\s*mart", "Lotte Mart"), (r"big\s*c\b|go!", "Big C / GO!"), (r"aeon", "AEON"),
    (r"uniqlo", "UNIQLO"), (r"guardian", "Guardian"), (r"winmart", "WinMart"),
]
ADDR_RE = _f(r"\b(?:so|to|khu|duong|d\.|pho|p\.|phuong|q\.|quan|huyen|tp\.?|thanh\s*pho|tinh|t\.|thi\s*xa|"
             r"xa|thon|cho|ngo|ngach|kiot|tang|toa|ha\s*noi|hcm|ho\s*chi\s*minh|quang\s*ninh|da\s*nang)\b")


def _store(lines):
    head = lines[:8]
    for line in head:
        na = _na(line)
        for pat, name in KNOWN_STORES:
            if re.search(pat, na):
                # Giữ tên chi nhánh nếu có (vd 'VM+ QNH Tổ 7, Khu Minh Tiến A')
                return name, line.strip()
    for line in head:
        na = _na(line)
        letters = re.findall(r"[A-Za-zÀ-ỹ]", line)
        if len(letters) >= 4 and not META_RE.search(na) and not HEADER_RE.search(na) \
                and not _line_money(line) and not ADDR_RE.search(na) \
                and not any(BARCODE_RE.match(t) for t in line.split()):
            return line.strip(), ""
    return NOT_FOUND, ""


def _address(lines, store_line):
    out = []
    for line in lines[:10]:
        na = _na(line)
        if re.search(r"h\S?a\s*\S?\S?n|phieu|mat\s*hang|ten\s*hang|ngay", na):
            break
        m = re.search(r"(?:dia\s*chi|d/c|dc)\s*[:.]?\s*", na)
        if m:
            return line[m.end():].strip(" :.") or NOT_FOUND
        if line.strip() != store_line and ADDR_RE.search(na) and not re.search(r"\btel\b|\bdt\b|\bmst\b|email|web", na):
            out.append(line.strip(" ,"))
    return ", ".join(out[:3]) if out else NOT_FOUND


MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                          "oct", "nov", "dec"])}


class _M:                                            # giả lập re.Match cho ngày tiếng Anh
    def __init__(self, d, m, y):
        self._g = (None, str(d), str(m), y)

    def group(self, i):
        return self._g[i]


def _en_date(text):
    m = re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{2,4})\b",
                  text, re.I)
    return _M(m.group(1), MONTHS[m.group(2).lower()], m.group(3)) if m else None


def extract_invoice(lines, tokens=None) -> dict:
    lines = [normalize_line(l.strip()) for l in (lines or []) if l and l.strip()]
    text = "\n".join(lines)
    na_text = _na(text)
    store, store_line = _store(lines)
    inv = {"store": store, "branch": store_line if store_line and store_line != store else ""}
    inv["address"] = _address(lines, store_line)

    m = re.search(r"\b(0\d{2,3}[.\s]?\d{3,4}[.\s]?\d{3,4})\b", text) or re.search(r"\b(0\d{9,10})\b", text)
    inv["phone"] = m.group(1) if m else NOT_FOUND
    m = re.search(r"(?:mst|ma\s*so\s*thue|tax\s*id)\s*[:.]?\s*(\d[\d\-\s]{8,15}\d)", na_text)
    inv["tax_code"] = m.group(1).strip() if m else NOT_FOUND
    m = (re.search(r"(?:so\s*gd|so\s*hd|ma\s*hd|so\s*ct|so\s*hoa\s*don)\s*[:.]?\s*([a-z0-9]{4,25})", na_text)
         or re.search(r"\bh\s*:\s*(\d{6,10})", na_text))
    inv["invoice_no"] = m.group(1).upper() if m else NOT_FOUND
    m = (re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*((?:19|20)?\d{2})\b", text)
         or re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s+((?:19|20)\d{2})\b", text)
         or _en_date(text))
    inv["date"] = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3) if len(m.group(3)) == 4 else '20' + m.group(3)}" \
        if m else NOT_FOUND
    m = (re.search(r"(?<![\d.,])([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?::([0-5]\d))?(?![\d.,])", text)
         or re.search(r"\d{4}\s*-\s*([01]?\d|2[0-3])[.,:]\s*([0-5]\d)()", text))
    inv["time"] = (f"{int(m.group(1)):02d}:{m.group(2)}" + (f":{m.group(3)}" if m.group(3) else "")) if m else NOT_FOUND
    m = re.search(r"(?:thu\s*ngan|nhan\s*vien|nvbh|ten\s*nv)\s*[:.]?\s*([^\n]{2,30})", na_text)
    inv["cashier"] = m.group(1).strip().upper() if m else NOT_FOUND

    items = parse_items(lines)

    total = (_money_for(lines, TOTAL_STRONG, exclude=DISC_RE)
             or _money_for(lines, _f(r"\bt\S?ng\s*c\S{0,2}ng"), exclude=DISC_RE)
             or _money_for(lines, TOTAL_RE, exclude=_f(DISC_RE.pattern + "|" + COUNT_RE.pattern)))
    inv["total"] = total
    inv["items"] = items
    inv["discount"] = _money_for(lines, DISC_RE, look_around=False) or 0
    paid_vals = [v for i in range(len(lines)) if PAID_RE.search(_na(lines[i]))
                 for v in [_money_for(lines[i:i + 2], PAID_RE)] if v]
    inv["paid"] = max(paid_vals) if paid_vals else None
    change_info = {}
    inv["change"] = _money_for(lines, CHANGE_RE, info=change_info)
    vat = re.search(r"(?:vat|gtgt)[^\n\d]{0,20}(\d{1,2})\s*%", na_text)
    inv["vat_rate"] = f"{vat.group(1)}%" if vat else NOT_FOUND
    pm = []
    for pat, name in [(r"tien\s*mat|cash", "Tiền mặt"), (r"\bvisa\b|master|the\s*tin\s*dung|so\s*the", "Thẻ"),
                      (r"chuyen\s*khoan|qr|momo|zalopay|vnpay", "Chuyển khoản / ví"), (r"vinid", "VinID")]:
        if re.search(pat, na_text):
            pm.append(name)
    inv["payment_method"] = ", ".join(pm) or NOT_FOUND

    inv["total_source"] = "dòng tổng" if total is not None else ""
    items_sum = sum(it["amount"] - (it.get("discount") or 0) for it in items)
    cap = max(v for v in (total, inv["paid"]) if v) if (total or inv["paid"]) else None
    if cap:                                          # 1 món không thể lớn hơn tổng HĐ / tiền khách trả
        items = [it for it in items if it["amount"] <= cap * 1.01]
        inv["items"] = items
        items_sum = sum(it["amount"] - (it.get("discount") or 0) for it in items)
    paid, change = inv["paid"], inv["change"]
    # 2 nguồn độc lập khớp nhau (tổng các món = khách trả) thắng 1 dòng tổng đọc lệch
    if items and paid == items_sum and total != items_sum:
        inv["total"], inv["total_source"] = items_sum, "đối chiếu (tổng món = khách trả)"
    elif total is None and items and paid and change is not None and paid - change == items_sum:
        inv["total"], inv["total_source"] = items_sum, "đối chiếu (khách trả − tiền thừa)"
    # Tiền thừa phải = khách trả − tổng
    if paid and inv["total"] and change is not None and paid >= inv["total"] \
            and abs(paid - inv["total"] - change) > 1 and (change_info.get("neighbor") or change > paid):
        inv["change"] = paid - inv["total"]

    # Tiền thừa = khách trả − tổng, nếu bill không in
    if inv["change"] is None and inv["paid"] and inv["total"] and inv["paid"] >= inv["total"]:
        inv["change"] = inv["paid"] - inv["total"]
    inv["checks"] = check_invoice(inv)
    return inv


def check_invoice(inv):
    """Kiểm tra ĐỦ / ĐÚNG: tổng các món (trừ giảm giá) có khớp tổng hoá đơn không."""
    items = inv["items"]
    s = sum(it["amount"] - (it.get("discount") or 0) for it in items)
    bad_lines = [it["name"] for it in items
                 if isinstance(it["qty"], int) and it["unit_price"] and it["qty"] * it["unit_price"] != it["amount"]
                 and it["unit_price"] != it["amount"]]
    total = inv.get("total")
    res = {"n_items": len(items), "items_sum": s, "bad_lines": bad_lines}
    if total is None:
        res.update(status="unknown", message="Không đọc được tổng tiền để đối chiếu.")
    elif not items:
        res.update(status="missing", message="Không tách được món nào.")
    elif abs(s - total) <= max(1, 0.005 * total):
        res.update(status="ok", message=f"Đủ: tổng {len(items)} món = {fmt_money(s)} đ khớp tổng hoá đơn.")
    elif abs(s - (inv.get("discount") or 0) - total) <= max(1, 0.005 * total) \
            or abs(sum(it["amount"] for it in items) - total) <= max(1, 0.005 * total):
        res.update(status="ok", message=f"Đủ: tổng món − giảm giá = {fmt_money(total)} đ khớp tổng hoá đơn.")
    elif s < total:
        res.update(status="missing", missing_amount=total - s,
                   message=f"Có thể thiếu món: tổng các món {fmt_money(s)} đ < tổng hoá đơn {fmt_money(total)} đ "
                           f"(lệch {fmt_money(total - s)} đ).")
    else:
        res.update(status="excess", message=f"Tổng các món {fmt_money(s)} đ > tổng hoá đơn {fmt_money(total)} đ "
                                             "— có dòng bị đọc nhầm/thừa.")
    return res


# ------------------------------------------------------------------
# Chat: trả lời câu hỏi trên hoá đơn đã trích xuất
# ------------------------------------------------------------------
FIELD_LABELS = [
    ("store", "Cửa hàng"), ("branch", "Chi nhánh"), ("address", "Địa chỉ"), ("phone", "Điện thoại"),
    ("tax_code", "Mã số thuế"), ("invoice_no", "Số hoá đơn / GD"), ("date", "Ngày"), ("time", "Giờ"),
    ("cashier", "Thu ngân"), ("total", "Tổng thanh toán"), ("discount", "Giảm giá"), ("paid", "Khách trả"),
    ("change", "Tiền thừa"), ("vat_rate", "Thuế VAT"), ("payment_method", "Hình thức thanh toán"),
]
MONEY_FIELDS = {"total", "discount", "paid", "change"}

INTENTS = {   # regex trên câu hỏi đã bỏ dấu
    "items":   r"\bmon\b|san pham|mat hang|hang hoa|hang da mua|\bda mua\b|\bmua (gi|nhung)|items?\b|"
               r"chi tiet|danh sach|liet ke|\bhang (?:gi|nao)\b",
    "total":   r"\btong\b|\btotal\b|bao nhieu tien|het bao nhieu|phai tra|thanh toan|tong cong",
    "paid":    r"khach tra|khach dua|da tra|tra bao nhieu",
    "change":  r"tien thua|tra lai|thoi lai",
    "discount": r"giam gia|khuyen mai|chiet khau|\bkm\b",
    "date":    r"\bngay\b|thoi gian|khi nao|\bdate\b|luc nao|\bgio\b",
    "store":   r"cua hang|nguoi ban|ben ban|don vi ban|\bshop\b|\bo dau\b|sieu thi|ten quan",
    "address": r"dia chi|o dau",
    "phone":   r"dien thoai|\bsdt\b|hotline|so dt",
    "invoice_no": r"so hoa don|so hd|ma hd|so gd|ma giao dich",
    "cashier": r"thu ngan|nhan vien",
    "payment": r"hinh thuc|phuong thuc|tra bang|the|tien mat",
    "check":   r"\bdu\b|thieu|kiem tra|khop|dung khong",
    "max":     r"dat nhat|nhieu tien nhat|cao nhat|lon nhat",
    "all":     r"tat ca|toan bo|tom tat|thong tin|trich xuat|\ball\b",
}
INTENT_FIELDS = {"total": ["total"], "paid": ["paid"], "change": ["change"], "discount": ["discount"],
                 "date": ["date", "time"], "store": ["store", "branch", "address"], "address": ["address"],
                 "phone": ["phone"], "invoice_no": ["invoice_no"], "cashier": ["cashier"],
                 "payment": ["payment_method", "paid"]}


def items_table(inv):
    rows = [[it["name"], str(it["qty"] if it["qty"] is not None else ""), it.get("unit") or "",
             fmt_money(it["unit_price"]), fmt_money(it["amount"]),
             fmt_money(it["discount"]) if it.get("discount") else ""] for it in inv["items"]]
    return {"title": "Danh sách hàng đã mua",
            "columns": ["Tên hàng", "SL", "ĐVT", "Đơn giá", "Thành tiền", "Giảm giá"], "rows": rows}


def field_list(inv, keys):
    out = []
    for k, label in FIELD_LABELS:
        if k in keys:
            v = inv.get(k)
            if k in MONEY_FIELDS:
                v = (fmt_money(v) + " đ") if v else (NOT_FOUND if k != "discount" else "0 đ")
            out.append({"name": label, "value": v or NOT_FOUND})
    return out


def answer_query(query: str, inv: dict) -> dict:
    q = strip_accents(query)
    intents = {name for name, pat in INTENTS.items() if re.search(pat, q)}
    res = {"answer": "", "fields": [], "tables": [], "engine": "offline"}
    chk = inv["checks"]

    if "max" in intents and inv["items"]:
        it = max(inv["items"], key=lambda x: x["amount"])
        res["answer"] = f"Món đắt nhất: **{it['name']}** — {fmt_money(it['amount'])} đ."
        return res
    if "check" in intents and not re.search(r"danh sach|liet ke", q):
        res["answer"] = chk["message"]
        res["fields"] = field_list(inv, ["total"]) + [{"name": "Tổng các món", "value": fmt_money(chk["items_sum"]) + " đ"},
                                                      {"name": "Số món", "value": str(chk["n_items"])}]
        return res
    if "items" in intents or "all" in intents or not intents:
        if inv["items"]:
            res["tables"].append(items_table(inv))
            res["answer"] = f"Hoá đơn có **{len(inv['items'])} món**. " + chk["message"]
        else:
            res["answer"] = "Không tách được danh sách món từ ảnh này. " + chk["message"]
        keys = ["store", "date", "time", "total"] if "all" not in intents else [k for k, _ in FIELD_LABELS]
        res["fields"] = field_list(inv, keys)
        if intents - {"items", "all"}:
            for i in intents - {"items", "all", "check", "max"}:
                for k in INTENT_FIELDS.get(i, []):
                    if k not in keys:
                        res["fields"] += field_list(inv, [k])
        return res
    if "check" in intents:
        res["answer"] = chk["message"]
        res["fields"] = field_list(inv, ["total"]) + [{"name": "Tổng các món", "value": fmt_money(chk["items_sum"]) + " đ"}]
        return res
    keys = []
    for i in intents:
        keys += INTENT_FIELDS.get(i, [])
    res["fields"] = field_list(inv, keys)
    found = [f for f in res["fields"] if f["value"] != NOT_FOUND]
    res["answer"] = "; ".join(f"{f['name']}: **{f['value']}**" for f in found) if found else \
        "Không tìm thấy thông tin này trên hoá đơn."
    return res
