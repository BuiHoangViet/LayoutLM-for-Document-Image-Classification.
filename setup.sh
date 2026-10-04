#!/bin/bash
# setup.sh — Cài môi trường (Linux / Colab). Trên Windows chạy các lệnh pip tương tự.
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt
pip install -q --no-deps vietocr
mkdir -p ~/.cache/vietocr
[ -f ~/.cache/vietocr/vgg_transformer.pth ] || \
    curl -L -o ~/.cache/vietocr/vgg_transformer.pth https://vocr.vn/data/vietocr/vgg_transformer.pth
echo "✅ Setup hoàn tất! Chạy: streamlit run app.py"
