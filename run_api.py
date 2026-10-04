# ════════════════════════════════════════════════════════
#  run_api.py — Chạy FastAPI cho Mobile App (local / Colab)
#    python run_api.py            # chỉ chạy local http://localhost:8000
#    python run_api.py --ngrok    # mở tunnel ngrok (cần NGROK_AUTHTOKEN)
# ════════════════════════════════════════════════════════
import os
import sys
import argparse
import subprocess

import config

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--ngrok", action="store_true")
args = ap.parse_args()

proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0",
                         "--port", str(args.port)], cwd=config.PROJECT_DIR)
tunnel = None
if args.ngrok:
    from pyngrok import ngrok, conf
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        sys.exit("❌ Thiếu biến môi trường NGROK_AUTHTOKEN (lấy tại https://dashboard.ngrok.com)")
    conf.get_default().auth_token = token
    tunnel = ngrok.connect(args.port, "http")
    print(f"\n  ✅ API cho Mobile App: {tunnel.public_url}\n")
else:
    print(f"\n  ✅ API: http://localhost:{args.port}  (docs: /docs)\n")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    if tunnel:
        from pyngrok import ngrok
        ngrok.disconnect(tunnel.public_url)
