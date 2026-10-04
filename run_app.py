# ════════════════════════════════════════════════════════
#  run_app.py — Chạy giao diện web SmartDoc AI (local / Colab)
#    python run_app.py            # http://localhost:8501
#    python run_app.py --ngrok    # mở tunnel ngrok (cần NGROK_AUTHTOKEN)
# ════════════════════════════════════════════════════════
import os
import sys
import argparse
import subprocess

import config

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--ngrok", action="store_true")
args = ap.parse_args()

proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(config.PROJECT_DIR, "app.py"),
                         "--server.port", str(args.port), "--server.headless", "true"],
                        cwd=config.PROJECT_DIR)
tunnel = None
if args.ngrok:
    from pyngrok import ngrok, conf
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        sys.exit("❌ Thiếu biến môi trường NGROK_AUTHTOKEN (lấy tại https://dashboard.ngrok.com)")
    conf.get_default().auth_token = token
    tunnel = ngrok.connect(args.port, "http")
    print(f"\n  ✅ SmartDoc AI: {tunnel.public_url}\n")
else:
    print(f"\n  ✅ SmartDoc AI: http://localhost:{args.port}\n")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    if tunnel:
        from pyngrok import ngrok
        ngrok.disconnect(tunnel.public_url)
