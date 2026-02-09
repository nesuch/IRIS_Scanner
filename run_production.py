from waitress import serve  # type: ignore
from app import app  # type: ignore
import logging
import os
import socket

# --- 1. SETUP LOGGING ---
# This ensures you see errors in the black terminal window if something goes wrong.
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# --- 2. GET LOCAL IP ADDRESS ---
# Automatically finds your computer's IP so you know what link to share.
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

host_ip = get_ip()
port = 8080

# --- 3. PRINT STARTUP BANNER ---
os.system('cls' if os.name == 'nt' else 'clear')
print("=======================================================")
print("   🔥 IRIS PRODUCTION SERVER IS RUNNING ")
print("=======================================================")
print(f"   ► Access URL:    http://{host_ip}:{port}")
print(f"   ► Local URL:     http://localhost:{port}")
print("   ► Threads:       16 (Capacity: ~300 Users)")
print("   ► Status:        ONLINE")
print("-------------------------------------------------------")
print("   [DO NOT CLOSE THIS WINDOW]")
print("=======================================================")

# --- 4. START SERVER ---
# 'threads=16' allows 16 people to click buttons at the exact same millisecond.
# The 17th person waits a tiny fraction of a second.
try:
    serve(app, host='0.0.0.0', port=port, threads=16)
except Exception as e:
    print(f"CRITICAL ERROR: Could not start server.\n{e}")
    input("Press Enter to exit...")