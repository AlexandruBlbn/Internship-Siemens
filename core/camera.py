import os, time, subprocess
from datetime import datetime
from utils.fileio import load_json, save_json
from .config import CAMERA_REQUEST_FILE, CAMERA_RELEASE_FILE

def capture_picamera(photo_path):
    os.makedirs(os.path.dirname(photo_path), exist_ok=True)
    save_json(CAMERA_REQUEST_FILE, {
        "action": "release_camera",
        "timestamp": datetime.now().isoformat()
    })
    start = time.time()
    while time.time() - start < 7:
        status = load_json(CAMERA_RELEASE_FILE) or {}
        if status.get("status") == "released":
            os.remove(CAMERA_RELEASE_FILE)
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Timeout la eliberarea camerei")

    tmp = os.path.splitext(photo_path)[0] + ".jpg"
    cmd = [
        "libcamera-still", "-t", "800", "-o", tmp,
        "--width", "1440", "--height", "1080",
        "--nopreview", "--autofocus-mode", "auto"
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 10000:
        raise RuntimeError("Eroare captură Pi Camera")
    import cv2
    img = cv2.imread(tmp)
    cv2.imwrite(photo_path, img)
    os.remove(tmp)
    os.remove(CAMERA_REQUEST_FILE)