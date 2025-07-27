import os
import sys
import time
import csv
import cv2
import requests
import face_recognition
import numpy as np
import signal
import argparse
from datetime import datetime
from threading import Lock

# -----------------------------
# CONFIGURATION
# -----------------------------
API_TOKEN = "ap5igza77o7kq8j6xtfkacv6g19xma"
USER_KEY = "us9hse3b4mvx8e23fobahwad1e53kd"
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
LOG_CSV = "face_recognition_log.csv"
LOCKFILE = "/tmp/face_rec.lock"
CAMERA_INDEX = 0
FRAME_WIDTH = 500
NOTIFY_COOLDOWN = 30  # seconds per person

# -----------------------------
# PUSHOVER NOTIFICATION
# -----------------------------
def send_notification(message):
    data = {
        "token": API_TOKEN,
        "user": USER_KEY,
        "message": message
    }
    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data=data)
        if response.status_code == 200:
            print(f"[INFO] Pushover notification sent: {message}")
        else:
            print(f"[ERROR] Failed to send Pushover notification: {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception sending Pushover notification: {e}")

# -----------------------------
# LOGGING
# -----------------------------
def log_detection(name, cam_id=0):
    try:
        with open(LOG_CSV, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat(), name, cam_id])
    except Exception as e:
        print(f"[ERROR] Failed to log detection: {e}")

# -----------------------------
# KNOWN FACES LOADING
# -----------------------------
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
            except Exception as e:
                print(f"[WARNING] Could not process {filepath}: {e}")
    print(f"[INFO] Loaded {len(known_encodings)} known faces.")
    return known_encodings, known_names

# -----------------------------
# LOCKFILE HANDLING
# -----------------------------
def check_lockfile():
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r") as f:
                pid = int(f.read().strip())
            if pid and os.path.exists(f"/proc/{pid}"):
                print(f"[WARN] Script already running with PID {pid}")
                sys.exit(1)
            else:
                print("[INFO] Removing stale lockfile.")
                os.remove(LOCKFILE)
        except Exception:
            os.remove(LOCKFILE)
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lockfile():
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

# -----------------------------
# MAIN FACE RECOGNITION FUNCTION
# -----------------------------
def run_face_recognition(preview=True):
    check_lockfile()
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    last_notify = {}
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}")
        remove_lockfile()
        return
    print("[INFO] Starting facial recognition. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame from webcam.")
                break
            small_frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            names_in_frame = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < 0.5:
                        name = known_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                names_in_frame.append(name)
                if name != "Unknown":
                    now = time.time()
                    if name not in last_notify or now - last_notify[name] > NOTIFY_COOLDOWN:
                        send_notification(f"🟢 {name} detected by face recognition")
                        log_detection(name, CAMERA_INDEX)
                        last_notify[name] = now
                        # --- Save snapshot in their folder as imgN.jpg ---
                        person_dir = os.path.join(KNOWN_FACES_DIR, name)
                        os.makedirs(person_dir, exist_ok=True)
                        existing_imgs = [f for f in os.listdir(person_dir) if f.lower().startswith('img') and f.lower().endswith('.jpg')]
                        n = len(existing_imgs) + 1
                        snapshot_path = os.path.join(person_dir, f"img{n}.jpg")
                        cv2.imwrite(snapshot_path, small_frame)
                else:
                    # Optionally log unknowns and save snapshot
                    log_detection("Unknown", CAMERA_INDEX)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{ts}.jpg")
                    cv2.imwrite(unknown_path, small_frame)
                # Draw box and label
                if preview:
                    top, right, bottom, left = [int(v * frame.shape[0] / small_frame.shape[0]) for v in face_location]
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green for known
                    else:
                        color = (0, 0, 255)  # Red for unknown
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if preview:
                cv2.imshow("Facial Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        remove_lockfile()
        print("[INFO] Facial recognition stopped.")

# -----------------------------
# CLI ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='Show video window with annotations')
    args = parser.parse_args()
    run_face_recognition(preview=args.preview)
