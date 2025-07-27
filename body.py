import cv2
from ultralytics import YOLO
import time, requests

PUSHOVER_USER_KEY = "your_user_key"
PUSHOVER_API_TOKEN = "your_api_token"

def send_pushover(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": PUSHOVER_API_TOKEN,
        "user": PUSHOVER_USER_KEY,
        "message": msg,
    })

model = YOLO("yolov5s.pt")
cap = cv2.VideoCapture(0)

last_alert_time = 0
cooldown = 30  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:  # person
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Pushover notification
                now = time.time()
                if now - last_alert_time > cooldown:
                    send_pushover("🚨 Critical Time: Person detected!")
                    last_alert_time = now

    cv2.imshow("Person Detection (YOLOv5)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()