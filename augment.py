import cv2, os, sys
from datetime import datetime

class_name = sys.argv[1]                      # usage: python capture.py test
save_dir = f"fmfdataset/raw/{class_name}"
os.makedirs(save_dir, exist_ok=True)
count = len(os.listdir(save_dir))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not open the webcam")
        break

    clean = frame.copy()                      # copy without rectangles, used for cropping
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Photos: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 32:                             # spacebar
        if len(faces) == 0:
            print("No face detected, please try again")
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
        pw, ph = int(w * 0.2), int(h * 0.2)                  # 20% padding
        x1, y1 = max(0, x - pw), max(0, y - ph)
        x2, y2 = min(clean.shape[1], x + w + pw), min(clean.shape[0], y + h + ph)

        count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{save_dir}/{class_name}_{ts}_{count}.jpg", clean[y1:y2, x1:x2])
        print("Saved photo", count)

cap.release()
cv2.destroyAllWindows()