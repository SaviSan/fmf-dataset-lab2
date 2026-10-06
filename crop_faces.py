import cv2, os, sys
from datetime import datetime

src, class_name = sys.argv[1], sys.argv[2]
dst = f"fmfdataset/raw/{class_name}"
os.makedirs(dst, exist_ok=True)

cascades = [cv2.CascadeClassifier(cv2.data.haarcascades + f)
            for f in ["haarcascade_frontalface_default.xml",
                      "haarcascade_frontalface_alt2.xml"]]
settings = [(1.1, 5, 60), (1.05, 3, 50), (1.03, 2, 40)]   # scaleFactor, minNeighbors, minSize

def find_face(gray):
    for c in cascades:
        for sf, mn, ms in settings:
            faces = c.detectMultiScale(gray, sf, mn, minSize=(ms, ms))
            if len(faces) > 0:
                return max(faces, key=lambda f: f[2] * f[3])
    return None

count = len(os.listdir(dst))
for fn in os.listdir(src):
    img = cv2.imread(os.path.join(src, fn))
    if img is None:
        continue
    s = 1000 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    face = find_face(gray)
    if face is None:
        print("face na:", fn)
        continue
    x, y, w, h = face
    pw, ph = int(w * 0.2), int(h * 0.2)      # 20% padding
    crop = img[max(0, y - ph):min(img.shape[0], y + h + ph),
               max(0, x - pw):min(img.shape[1], x + w + pw)]
    count += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{dst}/{class_name}_{ts}_{count}.jpg", crop)
print(count, "saved")