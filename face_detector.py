import cv2

class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w], (x, y, x+w, y+h)
        return None, None
