import threading
import cv2

class ThreadedCamera:
    def __init__(self, rtsp_url):
        self.capture = cv2.VideoCapture(rtsp_url)
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.capture.release()
