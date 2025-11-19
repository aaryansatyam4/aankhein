import requests
import threading
import time
import cv2
import numpy as np

class ESP32Camera:
    """
    Works with ESP32-CAM streams like:
    http://IP:81/stream  (multipart/x-mixed-replace)
    """

    def __init__(self, url=None):
        from config.settings import settings
        self.url = url or settings.ESP32_URL
        self.running = False
        self.thread = None
        self.last_frame = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None

    def _reader(self):
        """
        Reads multipart MJPEG stream manually.
        Compatible with ESP32-CAM.
        """
        print("📡 Connecting to ESP32 stream:", self.url)

        while self.running:
            try:
                r = requests.get(self.url, stream=True, timeout=5)

                bytes_buffer = bytes()

                for chunk in r.iter_content(chunk_size=1024):
                    if not self.running:
                        break

                    bytes_buffer += chunk
                    a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
                    b = bytes_buffer.find(b'\xff\xd9')  # JPEG end

                    if a != -1 and b != -1:
                        jpg = bytes_buffer[a:b+2]
                        bytes_buffer = bytes_buffer[b+2:]

                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.last_frame = frame

            except Exception as e:
                print("⚠️ ESP32 reconnecting in 1 sec:", e)
                time.sleep(1)

    def get_best_frames(self, count=6, delay=0.1):
        frames = []
        for _ in range(count):
            frames.append(None if self.last_frame is None else self.last_frame.copy())
            time.sleep(delay)
        return frames
