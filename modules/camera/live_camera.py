import cv2
import time
import threading

class LiveCamera:
    """
    Continuously captures frames from webcam in the background.
    For blind-friendly real-time capture.
    """

    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.last_frame = None
        self.running = False
        self.thread = None

    def _stream(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame

    def start(self):
        """Starts background frame streaming."""
        self.running = True
        self.thread = threading.Thread(target=self._stream, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops camera stream."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

    def get_best_frames(self, count=6, delay=0.12):
        """
        Returns N latest frames in quick succession.
        Perfect for multi-shot OCR.
        """
        frames = []

        for _ in range(count):
            frames.append(self.last_frame)
            time.sleep(delay)

        return frames
