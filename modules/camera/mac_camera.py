import cv2
from config.settings import settings
import time
class MacCamera:
    """
    Handles capturing frames from the MacBook camera using OpenCV.
    """

    def __init__(self):
        self.camera_index = settings.CAMERA_INDEX
        self.save_path = settings.IMAGE_SAVE_PATH

    def capture(self):
        """Capture a single image from the Mac camera."""
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise Exception("❌ Could not access Mac camera")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("❌ Failed to capture image from Mac camera")

        # Save image
        cv2.imwrite(self.save_path, frame)

        if settings.DEBUG:
            print(f"📸 Image captured from Mac camera → {self.save_path}")

        return frame
    def capture_multiple(self, count=6, delay=0.15):
        """Capture multiple frames to improve OCR accuracy."""
        frames = []
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise Exception("Could not access Mac camera")

        for _ in range(count):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            time.sleep(delay)

        cap.release()
        return frames
