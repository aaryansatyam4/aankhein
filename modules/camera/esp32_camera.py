import requests
import numpy as np
import cv2
from config.settings import settings

class ESP32Camera:
    """
    Captures images from ESP32-CAM through HTTP stream.
    """

    def __init__(self):
        self.url = settings.ESP32_URL
        self.save_path = settings.IMAGE_SAVE_PATH

    def capture(self):
        """Capture image from ESP32-CAM."""
        try:
            response = requests.get(self.url, timeout=5)

            if response.status_code != 200:
                raise Exception("❌ Failed to get response from ESP32 camera")

            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                raise Exception("❌ Invalid image received from ESP32")

            # Save image
            cv2.imwrite(self.save_path, frame)

            if settings.DEBUG:
                print(f"📡 Image captured from ESP32 → {self.save_path}")

            return frame

        except Exception as e:
            raise Exception(f"❌ ESP32 capture error: {e}")
