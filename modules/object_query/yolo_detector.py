from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, model_path="models/yolo/yolov8n.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")

        # Load model completely offline
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.3):
        if frame is None:
            return []

        results = self.model.predict(
            frame,
            conf=conf,
            verbose=False
        )

        dets = []
        r = results[0]

        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            name = r.names[cls]
            score = float(box.conf.cpu().numpy()[0])
            dets.append((name, score))

        return dets
