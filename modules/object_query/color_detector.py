import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorDetector:
    COLORS = {
        "red": ([0, 0, 60], [80, 80, 255]),
        "green": ([0, 60, 0], [80, 255, 80]),
        "blue": ([60, 0, 0], [255, 80, 80]),
        "black": ([0, 0, 0], [50, 50, 50]),
        "white": ([200, 200, 200], [255, 255, 255]),
        "yellow": ([0, 180, 180], [80, 255, 255]),
        "brown": ([20, 40, 80], [90, 120, 180]),
        "gray": ([80, 80, 80], [180, 180, 180]),
    }

    def detect_dominant_color(self, frame):
        img = cv2.resize(frame, (64, 64)) 
        img = img.reshape((-1, 3))

        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        kmeans.fit(img)

        dom = kmeans.cluster_centers_.astype(int)

        best_color = "unknown"
        best_score = 0

        for name, (low, high) in self.COLORS.items():
            low = np.array(low)
            high = np.array(high)

            for c in dom:
                if np.all(c >= low) and np.all(c <= high):
                    score = np.linalg.norm(c)
                    if score > best_score:
                        best_score = score
                        best_color = name

        return best_color
