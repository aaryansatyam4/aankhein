import easyocr
from config.settings import settings
import numpy as np
import cv2

class OCRReader:
    """
    Handles offline text extraction using EasyOCR.
    """

    def __init__(self):
        print("📖 Initializing OCR Reader...")
        self.reader = easyocr.Reader(
            settings.OCR_LANG,
            gpu=(settings.DEVICE == "cuda")
        )
        self.min_conf = settings.OCR_MIN_CONFIDENCE

    def read(self, frame):
        """
        Extract text from the frame using EasyOCR.
        Returns a clean combined string.
        """

        if isinstance(frame, str):
            # if path is passed instead of image
            frame = cv2.imread(frame)

        results = self.reader.readtext(frame)

        extracted = []
        for bbox, text, confidence in results:
            if confidence >= self.min_conf:
                extracted.append(text)

        if not extracted:
            return "No readable text found."

        final_text = " ".join(extracted)

        if settings.DEBUG:
            print(f"📝 OCR Extracted Text: {final_text}")

        return final_text
    
    def read_multiple(self, frames):
        """
        Run OCR on multiple frames & merge results.
        """
        all_texts = []

        for frame in frames:
            results = self.reader.readtext(frame)
            for _, text, conf in results:
                if conf >= self.min_conf:
                    all_texts.append(text)

        if not all_texts:
            return "No readable text found."

        # Voting system: most frequent words
        from collections import Counter
        counts = Counter(all_texts)

        best_text = counts.most_common(1)[0][0]  # highest vote

        return best_text
