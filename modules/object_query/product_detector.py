import cv2
from pyzbar import pyzbar
from config.settings import settings

class ProductDetector:
    """
    Hybrid product recognition system:
     1. Barcode scanning
     2. OCR-based name detection
     3. BLIP product caption
     4. CLIP product classes
    """

    def __init__(self, ocr, blip, clip):
        self.ocr = ocr
        self.blip = blip
        self.clip = clip

    def detect_barcodes(self, frame):
        decoded = pyzbar.decode(frame)
        if not decoded:
            return None
        results = []
        for d in decoded:
            barcode = d.data.decode("utf-8")
            results.append(barcode)
        return results

    def lookup_product(self, code):
        """
        Simple local database (expandable).
        """
        sample_db = {
            "8901030701234": "Dairy Milk Chocolate 52g",
            "8902080123456": "Tropicana Orange Juice 1L",
            "012345678905": "Coca-Cola 500ml",
        }
        return sample_db.get(code, None)

    def identify(self, frame):
        """
        Full pipeline:
        1. Try barcode
        2. Try OCR-based packaging name
        3. Try BLIP product caption
        4. Try CLIP product type
        """

        # 1. Barcode detection
        barcodes = self.detect_barcodes(frame)
        if barcodes:
            code = barcodes[0]
            product_name = self.lookup_product(code)
            if product_name:
                return f"Product identified: {product_name} (barcode {code})"
            else:
                return f"Barcode detected ({code}), but product not in local DB."

        # 2. OCR extraction for keywords
        text = self.ocr.read(frame).lower()
        if any(k in text for k in ["juice", "milk", "choco", "cream", "soap", "oil"]):
            return f"Product text detected: {text}"

        # 3. BLIP product caption
        caption = self.blip.caption(frame, mode="product")
        if caption:
            return f"Product description: {caption}"

        # 4. CLIP fallback detection
        labels = ["snack", "drink", "juice", "chocolate", "bottle", "carton", "packet"]
        best = None
        best_score = 0

        for label in labels:
            res = self.clip.query(frame, label)
            conf = float(res.split("Confidence: ")[-1])
            if conf > best_score:
                best_score = conf
                best = label

        return f"Product type seems like: {best} ({best_score:.2f})"
