import re

class ProductDetailsExtractor:
    """
    Extracts full product details from OCR text:
    - MRP / Price
    - Expiry date
    - Manufacturing date
    - Batch number
    - Net weight / volume
    - Ingredients
    """

    def __init__(self):
        pass

    def extract(self, text):
        text_lower = text.lower()

        details = {
            "mrp": None,
            "expiry": None,
            "mfg": None,
            "batch": None,
            "net_weight": None,
            "ingredients": None,
        }

        # -----------------------------
        # 1. Extract MRP / Price
        # -----------------------------
        mrp_patterns = [
            r"mrp[^0-9]*([0-9]+)",
            r"price[^0-9]*([0-9]+)",
            r"rs[^0-9]*([0-9]+)",
            r"₹\s*([0-9]+)"
        ]
        for pattern in mrp_patterns:
            m = re.search(pattern, text_lower)
            if m:
                details["mrp"] = m.group(1)
                break

        # -----------------------------
        # 2. Extract Expiry Date
        # -----------------------------
        exp_patterns = [
            r"exp[^0-9]*([0-9/ -]+)",
            r"expiry[^0-9]*([0-9/ -]+)",
            r"use before[^:]*[: ]*([0-9a-zA-Z/ -]+)",
            r"best before[^:]*[: ]*([0-9a-zA-Z/ -]+)"
        ]
        for pattern in exp_patterns:
            m = re.search(pattern, text_lower)
            if m:
                details["expiry"] = m.group(1)
                break

        # -----------------------------
        # 3. Manufacturing Date
        # -----------------------------
        mfg_patterns = [
            r"mfg[^0-9]*([0-9/ -]+)",
            r"manufactured[^0-9]*([0-9/ -]+)",
        ]
        for pattern in mfg_patterns:
            m = re.search(pattern, text_lower)
            if m:
                details["mfg"] = m.group(1)
                break

        # -----------------------------
        # 4. Batch Number
        # -----------------------------
        batch_patterns = [
            r"batch[^a-zA-Z0-9]*([a-zA-Z0-9-]+)"
        ]
        for pattern in batch_patterns:
            m = re.search(pattern, text_lower)
            if m:
                details["batch"] = m.group(1)
                break

        # -----------------------------
        # 5. Net Weight / Volume
        # -----------------------------
        weight_patterns = [
            r"([0-9]+ ?(g|kg|ml|l))",
            r"net weight[^0-9]*([0-9]+ ?(g|kg|ml|l))"
        ]
        for pattern in weight_patterns:
            m = re.search(pattern, text_lower)
            if m:
                details["net_weight"] = m.group(1)
                break

        # -----------------------------
        # 6. Ingredients
        # -----------------------------
        ing_patterns = [
            r"ingredients[^:]*[: ]*(.*)"
        ]
        for pattern in ing_patterns:
            m = re.search(pattern, text, flags=re.I)
            if m:
                details["ingredients"] = m.group(1).strip()
                break

        return details

    def format_details(self, details):
        output = []

        if details["mrp"]:
            output.append(f"MRP: ₹{details['mrp']}")

        if details["expiry"]:
            output.append(f"Expiry Date: {details['expiry']}")

        if details["mfg"]:
            output.append(f"Manufacturing Date: {details['mfg']}")

        if details["batch"]:
            output.append(f"Batch Number: {details['batch']}")

        if details["net_weight"]:
            output.append(f"Net Weight: {details['net_weight']}")

        if details["ingredients"]:
            output.append(f"Ingredients: {details['ingredients']}")

        if not output:
            return "No product details could be extracted."

        return " — ".join(output)
