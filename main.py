"""
main.py — Final (ESP32 default, Speed mode, Ultra-clean terminal, Medium analysis logged)

- Uses modules/camera/esp32_camera.ESP32Camera by default
- Speed mode (resizes frames to ~640px width)
- Forces BLIP & CLIP to CPU (avoids MPS device mismatch)
- Ultra-clean terminal output (only concise result block)
- Full analysis logged to logs/analysis_TIMESTAMP.txt (one JSON per request)
- Supports --debug (typed commands) and normal voice mode
- Offline YOLO integrated for scene description (models/yolo/yolov8n.pt)
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime

from config.settings import settings

# Camera & modules (must exist in your project)
if settings.USE_ESP32:
    from modules.camera.esp32_camera import ESP32Camera as Camera
else:
    from modules.camera.live_camera import LiveCamera as Camera
from modules.ocr.ocr_reader import OCRReader
from modules.caption.blip_caption import BlipCaption
from modules.object_query.clip_query import CLIPQuery
from modules.object_query.product_detector import ProductDetector
from modules.object_query.product_details import ProductDetailsExtractor
from modules.audio.tts_engine import TTSEngine
from modules.audio.stt_engine import STTEngine
from modules.object_query.color_detector import ColorDetector


# NEW: YOLO detector (offline)
from modules.object_query.yolo_detector import YOLODetector

# Helpers
import cv2

# short CLIP label set (customize)
CLIP_LABELS = [
    "bottle", "phone", "laptop", "book", "hand", "person", "snack",
    "chocolate", "juice", "carton", "can", "packet", "medicine", "cosmetics",
    "remote", "keyboard", "shoe", "watch", "glasses", "camera"
]

# product keyword boosts (hybrid name extraction)
PRODUCT_KEYWORDS = {
    "aloo", "bhujiya", "chips", "biscuits", "juice", "milk", "dairy", "chocolate",
    "cookie", "namkeen", "masala", "kurkure", "maggi", "coke", "cola", "pepsi",
    "snack", "packet"
}


class SimpleLogger:
    def __init__(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(folder, f"analysis_{ts}.txt")

    def write(self, obj: dict):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            # avoid noisy terminal output; log trace to file if needed
            try:
                with open(self.path + ".err", "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc() + "\n")
            except Exception:
                pass


class VisionGogglesController:
    def __init__(self, speed_mode=True):
        self.speed_mode = speed_mode
        # initialize core components
        try:
            self.camera = Camera()
            self.camera.start()
            self.camera_start_time = time.time()
        except Exception:
            raise

        self.ocr = OCRReader()
        self.blip = BlipCaption()
        self.clip = CLIPQuery()
        self.product = ProductDetector(self.ocr, self.blip, self.clip)
        self.product_details = ProductDetailsExtractor()
        self.tts = TTSEngine()
        self.stt = STTEngine()
        self.logger = SimpleLogger()

        # Initialize offline YOLO detector (models/yolo/yolov8n.pt expected)
        try:
            self.yolo = YOLODetector(model_path="models/yolo/yolov8n.pt")
            
        except Exception:
            # If model missing or fails, keep going without YOLO
            if settings.DEBUG:
                print("YOLO initialization failed:", traceback.format_exc())
            self.yolo = None
        
        self.color_detector = ColorDetector()

        # Force BLIP & CLIP to CPU to avoid MPS/CPU mismatch and crashes
        try:
            if hasattr(self.blip, "model"):
                self.blip.model.to("cpu")
        except Exception:
            pass
        try:
            if hasattr(self.clip, "model"):
                self.clip.model.to("cpu")
        except Exception:
            pass

        self.device = "cpu"
        self.last_result = None

    # -------------------------
    # Utilities
    # -------------------------
    def _resize_for_speed(self, frame):
        if frame is None:
            return None
        if not self.speed_mode:
            return frame
        h, w = frame.shape[:2]
        target_w = 640
        if w <= target_w:
            return frame
        target_h = int(h * (target_w / w))
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _ocr_per_frame(self, frames):
        per_frame = []
        all_texts = []
        confs = []
        for f in frames:
            if f is None:
                per_frame.append([])
                continue
            img = self._resize_for_speed(f)
            try:
                results = self.ocr.reader.readtext(img)
            except Exception:
                results = []
            frame_txt = []
            for bbox, text, conf in results:
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                frame_txt.append((text, conf_f))
                if conf_f >= getattr(self.ocr, "min_conf", 0.4):
                    all_texts.append(text.strip())
                    confs.append(conf_f)
            per_frame.append(frame_txt)
        merged = " ".join(all_texts).strip()
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        if not merged and all_texts:
            merged = all_texts[0]
        if not merged:
            merged = "No readable text found."
        return per_frame, merged, avg_conf

    def _get_blip_candidates(self, frame, mode="default", top_k=3):
        try:
            # attempt beam-based candidates using module's generate helper if present
            candidates = []
            if hasattr(self, "blip_beams"):
                # use local safe wrapper
                candidates = self.blip_beams(frame, mode=mode, num_beams=6, top_k=top_k)
            if candidates:
                return candidates
            # fallback to single caption
            cap = self.blip.caption(frame, mode=mode)
            return [cap] if cap else []
        except Exception:
            try:
                cap = self.blip.caption(frame, mode=mode)
                return [cap] if cap else []
            except Exception:
                return []

    def blip_beams(self, frame, mode="default", num_beams=5, top_k=3):
        # safe CPU beam generation
        try:
            prompts = {
                "default": "A photo showing",
                "object": "A photo of the object being held which is",
                "product": "A photo of a packaged product which is",
            }
            prompt = prompts.get(mode, prompts["default"])
            from PIL import Image
            pil = Image.fromarray(self._resize_for_speed(frame))
            processor = self.blip.processor
            model = self.blip.model
            inputs = processor(images=pil, return_tensors="pt").to(self.device)
            text_inputs = processor.tokenizer(prompt, return_tensors="pt").to(self.device)
            out_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=text_inputs.input_ids,
                num_beams=num_beams,
                num_return_sequences=min(top_k, num_beams),
                max_length=40,
                repetition_penalty=1.1,
            )
            captions = []
            for oid in out_ids:
                cap = processor.decode(oid, skip_special_tokens=True)
                if cap.lower().startswith(prompt.lower()):
                    cap = cap[len(prompt):].strip()
                captions.append(cap)
            # unique preserve order
            seen = set()
            unique_caps = []
            for c in captions:
                if c not in seen:
                    unique_caps.append(c)
                    seen.add(c)
                if len(unique_caps) >= top_k:
                    break
            return unique_caps
        except Exception:
            # do not print to terminal; return empty for silent operation
            if settings.DEBUG:
                traceback.print_exc()
            return []

    def clip_label_scores(self, frame, labels=CLIP_LABELS, top_k=5):
        try:
            processor = self.clip.processor
            model = self.clip.model
            from PIL import Image
            pil = Image.fromarray(self._resize_for_speed(frame))
            inputs = processor(text=labels, images=pil, return_tensors="pt", padding=True).to(self.device)
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
            label_scores = list(zip(labels, probs.tolist()))
            label_scores.sort(key=lambda x: x[1], reverse=True)
            return label_scores[:top_k]
        except Exception:
            if settings.DEBUG:
                traceback.print_exc()
            return []

    # -------------------------
    # product name extraction (hybrid heuristics)
    # -------------------------
    def extract_product_name(self, merged_ocr_text, per_frame_texts, blip_caption):
        stopwords = {"mrp", "price", "batch", "mfg", "manufactured", "expiry", "net", "weight",
                     "ingredients", "address", "fssai", "date", "best", "before", "kg", "g", "ml", "l", "rs", "no"}
        candidates = []
        for frame_list in per_frame_texts:
            words = [t for t, c in frame_list if t and len(t.strip()) > 0]
            if words:
                candidates.append(" ".join(words).strip())
        merged_words = merged_ocr_text.split()
        for n in (4, 3, 2):
            for i in range(max(0, len(merged_words) - n + 1)):
                seq = " ".join(merged_words[i:i + n]).strip()
                if seq:
                    candidates.append(seq)
        if blip_caption:
            candidates.append(blip_caption)
        scored = []
        for cand in candidates:
            low = cand.lower()
            if any(sw in low for sw in ["ingredient", "net weight", "mfg", "batch", "expiry", "address", "fssai"]):
                continue
            letters = sum(1 for ch in low if ch.isalpha())
            digits = sum(1 for ch in low if ch.isdigit())
            total = max(1, len(low))
            letter_ratio = letters / total
            words = low.split()
            word_count = len(words)
            keyword_boost = 0
            for kw in PRODUCT_KEYWORDS:
                if kw in low:
                    keyword_boost += 1.5
            digit_penalty = - (digits / total) * 1.5
            wc_bonus = 1.0 if 1 <= word_count <= 4 else 0.5
            score = (letter_ratio * 2.0) + keyword_boost + wc_bonus + digit_penalty
            scored.append((score, cand))
        best_clean = None
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            best_clean = best.strip(" -:,.")
            if sum(ch.isdigit() for ch in best_clean) > len(best_clean) * 0.4:
                best_clean = None
        if not best_clean or (isinstance(best_clean, str) and best_clean.lower().startswith("no readable")):
            words = [w for w in merged_ocr_text.split() if any(ch.isalpha() for ch in w)]
            if words:
                best_clean = " ".join(words[:4])
        if not best_clean and blip_caption:
            best_clean = " ".join(blip_caption.split()[:4])
        if not best_clean:
            best_clean = "Unknown product"
        return best_clean

    # -------------------------
    # handle a single user command (typed or voice)
    # -------------------------
    def handle_command(self, command_text):
        if not command_text:
            return None
        cmd = command_text.lower().strip()

        # intents
        intent_price = any(x in cmd for x in ["mrp", "price", "cost", "how much", "what is the mrp", "what is the price"])
        intent_name = any(x in cmd for x in ["what is this product", "which product", "name this", "identify this product", "what product", "what is this product?"])
        intent_details = any(x in cmd for x in ["details", "product details", "read details", "what are the details"])
        intent_read = any(x in cmd for x in ["read", "text", "scan", "read this"])
        intent_describe = any(x in cmd for x in ["describe", "what do you see", "tell me about"])
        intent_identify = any(x in cmd for x in ["what is", "identify", "which", "what am", "is this", "what is this"])

        need_ocr = intent_read or intent_details or intent_price or intent_name
        per_frame = []
        merged_text = ""
        avg_conf = 0.0

        # Capture a single frame first for all operations
        frame = self.camera.last_frame

        if frame is None:
            # Wait a moment for the camera stream to provide a frame
            time.sleep(0.5)
            frame = self.camera.last_frame

        if frame is None:
            if settings.DEBUG:
                print("Camera did not provide a frame.")
            return None

        if need_ocr:
            count = 10 if intent_details else 6
            frames = self.camera.get_best_frames(count=count, delay=0.10)
            frames = [f for f in frames if f is not None]
            per_frame, merged_text, avg_conf = self._ocr_per_frame(frames)

        frame_for_models = self._resize_for_speed(frame)

        blip_mode = "product" if (intent_name or intent_details or intent_price) else "default"
        blip_candidates = self._get_blip_candidates(frame_for_models, mode=blip_mode, top_k=3)
        blip_caption = blip_candidates[0] if blip_candidates else (self.blip.caption(frame_for_models, mode=blip_mode) or "")

        clip_scores = self.clip_label_scores(frame_for_models, labels=CLIP_LABELS, top_k=5)

        prod_report = None
        try:
            prod_report = self.product.identify(frame_for_models)
        except Exception:
            prod_report = None

        details_struct = None
        details_formatted = None
        if intent_details or intent_price or intent_name:
            try:
                details_struct = self.product_details.extract(merged_text)
                details_formatted = self.product_details.format_details(details_struct)
            except Exception:
                details_struct = None
                details_formatted = None

        # Run YOLO only for relevant intents (describe / identify / name)
        yolo_detections = []
        try:
            if self.yolo and (intent_describe or intent_identify or intent_name):
                yolo_detections = self.yolo.detect(frame_for_models)
        except Exception:
            yolo_detections = []
            if settings.DEBUG:
                traceback.print_exc()

        user_output_type = None
        user_output_value = None

        if intent_price:
            price = None
            if details_struct and details_struct.get("mrp"):
                price = details_struct.get("mrp")
            if not price and merged_text:
                import re
                m = re.search(r"(?:mrp|price|₹|rs)[^\d]*([0-9]{1,6})", merged_text.lower())
                if m:
                    price = m.group(1)
            if price:
                user_output_type = "price"
                user_output_value = f"₹{price}"
            else:
                user_output_type = "price"
                user_output_value = "Price not found"

        elif intent_name:
            name = None
            if isinstance(prod_report, str) and prod_report.startswith("Product identified:"):
                try:
                    name = prod_report.split(":", 1)[1].split("(")[0].strip()
                except Exception:
                    name = prod_report
            if not name:
                name = self.extract_product_name(merged_text, per_frame, blip_caption)
            # If YOLO found a clear class with high score, prefer it as short hint (optional)
            if not name and yolo_detections:
                name = yolo_detections[0][0]
            user_output_type = "name"
            user_output_value = name

        elif intent_details:
            user_output_type = "details"
            if details_formatted:
                user_output_value = details_formatted
            elif prod_report:
                user_output_value = prod_report
            else:
                user_output_value = blip_caption or "No details found"

        elif intent_read:
            user_output_type = "read"
            user_output_value = merged_text

        elif intent_describe:
            user_output_type = "describe"
            # Medium style: BLIP + YOLO detections with confidences
            if yolo_detections:
                det_text = ", ".join([f"{name} ({score*100:.0f}%)" for name, score in yolo_detections])
                if blip_caption:
                    dominant_color = "unknown"
                    if yolo_detections and yolo_detections[0][0] == "person":
                        dominant_color = self.color_detector.detect_dominant_color(frame_for_models)

                    if yolo_detections:
                        det_text = ", ".join([f"{name} ({score*100:.0f}%)" for name, score in yolo_detections])
                        if dominant_color != "unknown":
                            user_output_value = f"{blip_caption}. I also detected: {det_text}. The person appears to be wearing {dominant_color} clothing."
                        else:
                            user_output_value = f"{blip_caption}. I also detected: {det_text}."
                    else:
                        user_output_value = blip_caption

                else:
                    user_output_value = f"I detected: {det_text}."
            else:
                user_output_value = blip_caption or "No notable objects detected."

        elif intent_identify:
            name_guess = None
            if isinstance(prod_report, str) and prod_report.startswith("Product identified:"):
                try:
                    name_guess = prod_report.split(":", 1)[1].split("(")[0].strip()
                except Exception:
                    name_guess = prod_report
            if not name_guess:
                if merged_text and merged_text != "No readable text found.":
                    name_guess = self.extract_product_name(merged_text, per_frame, blip_caption)
            if not name_guess and yolo_detections:
                name_guess = yolo_detections[0][0]
            if not name_guess and clip_scores:
                name_guess = clip_scores[0][0]
            if not name_guess:
                name_guess = blip_caption
            user_output_type = "identify"
            user_output_value = name_guess

        else:
            parts = []
            if merged_text and merged_text != "No readable text found.":
                parts.append(merged_text)
            if blip_caption:
                parts.append(blip_caption)
            if clip_scores:
                parts.append(f"{clip_scores[0][0]} ({clip_scores[0][1]:.2f})")
            if yolo_detections:
                parts.append(", ".join([f"{n}({s*100:.0f}%)" for n, s in yolo_detections[:3]]))
            user_output_type = "summary"
            user_output_value = " — ".join(parts) if parts else "No result"

        payload = {
            "timestamp": datetime.now().isoformat(),
            "command": cmd,
            "intent": {
                "price": intent_price,
                "name": intent_name,
                "details": intent_details,
                "read": intent_read,
                "describe": intent_describe,
                "identify": intent_identify,
            },
            "ocr_avg_conf": avg_conf,
            "merged_ocr": merged_text,
            "per_frame_ocr": per_frame,
            "blip_candidates": blip_candidates,
            "blip_caption": blip_caption,
            "clip_top": clip_scores,
            "yolo": yolo_detections,
            "product_detector": prod_report,
            "product_details_struct": details_struct,
            "product_details_formatted": details_formatted,
            "user_output_type": user_output_type,
            "user_output_value": user_output_value,
        }

        try:
            self.logger.write(payload)
        except Exception:
            pass

        try:
            if user_output_type == "price":
                out = f"Price: {user_output_value}"
            elif user_output_type == "name":
                out = f"Product: {user_output_value}"
            elif user_output_type == "details":
                out = f"Details: {user_output_value}"
            elif user_output_type == "read":
                out = f"OCR: {user_output_value}"
            elif user_output_type == "describe":
                out = f"Description: {user_output_value}"
            elif user_output_type == "identify":
                out = f"Identify: {user_output_value}"
            else:
                out = f"{user_output_value}"

            # ULTRA-CLEAN terminal output
            print("\n=== RESULT ===")
            print(out)
            print("=== END ===\n")

            try:
                self.tts.speak(out)
            except Exception:
                pass

            self.last_result = {"type": user_output_type, "value": user_output_value}
        except Exception:
            pass

        return self.last_result

    # -------------------------
    # loops
    # -------------------------
    def run_debug(self):
        self.tts.speak("Debug mode enabled. Type your command.")
        try:
            while True:
                cmd = input("\nEnter command: ").strip()
                if not cmd:
                    continue
                if cmd.lower() in ("exit", "quit", "stop"):
                    self.tts.speak("Exiting debug mode.")
                    break
                self.handle_command(cmd)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if hasattr(self.camera, 'stop'):
                    self.camera.stop()
            except Exception:
                pass

    def run_normal(self):
        self.tts.speak("Vision goggles ready.")
        try:
            while True:
                self.stt.listen_for_wake_word()
                self.tts.speak("Yes, what would you like me to do?")
                cmd = self.stt.listen("Listening for your command.")
                if not cmd:
                    continue
                if cmd.lower().strip() in ("exit", "quit", "stop"):
                    self.tts.speak("Shutting down.")
                    break
                self.handle_command(cmd)
        except Exception:
            pass
        finally:
            try:
                self.camera.stop()
            except Exception:
                pass


if __name__ == "__main__":
    controller = VisionGogglesController(speed_mode=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        controller.run_debug()
    else:
        controller.run_normal()
