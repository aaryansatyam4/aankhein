"""
main.py — FINAL (Speed mode + Ultra-clean terminal + Medium analysis logged)

Features:
- Speed mode: half-resolution frames for faster OCR/BLIP/CLIP
- Forces BLIP and CLIP models to CPU to avoid MPS device errors
- Ultra-clean terminal output (Option D): only shows the concise result block
- Full medium analysis saved as JSON lines to logs/analysis_TIMESTAMP.txt
- Improved product-name extraction (hybrid heuristics)
- Supports debug mode (--debug) and normal voice mode

Usage:
    python main.py --debug
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from collections import Counter

# project settings (expects settings.CAMERA_INDEX, settings.DEVICE, settings.DEBUG)
from config.settings import settings

# Camera and modules (these must exist in your project)
from modules.camera.live_camera import LiveCamera
from modules.ocr.ocr_reader import OCRReader
from modules.caption.blip_caption import BlipCaption
from modules.object_query.clip_query import CLIPQuery
from modules.object_query.product_detector import ProductDetector
from modules.object_query.product_details import ProductDetailsExtractor
from modules.audio.tts_engine import TTSEngine
from modules.audio.stt_engine import STTEngine

# Helpers
import cv2
import numpy as np
import torch

# small CLIP labels (customize if needed)
CLIP_LABELS = [
    "bottle", "phone", "laptop", "book", "hand", "person", "snack",
    "chocolate", "juice", "carton", "can", "packet", "medicine", "cosmetics",
    "remote", "keyboard", "shoe", "watch", "glasses", "camera"
]

# Food/product keywords to boost product-name selection (hybrid)
PRODUCT_KEYWORDS = {
    "aloo", "bhujiya", "chips", "biscuits", "tango", "tropicana", "juice", "milk",
    "dairy", "chocolate", "oreo", "cookie", "namkeen", "masala", "oreo", "kurkure",
    "maggi", "coke", "cola", "sprite", "pepsi", "tiffin", "snack", "packet"
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
            traceback.print_exc()


class VisionGogglesController:
    def __init__(self, speed_mode=True):
        # speed_mode True -> use half-resolution resizing (faster)
        self.speed_mode = speed_mode

        print("🔧 Initializing Vision Goggles (final)...")
        try:
            # camera (live stream) - this class provides last_frame and get_best_frames()
            self.camera = LiveCamera(cam_index=settings.CAMERA_INDEX)
            self.camera.start()

            # vision/audio modules
            self.ocr = OCRReader()
            self.blip = BlipCaption()
            self.clip = CLIPQuery()
            self.product = ProductDetector(self.ocr, self.blip, self.clip)
            self.product_details = ProductDetailsExtractor()
            self.tts = TTSEngine()
            self.stt = STTEngine()

            # logger
            self.logger = SimpleLogger()

            # Force BLIP and CLIP models to CPU to avoid MPS mismatch and improve stability
            try:
                if hasattr(self.blip, "model"):
                    self.blip.model.to("cpu")
                if hasattr(self.clip, "model"):
                    self.clip.model.to("cpu")
            except Exception:
                # ignore if not available
                pass

            # device used in this controller (CPU)
            self.device = "cpu"

            # keep last spoken/result if needed
            self.last_result = None

            print("✅ Initialization complete.")
        except Exception as e:
            print("❌ Initialization failed during init.")
            traceback.print_exc()
            raise e

    # ---------------------
    # Utilities
    # ---------------------
    def _resize_for_speed(self, frame):
        """Resize frame for faster processing (maintain aspect ratio)."""
        if frame is None:
            return None
        if not self.speed_mode:
            return frame
        h, w = frame.shape[:2]
        target_w = 640
        target_h = int(h * (target_w / w))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _ocr_per_frame(self, frames):
        """Run OCR on each frame (uses self.ocr.reader) and return per-frame items, merged text and avg conf."""
        per_frame = []
        all_texts = []
        confs = []
        for f in frames:
            if f is None:
                per_frame.append([])
                continue
            # resize for OCR speed
            img = self._resize_for_speed(f)
            try:
                results = self.ocr.reader.readtext(img)
            except Exception:
                results = []
            frame_txt = []
            for bbox, text, conf in results:
                frame_txt.append((text, float(conf)))
                if conf >= getattr(self.ocr, "min_conf", 0.4):
                    all_texts.append(text.strip())
                    confs.append(float(conf))
            per_frame.append(frame_txt)
        merged = " ".join(all_texts).strip()
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        if not merged and all_texts:
            merged = all_texts[0]
        if not merged:
            merged = "No readable text found."
        return per_frame, merged, avg_conf

    def _get_blip_beams(self, frame, mode="default", top_k=3):
        """Return top-k BLIP captions (best-effort)."""
        try:
            beams = self.blip_beams(frame, mode=mode, num_beams=6, top_k=top_k)
            if beams:
                return beams
            # fallback to single caption
            caption = self.blip.caption(frame, mode=mode)
            return [caption] if caption else []
        except Exception:
            try:
                c = self.blip.caption(frame, mode=mode)
                return [c] if c else []
            except Exception:
                return []

    def blip_beams(self, frame, mode="default", num_beams=5, top_k=3):
        """Internal BLIP beam generator using the blip module's processor & model (safe on CPU)."""
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
            if settings.DEBUG:
                traceback.print_exc()
            return []

    def clip_label_scores(self, frame, labels=CLIP_LABELS, top_k=5):
        """Return top-k CLIP label scores."""
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

    # -----------------------
    # Product name extractor (hybrid heuristics)
    # -----------------------
    def extract_product_name(self, merged_ocr_text, per_frame_texts, blip_caption):
        """
        Smart product name selection:
         - split merged OCR into candidate lines
         - discard lines with stopwords
         - score lines by letter ratio, word count, presence of product keywords, and high confidence tokens
         - fallback to BLIP caption if needed
        """
        stopwords = {"mrp", "price", "batch", "mfg", "manufactured", "expiry", "expiry:", "expiry:", "net", "weight",
                     "ingredients", "address", "fssai", "date", "best", "before", "kg", "g", "ml", "l", "kg.", "₹", "rs", "no"}
        # build candidate lines from per-frame texts first (higher chance near-top)
        candidates = []
        for frame_list in per_frame_texts:
            # make strings from frame tokens
            words = [t for t, c in frame_list if t and len(t.strip()) > 0]
            if words:
                line = " ".join(words).strip()
                candidates.append(line)
        # also add merged lines split by punctuation/newline
        merged_lines = [l.strip() for l in merged_ocr_text.replace("/", " ").replace(",", " ").split() if l.strip()]
        # but prefer multi-word sequences; reconstruct possible lines by sliding window
        # create simple N-gram candidates from merged text (2-4 words)
        merged_words = merged_ocr_text.split()
        for n in (4, 3, 2):
            for i in range(max(0, len(merged_words) - n + 1)):
                seq = " ".join(merged_words[i:i + n]).strip()
                if seq:
                    candidates.append(seq)
        # include blip caption short phrases
        if blip_caption:
            candidates.append(blip_caption)

        # scoring
        scored = []
        for cand in candidates:
            low = cand.lower()
            # discard if contains stopwords majority
            if any(sw in low for sw in ["ingredient", "net weight", "mfg", "batch", "expiry", "address", "fssai"]):
                # skip lines primarily technical
                continue
            # letter ratio
            letters = sum(1 for ch in low if ch.isalpha())
            digits = sum(1 for ch in low if ch.isdigit())
            total = max(1, len(low))
            letter_ratio = letters / total
            words = low.split()
            word_count = len(words)
            # keyword boost
            keyword_boost = 0
            for kw in PRODUCT_KEYWORDS:
                if kw in low:
                    keyword_boost += 1.5
            # penalty if many digits (likely batch number)
            digit_penalty = - (digits / total) * 1.5
            # prefer 1-4 words product names
            wc_bonus = 1.0 if 1 <= word_count <= 4 else 0.5
            score = (letter_ratio * 2.0) + keyword_boost + wc_bonus + digit_penalty
            # small boost for presence in first frames (index)
            # earlier candidates come from per-frame order; no extra compute here
            scored.append((score, cand))

        # choose best candidate
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            # final cleanup: remove stray non-alphanumeric prefix/suffix
            best_clean = best.strip(" -:,.")
            # if best looks like numbers heavy, fallback
            if sum(ch.isdigit() for ch in best_clean) > len(best_clean) * 0.4:
                best_clean = None
        else:
            best_clean = None

        # fallback strategies
        if not best_clean or best_clean.lower().startswith("no readable"):
            # try to pick longest all-letter n-gram from merged words
            words = [w for w in merged_ocr_text.split() if any(ch.isalpha() for ch in w)]
            if words:
                cand = " ".join(words[:4])
                best_clean = cand

        if not best_clean:
            # fallback to BLIP short caption (first 3 words)
            if blip_caption:
                best_clean = " ".join(blip_caption.split()[:4])

        if not best_clean:
            best_clean = "Unknown product"

        return best_clean

    # -----------------------
    # Core command handling (ultra-clean terminal output)
    # -----------------------
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

        # capture frames for OCR if required
        need_ocr = intent_read or intent_details or intent_price or intent_name
        per_frame = []
        merged_text = ""
        avg_conf = 0.0
        if need_ocr:
            count = 10 if intent_details else 6
            frames = self.camera.get_best_frames(count=count, delay=0.10)
            # some frames may be None (camera warming) -> filter
            frames = [f for f in frames if f is not None]
            per_frame, merged_text, avg_conf = self._ocr_per_frame(frames)

        # get latest frame for BLIP/CLIP
        frame = self.camera.last_frame
        if frame is None:
            return None

        # resize frame for speed for these models (do not alter original camera state)
        frame_for_models = self._resize_for_speed(frame)

        # BLIP (beams) and caption
        blip_mode = "product" if (intent_name or intent_details or intent_price) else ("default" if intent_describe else "default")
        blip_beams = self._get_blip_beams(frame_for_models, mode=blip_mode, top_k=3)
        blip_caption = blip_beams[0] if blip_beams else (self.blip.caption(frame_for_models, mode=blip_mode) or "")

        # CLIP labels
        clip_scores = self.clip_label_scores(frame_for_models, labels=CLIP_LABELS, top_k=5)

        # product detector (barcode/OCR/BLIP/CLIP hybrid)
        prod_report = None
        try:
            prod_report = self.product.identify(frame_for_models)
        except Exception:
            prod_report = None

        # product structured details
        details_struct = None
        details_formatted = None
        if intent_details or intent_price or intent_name:
            try:
                details_struct = self.product_details.extract(merged_text)
                details_formatted = self.product_details.format_details(details_struct)
            except Exception:
                details_struct = None
                details_formatted = None

        # final user intent handling (ultra-clean outputs)
        user_output_type = None
        user_output_value = None

        if intent_price:
            price = None
            if details_struct and details_struct.get("mrp"):
                price = details_struct.get("mrp")
            # if not found, maybe OCR had 'MRP: 120/-' with symbols — try regex fallback
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
            # smart product name extraction using hybrid heuristics
            name = None
            # product_detector may return "Product identified: NAME (barcode ...)"
            if isinstance(prod_report, str) and prod_report.startswith("Product identified:"):
                try:
                    name = prod_report.split(":", 1)[1].split("(")[0].strip()
                except Exception:
                    name = prod_report
            if not name:
                name = self.extract_product_name(merged_text, per_frame, blip_caption)
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
            user_output_value = blip_caption

        elif intent_identify:
            # general identify -> try product name -> classify via clip -> blip
            name_guess = None
            if isinstance(prod_report, str) and prod_report.startswith("Product identified:"):
                try:
                    name_guess = prod_report.split(":", 1)[1].split("(")[0].strip()
                except Exception:
                    name_guess = prod_report
            if not name_guess:
                if merged_text and merged_text != "No readable text found.":
                    name_guess = self.extract_product_name(merged_text, per_frame, blip_caption)
            if not name_guess and clip_scores:
                name_guess = clip_scores[0][0]
            if not name_guess:
                name_guess = blip_caption
            user_output_type = "identify"
            user_output_value = name_guess

        else:
            # fallback summary
            parts = []
            if merged_text and merged_text != "No readable text found.":
                parts.append(merged_text)
            if blip_caption:
                parts.append(blip_caption)
            if clip_scores:
                parts.append(f"{clip_scores[0][0]} ({clip_scores[0][1]:.2f})")
            user_output_type = "summary"
            user_output_value = " — ".join(parts) if parts else "No result"

        # build payload for logs
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
            "blip_beams": blip_beams,
            "blip_caption": blip_caption,
            "clip_top": clip_scores,
            "product_detector": prod_report,
            "product_details_struct": details_struct,
            "product_details_formatted": details_formatted,
            "user_output_type": user_output_type,
            "user_output_value": user_output_value,
        }

        # write to logs (full analysis)
        try:
            self.logger.write(payload)
        except Exception:
            traceback.print_exc()

        # speak & print only the ultra-clean result block
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

            # concise TTS
            try:
                self.tts.speak(out)
            except Exception:
                # ignore TTS failures
                pass

            # store last result
            self.last_result = {"type": user_output_type, "value": user_output_value}
        except Exception:
            traceback.print_exc()

        return self.last_result

    # -----------------------
    # Loops
    # -----------------------
    def run_debug(self):
        # Debug mode accepts typed commands but CLI remains ultra-clean
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
            self.camera.stop()

    def run_normal(self):
        # Voice-based loop
        self.tts.speak("Vision goggles ready.")
        try:
            while True:
                # wake-word -> listen
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
            traceback.print_exc()
        finally:
            self.camera.stop()


if __name__ == "__main__":
    controller = VisionGogglesController(speed_mode=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        controller.run_debug()
    else:
        controller.run_normal()
