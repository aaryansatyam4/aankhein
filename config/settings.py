import torch

class Settings:
    """
    Global configuration for the Vision Goggles Project.
    All modules read from here to keep system consistent.
    """

    # ----------------------------
    # DEVICE SETTINGS
    # ----------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # CAMERA SETTINGS
    # ----------------------------
    USE_ESP32 = False        # False → use Mac camera | True → use ESP32 stream
    CAMERA_INDEX = 0         # Mac camera index
    ESP32_URL = "http://192.168.4.1/capture"  # ESP32 stream (future use)

    IMAGE_SAVE_PATH = "captured/frame.jpg"

    # ----------------------------
    # OCR SETTINGS
    # ----------------------------
    OCR_LANG = ["en"]        # languages for EasyOCR
    OCR_MIN_CONFIDENCE = 0.5 # ignore weak OCR reads

    # ----------------------------
    # BLIP (Captioning) SETTINGS
    # ----------------------------
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

    # ----------------------------
    # CLIP (Object Query) SETTINGS
    # ----------------------------
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

    # ----------------------------
    # AUDIO / TTS SETTINGS
    # ----------------------------
    TTS_RATE = 165           # speaking speed
    TTS_VOLUME = 1.0         # max volume

        # Speech-to-text (vosk)
    VOSK_MODEL_PATH = "models/vosk/vosk-model-small-en-us-0.15"


    # ----------------------------
    # DEBUG
    # ----------------------------
    DEBUG = True


settings = Settings()
