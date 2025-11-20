import pyttsx3
import time
from config.settings import settings

class TTSEngine:
    """
    Stable pyttsx3 TTS engine for macOS.
    Fixes: sentence cutting, stopping mid-way, partial speech.
    """

    def __init__(self):
        self._new_engine()

    def _new_engine(self):
        """Recreates engine for every use (macOS fix)."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", settings.TTS_RATE)
            self.engine.setProperty("volume", settings.TTS_VOLUME)

            # EXTRA FIX: flush all existing queued utterances
            self.engine.stop()
        except Exception as e:
            print("TTS Init Error:", e)
            self.engine = None

    def speak(self, text: str):
        if not text:
            return

        if settings.DEBUG:
            print(f"🗣 Speaking: {text}")

        # FULL FIX: recreate engine BEFORE every speak
        self._new_engine()

        try:
            self.engine.say(text)

            # EXTRA FIX: ensure synchronous blocking
            self.engine.runAndWait()

            # EXTRA FIX: wait a tiny bit to flush buffers
            time.sleep(0.08)

            # EXTRA FIX: stop engine (prevents truncation on next speak)
            self.engine.stop()

        except Exception as e:
            print("TTS Speak Error:", e)
