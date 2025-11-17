import pyttsx3
from config.settings import settings

class TTSEngine:
    """
    Offline Text-to-Speech engine using pyttsx3.
    """

    def __init__(self):
        print("🔊 Initializing TTS Engine...")

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", settings.TTS_RATE)
        self.engine.setProperty("volume", settings.TTS_VOLUME)

    def speak(self, text: str):
        """
        Speaks out text clearly via Mac speakers.
        """
        if settings.DEBUG:
            print(f"🗣️ Speaking: {text}")

        self.engine.say(text)
        self.engine.runAndWait()
