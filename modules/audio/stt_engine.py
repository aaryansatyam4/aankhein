import sounddevice as sd
import vosk
import queue
import json
from config.settings import settings

class STTEngine:
    """
    Offline Speech-to-Text using VOSK.
    Converts microphone audio into text commands.
    Supports wake-word detection.
    """

    def __init__(self):
        print("🎤 Initializing Offline STT Engine (VOSK)...")

        model_path = settings.VOSK_MODEL_PATH
        self.model = vosk.Model(model_path)

        self.audio_q = queue.Queue()
        self.samplerate = 16000

    def _callback(self, indata, frames, time, status):
        """Internal callback for real-time microphone audio."""
        if status:
            print(status)
        self.audio_q.put(bytes(indata))

    def listen(self, prompt="Speak now..."):
        """
        Listen continuously until user finishes speaking.
        Returns recognized text.
        """
        print(f"🎙️ {prompt}")

        recognizer = vosk.KaldiRecognizer(self.model, self.samplerate)

        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            while True:
                data = self.audio_q.get()

                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    result = json.loads(result_json)
                    text = result.get("text", "").strip()

                    if text:
                        if settings.DEBUG:
                            print(f"🗣️ Recognized: {text}")
                        return text

    def listen_for_wake_word(self, wake_words=["assistant", "hey assistant","hey","hello"]):
        """
        Listens continuously until a wake-word is spoken.
        """
        print("👂 Waiting for wake-word...")

        while True:
            text = self.listen(prompt="")
            for w in wake_words:
                if w.lower() in text.lower():
                    print("🚀 Wake-word detected!")
                    return True
