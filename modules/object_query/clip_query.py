from transformers import CLIPProcessor, CLIPModel
from config.settings import settings
from PIL import Image
import torch

class CLIPQuery:
    """
    Uses CLIP to answer text-query-based questions about the image.
    Example: 'what is in my hand?' or 'is this a bottle?'
    """

    def __init__(self):
        print("🔍 Initializing CLIP Query Model...")

        self.device = settings.DEVICE
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)

    def query(self, frame, user_query: str):
        """
        Compares the user query against the image using CLIP similarity.
        Returns a confidence score and natural language result.
        """
        img = Image.fromarray(frame)

        # Prepare inputs for CLIP
        inputs = self.processor(
            text=[user_query],
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Extract similarity score
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

        confidence = float(probs[0][0])

        if settings.DEBUG:
            print(f"🔎 CLIP Query: '{user_query}' → Confidence: {confidence:.2f}")

        # Return simple text summary
        if confidence > 0.60:
            return f"Yes, it matches your query. Confidence: {confidence:.2f}"
        elif confidence > 0.35:
            return f"Possibly, but I’m not fully sure. Confidence: {confidence:.2f}"
        else:
            return f"No, it does not match your query. Confidence: {confidence:.2f}"
