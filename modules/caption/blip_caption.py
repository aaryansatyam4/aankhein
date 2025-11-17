import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from config.settings import settings

class BlipCaption:

    def __init__(self):
        print("🖼️ Initializing BLIP (correct conditional mode)…")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def caption(self, frame, mode="default"):
        """
        Correct BLIP captioning with real conditioning.
        """

        img = Image.fromarray(frame)

        # Correct prompt prefixes based on BLIP training
        prompts = {
            "default": "A photo showing",
            "object": "A photo of the object being held which is",
            "product": "A photo of a packaged product which is",
        }

        prompt = prompts.get(mode, prompts["default"])

        # Preprocess image
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # Tokenize prompt
        text_inputs = self.processor.tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(self.device)

        # Generate conditioned caption
        output_ids = self.model.generate(
            pixel_values=inputs.pixel_values,
            input_ids=text_inputs.input_ids,
            max_length=40,
            num_beams=5,
            repetition_penalty=1.1
        )

        caption = self.processor.decode(
            output_ids[0], 
            skip_special_tokens=True
        )

        # Remove the prompt prefix
        if caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()

        # Debug
        if settings.DEBUG:
            print(f"📝 BLIP Caption ({mode}): {caption}")

        return caption
