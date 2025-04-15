from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=True
)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
image = Image.open("example.jpeg").convert("RGB")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)
print("Caption:", caption)
