import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=True
)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_captions_from_folder(folder_path):
    
    
    for filename in os.listdir(folder_path):

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            inputs = processor(images=image, return_tensors="pt")
            
            # caption
            with torch.no_grad():
                output = model.generate(**inputs)

            caption = processor.decode(output[0], skip_special_tokens=True)
            print(f"Caption for {filename}: {caption}")


folder_path = "ifolder"

# captions for all images in the folder
generate_captions_from_folder(folder_path)
