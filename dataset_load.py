import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------- CONFIG --------
IMAGE_FOLDER = r"C:\Users\govind singh\OneDrive\Desktop\Qwen 2.5\dataset\image"
OUTPUT_JSONL = r"C:\Users\govind singh\OneDrive\Desktop\Qwen 2.5\dataset\dataset.jsonl"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

def main():
    print("🔹 Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)

    image_files = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_files:
        raise RuntimeError("❌ No images found! Check folder path or extensions.")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for image_path in tqdm(image_files, desc="Generating captions"):
            try:
                image = Image.open(image_path).convert("RGB")

                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                output = model.generate(**inputs, max_new_tokens=30)

                caption = processor.decode(output[0], skip_special_tokens=True)

                record = {
                    "image_path": image_path.replace("\\", "/"),
                    "text": caption
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"⚠️ Skipped {image_path}: {e}")

    print(f"\n✅ Dataset created successfully: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()


