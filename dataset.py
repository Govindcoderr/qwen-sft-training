# import json
# from PIL import Image
# from torch.utils.data import Dataset

# class VisionTextDataset(Dataset):
#     def __init__(self, jsonl_path, processor):
#         self.processor = processor
#         self.samples = []

#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 self.samples.append(json.loads(line))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]

#         image = Image.open(item["image_path"]).convert("RGB")
#         text = item["text"]

#         inputs = self.processor(
#             images=image,
#             text=text,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=128
#         )

#         inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#         inputs["labels"] = inputs["input_ids"].clone()

#         return inputs


import os
import json
from PIL import Image
from torch.utils.data import Dataset


class VisionTextDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        self.samples = []

        self.jsonl_path = os.path.abspath(jsonl_path)
        self.dataset_root = os.path.abspath(
            os.path.join(self.jsonl_path, "../../..")
        )

        self.image_index = self._build_image_index(self.dataset_root)

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                if "image_path" not in data or "text" not in data:
                    raise ValueError(
                        "Each JSONL line must contain 'image_path' and 'text'"
                    )

                filename = os.path.basename(data["image_path"])
                if filename not in self.image_index:
                    raise FileNotFoundError(f"Image not found: {filename}")

                data["image_path"] = self.image_index[filename]
                self.samples.append(data)

        if not self.samples:
            raise RuntimeError("Dataset is empty")

        print(f"✅ Loaded {len(self.samples)} samples")

    def _build_image_index(self, root_dir):
        index = {}
        valid_ext = (".jpg", ".jpeg", ".png", ".webp")

        print("🔍 Indexing images (one-time)...")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_ext):
                    index[file] = os.path.join(root, file)

        if not index:
            raise RuntimeError("❌ No images found in dataset folder")

        print(f"✅ Indexed {len(index)} images")
        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # image = Image.open(item["image_path"]).convert("RGB")
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((448, 448), Image.BICUBIC)
        text = item["text"]

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            truncation=True
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs
