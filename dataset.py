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
        self.base_dir = os.path.dirname(self.jsonl_path)

        # Try common image roots
        self.possible_image_roots = [
            self.base_dir,
            os.path.join(self.base_dir, "images"),
            os.path.join(self.base_dir, "output", "images"),
            os.path.join(os.path.dirname(self.base_dir), "images"),
        ]

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                if "image_path" not in data or "text" not in data:
                    raise ValueError(
                        "Each JSONL line must contain 'image_path' and 'text'"
                    )

                image_path = self._resolve_image_path(data["image_path"])
                data["image_path"] = image_path

                self.samples.append(data)

        if len(self.samples) == 0:
            raise RuntimeError("Dataset is empty")

        print(f"✅ Loaded {len(self.samples)} samples")

    def _resolve_image_path(self, image_path):
        # Case 1: absolute path
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        # Case 2: relative to jsonl
        candidate = os.path.join(self.base_dir, image_path)
        if os.path.exists(candidate):
            return candidate

        # Case 3: search common image folders
        image_name = os.path.basename(image_path)
        for root in self.possible_image_roots:
            candidate = os.path.join(root, image_name)
            if os.path.exists(candidate):
                return candidate

        # ❌ Not found
        raise FileNotFoundError(
            f"❌ Image not found: {image_path}\n"
            f"Searched in: {self.possible_image_roots}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs
