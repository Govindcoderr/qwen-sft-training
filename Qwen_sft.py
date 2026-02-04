import os
import torch
from transformers import DataCollatorWithPadding
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from dataset import VisionTextDataset
from transformers import DataCollatorWithPadding
import torch


# 🔴 🔴 🔴 ONLY THIS LINE YOU CHANGE FOR 7B → 72B 🔴 🔴 🔴
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"   # future

DATASET_ROOT = "dataset"


# class QwenVLCollator:
#     def __init__(self, processor):
#         # tokenizer is inside processor
#         self.text_collator = DataCollatorWithPadding(
#             tokenizer=processor.tokenizer,
#             padding=True
#         )

#     def __call__(self, features):
#         # 1️⃣ pixel_values alag nikalo
#         pixel_values = [f.pop("pixel_values") for f in features]

#         # 2️⃣ text + labels ko PAD karo (variable length safe)
#         batch = self.text_collator(features)

#         # 3️⃣ images stack karo
#         batch["pixel_values"] = torch.stack(pixel_values)

#         return batch

class QwenVLCollator:
    def __init__(self, processor):
        self.text_collator = DataCollatorWithPadding(
            tokenizer=processor.tokenizer,
            padding=True
        )

    def __call__(self, features):
        pixel_values = [f.pop("pixel_values") for f in features]
        labels = [f.pop("labels") for f in features]

        batch = self.text_collator(features)

        max_len = max(len(l) for l in labels)
        padded_labels = []
        for l in labels:
            pad_len = max_len - len(l)
            if pad_len > 0:
                l = torch.cat(
                    [l, torch.full((pad_len,), -100, dtype=l.dtype)]
                )
            padded_labels.append(l)

        batch["labels"] = torch.stack(padded_labels)
        batch["pixel_values"] = torch.stack(pixel_values)
        return batch



def find_jsonl_file(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                print("📄 Using dataset:", os.path.join(root, file))
                return os.path.join(root, file)
    raise FileNotFoundError("No .jsonl found inside dataset folder")


def main():
    dataset_path = find_jsonl_file(DATASET_ROOT)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # Windows-safe
        device_map="auto",
        trust_remote_code=True
    )

    train_dataset = VisionTextDataset(
        jsonl_path=dataset_path,
        processor=processor
    )

    training_args = TrainingArguments(
        output_dir="./qwen25_vl_sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # data_collator=QwenVLCollator()
        data_collator = QwenVLCollator(processor)

    )

    trainer.train()

    model.save_pretrained("./qwen25_vl_sft_output")
    processor.save_pretrained("./qwen25_vl_sft_output")

    print("✅ Training complete & model saved")


if __name__ == "__main__":
    main()


# code for fsdf    

# import os
# import torch
# from transformers import (
#     AutoProcessor,
#     Qwen2_5_VLForConditionalGeneration,
#     TrainingArguments,
#     Trainer
# )
# from dataset import VisionTextDataset


# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# DATASET_ROOT = "dataset"


# def find_jsonl_file(root_dir):
#     jsonl_files = []
#     for root, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".jsonl"):
#                 jsonl_files.append(os.path.join(root, file))

#     if not jsonl_files:
#         raise FileNotFoundError(" No .jsonl file found inside dataset folder")

#     if len(jsonl_files) > 1:
#         print(" Multiple JSONL files found, using first one")

#     print(" Using dataset:", jsonl_files[0])
#     return jsonl_files[0]


# def main():
#     dataset_path = find_jsonl_file(DATASET_ROOT)

#     processor = AutoProcessor.from_pretrained(
#         MODEL_ID,
#         trust_remote_code=True
#     )

#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True
#     )

#     train_dataset = VisionTextDataset(
#         jsonl_path=dataset_path,
#         processor=processor
#     )

#     training_args = TrainingArguments(
#         output_dir="./qwen25_vl_fsdp",
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-5,
#         num_train_epochs=3,
#         bf16=True,
#         logging_steps=10,
#         save_steps=500,
#         save_total_limit=2,
#         report_to="none",
#         remove_unused_columns=False,

#         #  FSDP CONFIG 
#         fsdp="full_shard auto_wrap",
#         fsdp_transformer_layer_cls_to_wrap="Qwen2_5_VLDecoderLayer",
#         fsdp_min_num_params=1e8,

#         # Stability
#         gradient_checkpointing=True,
#         ddp_find_unused_parameters=False,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset
#     )

#     trainer.train()

#     model.save_pretrained("./qwen25_vl_fsdp")
#     processor.save_pretrained("./qwen25_vl_fsdp")

#     print(" FSDP training complete")


# if __name__ == "__main__":
#     main()
