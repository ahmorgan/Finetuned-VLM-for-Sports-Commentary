import torch
import os
import numpy as np
# comment these env vars out if not using mps
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Allow up to 75% of unified memory for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.95"
# Start reclaiming cached buffers when 60% is used
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.60"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata
from tennis_dataset import TennisPointDataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import logging
from transformers import QuantoConfig
from transformers import TrainerCallback
import gc

logging.enable_progress_bar()

ANNOTATION_DIR = "data/annotations"
VIDEO_DIR = "data/videos"

use_frame_proportion = 0.20  # 25 fps / 5 = 5 fps (this is what works on my M4)

points_dataset = TennisPointDataset(annotation_dir=ANNOTATION_DIR, video_dir=VIDEO_DIR, use_frame_proportion=use_frame_proportion)  # 25 FPS video

point = points_dataset[0]

frames = point["messages"][0]["content"][0]["video"]

if not os.path.isdir("frame_debug"):
    os.mkdir("frame_debug")
for i, frame in enumerate(frames):
    frame.save(f"frame_debug/frame{i}.png")

quantization_config = QuantoConfig(weights="int4")
device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype="auto", quantization_config=quantization_config, device_map=device)

for param in model.model.visual.parameters():
    param.requires_grad = False

def print_trainable_params(model):
    num = 0
    for param in model.parameters():
        if param.requires_grad:
            num += param.numel()
    return num

print(f"Number of trainable params: {print_trainable_params(model)}")

lora_config = LoraConfig(
    r=2,  # rank of rank-decomposed weight matrix
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05,
    # qkv and proj from vision transformer, everything else from llm
    target_modules=['q_proj', 'v_proj'],  # modules to apply LoRA to
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)

print(f"Number of trainable params in LoRA adapted model: {print_trainable_params(lora_model)}")

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=64 * 28 * 28)

def collate_fn(batch):
    """
    Processes a raw batch of data into input_ids that can be given to the model.
    This function also applies masking to the prompt in ret_batch["labels"].

    :param batch: raw batch of data from TennisPointDataset
    :return: processed batch of masked, padded input_ids
    """
    # Credit: Adapted code from https://www.datacamp.com/tutorial/fine-tuning-qwen3-vl-8b to write this collator.

    videos = [example["messages"][0]["content"][0]["video"] for example in batch]

    video_metadata = [
        VideoMetadata(
            total_num_frames=len(video),
            fps=25.0 * use_frame_proportion,
            duration=len(video) / (25.0 * use_frame_proportion)
        ) for video in videos
    ]

    texts = [processor.apply_chat_template(
        example["messages"],
        tokenize=False,
        return_tensors="pt",
        add_generation_prompt=False,
        video_metadata=[
            video_metadata[i]
        ]
    ) for i, example in enumerate(batch)]

    prompt_texts = [
        processor.apply_chat_template(
            example["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            video_metadata=[
                video_metadata[i]
            ]
        )
        for i, example in enumerate(batch)
    ]
    # returns pixel_values field (check this)
    ret_batch = processor(
                text=texts,
                videos=videos,
                return_tensors="pt",
                padding=True,
                truncation=True,
                video_metadata=video_metadata
                )

    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    prompt_lens = (prompt_ids != processor.tokenizer.pad_token_id).sum(dim=1)

    labels = ret_batch["input_ids"].clone()
    bs, seqlen = labels.shape

    # mask out the input text prompt in the labels so we can do instruction tuning
    for i in range(bs):
        pl = int(prompt_lens[i].item())
        pl = min(pl, seqlen)
        labels[i, :pl] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100
    ret_batch["labels"] = labels

    return ret_batch

def compute_metrics(eval_pred):
    # will compute perplexity and loss after N iterations here
    pass

class ClearTorchCache(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.empty_cache()
        gc.collect()
        print("Emptied MPS cache")

training_config = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=1,
    per_device_train_batch_size=1,  # we are on one device
    auto_find_batch_size=True,  # If batch size would cause OOM, halves its size until it works
    max_length=64,
    num_train_epochs=10,
    learning_rate=1e-4,
    optim='adamw_torch',
    save_strategy="steps",
    save_steps=100,
    logging_steps=1,
    logging_dir='./logs',
    output_dir='./checkpoint',
    report_to='none',
    dataset_kwargs={"skip_prepare_dataset": True},
    dataset_text_field=None,
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=lora_model,
    processing_class=processor,
    args=training_config,
    train_dataset=points_dataset,
    data_collator=collate_fn,
    callbacks=[ClearTorchCache()]  # --> callback after each training iteration which clears the MPS cache
)

torch.mps.empty_cache()
gc.collect()
trainer.train()