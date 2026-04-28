import torch
import os
import numpy as np
from torch.utils.data import random_split
from accelerate import Accelerator
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata
from tennis_dataset import TennisPointDataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import logging
from transformers import BitsAndBytesConfig
from transformers import TrainerCallback
import gc
import random
from evaluate import load

rouge = load("rouge")

logging.enable_progress_bar()

# accelerator = Accelerator()

HF_TOKEN="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"
ANNOTATION_DIR = "data/annotations"
VIDEO_DIR = "data/videos"

use_frame_proportion = 0.02  # 25 fps / 50 = 0.5 fps
train_vision_model = False
train_only_lmhead = False

points_dataset = TennisPointDataset(annotation_dir=ANNOTATION_DIR, video_dir=VIDEO_DIR, use_frame_proportion=use_frame_proportion)  # 25 FPS video

dataset_size = len(points_dataset)
train_size = int(0.80 * dataset_size)
eval_size = dataset_size - train_size  # 20% of the dataset
train_dataset, eval_dataset = random_split(points_dataset, [train_size, eval_size])

print(f"Total dataset size: {dataset_size}")
print(f"Training dataset size: {train_size}")
print(f"Evaluation dataset size: {eval_size}")
print(f"Settings: FPS: {25 * use_frame_proportion}, train vision model: {train_vision_model}, train only LM head: {train_only_lmhead}")

point = points_dataset[random.randint(0, len(points_dataset)-1)]

frames = point["messages"][0]["content"][0]["video"]
print(f"Number of frames in the debug video: {len(frames)}")

if not os.path.isdir("frame_debug"):
    os.mkdir("frame_debug")
for i, frame in enumerate(frames):
    frame.save(f"frame_debug/frame{i}.png")

"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda:" + str(accelerator.local_process_index)) if torch.cuda.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu
print(f"Using device: {device}")
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16, device_map=device, token=HF_TOKEN, attn_implementation="sdpa")
# model = prepare_model_for_kbit_training(model) 

def print_trainable_params(model):
    num = 0
    for name, param in model.named_parameters():
        if name == "lm_head":
            print(f"lm_head param count: {param.numel()}")
        if param.requires_grad:
            num += param.numel()
    return num

print(f"Number of trainable params: {print_trainable_params(model)}")

lora_config = LoraConfig(
    r=8,  # rank of rank-decomposed weight matrix
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj', 'lm_head'] if train_only_lmhead else ['q_proj', 'v_proj'],  # modules to apply LoRA to
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)
lora_model.enable_input_require_grads()
lora_model.model.model.visual.enable_input_require_grads()
lora_model.model.model.visual.gradient_checkpointing = True

if train_only_lmhead:
    for name, param in lora_model.named_parameters():
        if "lm_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

if not train_vision_model:
    for param in lora_model.model.model.visual.parameters():
        param.requires_grad = False

print(f"Number of trainable params in LoRA adapted model: {print_trainable_params(lora_model)}")

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
processor.video_processor.max_num_frames = 8

def collate_fn(batch):
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
    print(f"Total token count: {ret_batch['input_ids'].shape[-1]}")

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

    ret_batch.to(device, non_blocking=True)

    return ret_batch

def preprocess_logits_for_metrics(logits, labels):
    """
    converts full logits to argmax prediction to save memory
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """
    computes accuracy \
    cross-entropy loss is handled automatically by HF Trainer 
    and is logged as 'eval_loss'
    """
    print("Computing metrics on evaluation batch...")
    # predictions already argmaxed because of preprocess_logits_for_metrics
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # flatten everything
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # mask out padded tokens (-100)
    mask = labels != -100

    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    
    accuracy = correct / total if total > 0 else 0.0

    labels[labels == -100] = processor.tokenizer.pad_token_id
    predictions[predictions == -100] = processor.tokenizer.pad_token_id

    torch.cuda.empty_cache()
    gc.collect()
    
    return {"token_accuracy": accuracy}

class ClearTorchCache(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        print("Emptied CUDA cache")

class GenerationEvalCallback(TrainerCallback):
    """
    Jury-rigged evaluation callback to generate and print model outputs
    """

    def __init__(self, eval_dataset, processor, num_samples=16):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        with torch.no_grad():
            all_rouge = []
            for i in range(10):  # just do 10 random samples because full inference is slow
                idx = random.randint(0, len(self.eval_dataset)-1)
                sample = self.eval_dataset[idx]
                frames = sample["messages"][0]["content"][0]["video"]
                
                video_metadata = VideoMetadata(
                    total_num_frames=len(frames),
                    fps=25.0 * use_frame_proportion,
                    duration=len(frames) / (25.0 * use_frame_proportion)
                )

                messages = [sample["messages"][0]]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    video_metadata=[video_metadata]
                ).to(device)

                generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                print(f"\nGround truth description for sample {idx}:")
                print(sample["messages"][1]["content"])
                print("\nModel output:")
                print(output_text)
                print()

                rouge_score = rouge.compute(predictions=[output_text], references=[sample["messages"][1]["content"]])
                print(f"ROUGE score: {rouge_score}")
                all_rouge.append(rouge_score)
            avg_rouge = {key: np.mean([score[key] for score in all_rouge]) for key in all_rouge[0].keys()}
            print(f"Average ROUGE score: {avg_rouge}")
        model.train()


training_config = SFTConfig(
    #pad_to_multiple_of=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},  
    gradient_accumulation_steps=8,
    per_device_train_batch_size=1,  # we are on one device
    per_device_eval_batch_size=2,
    auto_find_batch_size=False,  # If batch size would cause OOM, halves its size until it works
    dataloader_pin_memory=False,  # maybe figure out how to use this if there is time
    max_length=4096,
    num_train_epochs=3,
    learning_rate=1e-4,
    optim='paged_adamw_8bit',
    max_grad_norm=1.0,
    save_strategy="steps",
    save_steps=300,
    logging_strategy="steps",
    logging_steps=1,
    logging_dir='./logs',
    eval_strategy="steps",
    eval_steps=10,
    eval_on_start=True,
    do_eval=True,
    output_dir='./checkpoint',
    report_to='none',
    dataset_kwargs={"skip_prepare_dataset": True},
    dataset_text_field=None,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=False,
    fp16=True,
    #ddp_find_unused_parameters=False
)

trainer = SFTTrainer(
    model=lora_model,
    processing_class=processor,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    callbacks=[ClearTorchCache(), GenerationEvalCallback(eval_dataset, processor)],  # --> callback after each training iteration which clears the gpu cache
    compute_metrics=compute_metrics
)

torch.cuda.empty_cache()
gc.collect()
trainer.train()