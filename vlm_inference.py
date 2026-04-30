import torch
import os
import numpy as np
# comment these env vars out if not using mps
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Allow up to 75% of unified memory for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.75"
# Start reclaiming cached buffers when 60% is used
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.60"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata
from tennis_dataset import TennisPointDataset
import cv2

ANNOTATION_DIR = "data/annotations"
VIDEO_DIR = "data/videos"

use_frame_proportion = 0.02 # 25 fps / 5 = 5 fps (this is what works on my M4)
use_checkpoint = True

points_dataset = TennisPointDataset(annotation_dir=ANNOTATION_DIR, video_dir=VIDEO_DIR, use_frame_proportion=use_frame_proportion)  # 25 FPS video

point = points_dataset[43]

frames = point["messages"][0]["content"][0]["video"]

if not os.path.isdir("frame_debug"):
    os.mkdir("frame_debug")
for i, frame in enumerate(frames):
    frame.save(f"frame_debug/frame{i}.png")

frames_video = [np.array(frame) for frame in frames]
writer = cv2.VideoWriter('frame_debug/point.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25 * use_frame_proportion, (frames_video[0].shape[1], frames_video[0].shape[0]))
for frame in frames_video:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
writer.release()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu

if use_checkpoint:
    model = Qwen3VLForConditionalGeneration.from_pretrained("checkpoint/checkpoint-279", dtype="auto", device_map=device).to(device)
else:
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map=device).to(device)

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=512 * 32 * 32)

video_metadata = VideoMetadata(
    total_num_frames=len(frames),
    fps=25.0 * use_frame_proportion,
    duration=len(frames) / (25.0 * use_frame_proportion)
)

inputs = processor.apply_chat_template(
    [point["messages"][0]],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    video_metadata=[video_metadata]
).to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)  # use model() to get loss and logits only during training loop
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\nGround truth description:")
    print(point["messages"][1]["content"])
    print("\nModel output:")
    print(output_text)

