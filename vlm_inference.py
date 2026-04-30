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

ANNOTATION_DIR = "data/annotations"
VIDEO_DIR = "data/videos"

use_frame_proportion = 0.20  # 25 fps / 5 = 5 fps (this is what works on my M4)

points_dataset = TennisPointDataset(annotation_dir=ANNOTATION_DIR, video_dir=VIDEO_DIR, use_frame_proportion=use_frame_proportion)  # 25 FPS video

point = points_dataset[0]

frames = point["frames"]

if not os.path.isdir("frame_debug"):
    os.mkdir("frame_debug")
for i, frame in enumerate(frames):
    frame.save(f"frame_debug/frame{i}.png")

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map=device).to(device)

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=512 * 28 * 28)

video_metadata = VideoMetadata(
    total_num_frames=len(frames),
    fps=25.0 * use_frame_proportion,
    duration=len(frames) / (25.0 * use_frame_proportion)
)

player1, player2 = point["player1"], point["player2"]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [frame for frame in frames],
            },
            {"type": "text", "text": f"This tennis clip shows a point between {player1} and {player2}. The score is {point['score']}. Describe what happens in the tennis clip, focusing on the players' actions and point outcome."}  # WE SHOULD ALSO PUT POINT METADATA HERE
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
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
    print(point["desc"])
    print("\nModel output:")
    print(output_text)

