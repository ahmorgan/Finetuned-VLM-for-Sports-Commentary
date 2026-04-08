import torch
import os
# comment these env vars out if not using mps
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Allow up to 75% of unified memory for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.75"
# Start reclaiming cached buffers when 60% is used
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.60"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16, device_map=device).to(device)

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=256 * 28 * 28)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

