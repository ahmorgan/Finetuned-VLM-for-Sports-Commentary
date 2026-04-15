import cv2
import torch
import os
import json 
from PIL import Image 
# comment these env vars out if not using mps
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Allow up to 75% of unified memory for MPS
#os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.75"
# Start reclaiming cached buffers when 60% is used
#os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.60"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ANNOTATION_FILE= "data/annotations/V009.json"
VIDEO_FILE = "data/videos/V009-002.mp4"
TEMP_FRAME_DIR="temp_frames"

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data=json.load(f)
def find_point(d):
    if isinstance(d, dict):
        if( "start" in d and 
            "end" in d and
            "name" in d and 
            "custom" in d and 
            isinstance(d["custom"],dict) and
            ("Score" in d["custom"] or "Winner" in d["custom"])):
            return d
        for v in d.values():
            result= find_point(v)
            if result is not None:
                return result
    elif isinstance(d,list):
        for item in d:
            result= find_point(item)
            if result is not None:
                return result
    return None 
point=find_point(data)

if point is None:
    raise ValueError("Could not find a point with start/end/desc")

start=int(point["start"])
end=int(point["end"])

print("using point")
print(json.dumps(point, indent=2))

os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
frame_indices=[start, (start+end)//2, end]
frame_paths=[]

cap=cv2.VideoCapture(VIDEO_FILE)
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame=cap.read()
    if ret:
        frame_path=os.path.join(TEMP_FRAME_DIR, f"{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    else:
        print("Warning could not read")
cap.release()

if not frame_paths:
    raise RuntimeError("No frames were extracted")

print("Saved frames:", frame_paths)






device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")  # change to cuda if using cuda enabled gpu
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16, device_map=device).to(device)

# max_pixels controls the visual token size (size of image patches passed to vision encoder transformer)
# we want *up to* 256 28x28 image patches here (could be smaller because the processor has to snap the image resolution to the nearest dimensions divisible by 28).
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=256 * 28 * 28)

messages = [
    {
        "role": "user",
        "content": (
            [{"type": "image", "image":p} for p in frame_paths]+
            [{"type": "text", "text": "Describe the following tennis clip "}]


            
        ),
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

    print("\nGround truth description:")
    print(point["desc"])
    print("\nModel output:")
    print(output_text)

