import os
import json
import cv2
from torch.utils.data import Dataset
from PIL import Image
import math
import numpy as np

class TennisPointDataset(Dataset):
    def __init__(self, annotation_dir, video_dir, use_frame_proportion=0.5):
        self.annotation_dir=annotation_dir
        self.video_dir=video_dir

        if math.floor(1 / use_frame_proportion) != 1 / use_frame_proportion:
            raise ValueError("1 / use_frame_proportion must not be a decimal")

        self.use_frame_proportion=use_frame_proportion
        self.samples=[]

        annotation_files= [
            f for f in os.listdir(annotation_dir)
            if f.endswith(".json")
        ]

        for ann_file in annotation_files:
            ann_path=os.path.join(annotation_dir, ann_file)
            video_name=os.path.splitext(ann_file)[0]+".mp4"
            video_path=os.path.join(video_dir, video_name)

            with open(ann_path,"r",encoding="utf-8") as f:
                data=json.load(f)
            points=data["classes"]["Point"]  # removed find_points(), we can just directly get the points array

            for point in points:
                self.samples.append({
                    "annotation_file":ann_file,
                    "video_path": video_path,
                    "point_name": point.get("name",""),
                    "start": int(point["start"]),
                    "end": int(point["end"]),
                    "desc": point.get("desc",""),
                    "score": point["custom"]["Score"]
                })
    def sample_frame_indices(self,start,end):
        if start-end == 1:
            return [start, end]
        step = int(1 / self.use_frame_proportion)
        indices = [start+i*step for i in range((end-start)//step if end-start % step == 0 else (end-start)//step+1)]
        if end not in indices:  # always include last frame
            indices.append(end)
        return indices
    def load_frames(self, video_path, start, end):
        frame_indices=self.sample_frame_indices(start,end)
        cap=cv2.VideoCapture(video_path)

        frames=[]
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame= cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                frames.append(frame)
        cap.release()
        return frames, frame_indices
    def convert_frames_to_PIL(self, frames):
        for i in range(len(frames)):
            frames[i] = Image.fromarray(frames[i])
        return frames
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames, frame_indices=self.load_frames(
            sample["video_path"],
            sample["start"],
            sample["end"]
        )
        frames = self.convert_frames_to_PIL(frames)
        return {
            "frames": frames,
            "frame_indices": frame_indices,
            "video_path": sample["video_path"],
            "annotation_file":sample["annotation_file"],
            "point_name": sample["point_name"],
            "start": sample["start"],
            "end": sample["end"],
            "desc": sample["desc"],
            "score": sample["score"]
        }
if __name__ == "__main__":
    dataset = TennisPointDataset(
        annotation_dir="data/annotations",
        video_dir="data/videos",
        use_frame_proportion=0.5
    )

    print("Total samples:", len(dataset))

    sample = dataset[740]
    print("Point name:", sample["point_name"])
    print("Start:", sample["start"])
    print("End:", sample["end"])
    print("Custom:", sample["score"])
    print("Loaded frames:", len(sample["frames"]))
    print("Frame indices:", sample["frame_indices"])
    print("Description: ", sample["desc"])

