import os
import json
import cv2
from torch.utils.data import Dataset

class TennisPointDataset(Dataset):
    def __init__(self, annotation_dir, video_dir, num_frames=5):
        self.annotation_dir=annotation_dir
        self.video_dir=video_dir
        self.num_frames=num_frames
        self.samples=[]

        annotation_files= [
            f for f in os.listdir(annotation_dir)
            if f.endswith(".json")
        ]

        for ann_file in annotation_files:
            ann_path=os.path.join(annotation_dir, ann_file)
            video_name=os.path.splitext(ann_file)[0]+".mp4"
            video_path=os.path.join(video_dir, video_name)

            if not os.path.exists(video_path):
                if os.path.splitext(ann_file)[0]=="V009":
                    alt_path=os.path.join(video_dir, "V009-002.mp4")
                    if os.path.exists(alt_path):
                        video_path=alt_path
                    else:
                        continue
                else:
                    continue
            with open(ann_path,"r",encoding="utf-8") as f:
                data=json.load(f)
            points=self.find_points(data)

            for point in points:
                self.samples.append({
                    "annotation_file":ann_file,
                    "video_path": video_path,
                    "point_name": point.get("name",""),
                    "start": int(point["start"]),
                    "end": int(point["end"]),
                    "desc": point.get("desc",""),
                    "custom": point.get("custom", {})

                })
    def find_points(self,d):
        points= []

        if isinstance(d, dict):
            if(
                "start" in d and 
                "end" in d and 
                "name" in d and 
                "custom" in d and 
                isinstance(d["custom"],dict) and
                ("Score" in d["custom"] or "Winner" in d["custom"] or "Server" in d["custom"])
            ):
                points.append(d)
            for v in d.values():
                points.extend(self.find_points(v))

        elif isinstance(d,list):
            for item in d:
                points.extend(self.find_points(item))
        return points
    def sample_frame_indices(self,start,end):
        if self.num_frames==1:
            return [(start+end)//2]
        step= max(1, (end-start)//(self.num_frames-1))
        indices=[start+i*step for i in range(self.num_frames)]
        indices[-1] =end
        return indices
    def load_frames(self, video_path, start, end):
        frame_indices=self.sample_frame_indices(start,end)
        cap=cv2.VideoCapture(video_path)

        frames=[]
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame= cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames, frame_indices
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames, frame_indices=self.load_frames(
            sample["video_path"],
            sample["start"],
            sample["end"]
        )
        return {
            "frames": frames,
            "frame_indices": frame_indices,
            "video_path": sample["video_path"],
            "annotation_file":sample["annotation_file"],
            "point_name": sample["point_name"],
            "start": sample["start"],
            "end": sample["end"],
            "desc": sample["desc"],
            "custom": sample["custom"]
        }
if __name__ == "__main__":
    dataset = TennisPointDataset(
        annotation_dir="data/annotations",
        video_dir="data/videos",
        num_frames=5
    )

    print("Total samples:", len(dataset))

    sample = dataset[0]
    print("Point name:", sample["point_name"])
    print("Start:", sample["start"])
    print("End:", sample["end"])
    print("Custom:", sample["custom"])
    print("Loaded frames:", len(sample["frames"]))
    print("Frame indices:", sample["frame_indices"])

