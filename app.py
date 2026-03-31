import json 
from src.object_tracker import track_video
import gradio as gr
import cv2
from pathlib import Path

import tempfile

exmaple_path= list(Path("inputs/").glob("*.mp4"))
example_list=[]
for p in exmaple_path:
    example_list.append(p)

def gradio_wrapper(video_file):
    print(f"The video type is :  {type(video_file)} and it's {video_file}")

    cap = cv2.VideoCapture(video_file)
    print(f"The cap type is :  {type(cap)} and it's {cap}")
    frames, data_log = track_video(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    

    fd,output_path = tempfile.mkstemp(suffix=".mp4")
    video=cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(width, height),isColor=True)


    for frame in frames:
        video.write(frame)
    video.release()

    return output_path, json.dumps(data_log, indent=2)


iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.Textbox(label="Tracking Data (JSON)", lines=2,max_length=5)
    ],
    title="YOLO Object Tracker",
    description="Upload a video and track people & vehicles with YOLO",
    examples=example_list
)

iface.launch()