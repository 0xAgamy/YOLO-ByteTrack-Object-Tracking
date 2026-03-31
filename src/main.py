

import object_tracker
import cv2
import json
cap=cv2.VideoCapture("inputs/3.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/output_video.mp4', fourcc, fps, (width, height))
    

frames, data_log=object_tracker.track_video(cap)
for frame in frames:
    out.write(frame)
out.release()
cap.release()
with open("outputs/datalog.json", 'w') as f:
    json.dump(data_log, f, indent=2)