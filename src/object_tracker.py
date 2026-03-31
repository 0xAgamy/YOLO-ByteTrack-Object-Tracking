import cv2
from ultralytics import YOLO
import json
import tempfile
import os
from pathlib import Path
device = 'cpu'
model= YOLO("models/yolo26n.pt")
model.to(device)
PERSON_CLASSES   = {0: "person"}
VEHICLE_CLASSES  = {
    1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 8: "boat",
}
TARGET_CLASSES   = {**PERSON_CLASSES, **VEHICLE_CLASSES}


def track_video(cap:cv2.VideoCapture):
   
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    
    frames=[]
    data_log = []
    frame_count = 0


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
    
        results = model.track(frame, persist=True, tracker='config/bytetrack-modified.yaml', classes=list(TARGET_CLASSES.keys()), verbose=False)
        
        result = results[0]
        result.boxes.cls.int()
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()      # Bounding boxes
            confs = result.boxes.conf.cpu().numpy()      # Confidence
            track_ids = result.boxes.id.int().cpu().numpy() # Track IDs
            class_names_ids= result.boxes.cls.int().cpu().numpy() 
            timestamp = frame_count / fps
            
            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)
                
                track_id = int(track_ids[i])
                class_id=int(class_names_ids[i])
                class_name = TARGET_CLASSES[class_id]
                conf = float(confs[i])
                
                if class_name == 'person':
                    color = (0, 255, 0) # Green
                else:
                    color = (225, 0, 0) # Blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} | ID:{track_id}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                record = {
                    "frame_index": frame_count,
                    "timestamp_sec": round(timestamp, 3),
                    "track_id": track_id,
                    "confidence": round(conf, 3),
                    "class_name":class_name,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
                data_log.append(record)
            frames.append(frame)
        
        # if frame_count % 30 == 0:
        #     print(f"Processed Frame {frame_count}")
            
        frame_count += 1
 

    return frames,data_log