import cv2
from ultralytics import YOLO
import json
device = 'cpu'
model= YOLO("models/yolo26n.pt")
model.to(device)
class_names = model.names

cap= cv2.VideoCapture("inputs/1-video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/tracked_video.mp4', fourcc, fps, (width, height))


data_log = []
print("Starting Processing...") # don't miss that 
frame_count = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
   
    results = model.track(frame, persist=True, tracker='conf/bytetrack-modified.yaml', classes=[0], verbose=False)
    
   
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
            class_name = class_names[class_id]
            conf = float(confs[i])
            
            
            color = (0, 255, 0) # Green
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

    out.write(frame)
    
    # Print progress - i think i will remove this -
    if frame_count % 30 == 0:
        print(f"Processed Frame {frame_count}")
        
    frame_count += 1

cap.release()
out.release()

with open('outputs/tracking_data.json', 'w') as f:
    json.dump(data_log, f, indent=2)

print("Processing Complete.")
