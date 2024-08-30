import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp6/weights/best.pt', source='local')

# Initialize the DeepSORT tracker
deepsort = DeepSort(max_age = 5)

video_path = 'assignment/sample.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)
    
    # Extract bounding boxes and confidences
    bbox_xywh = []
    confs = []
    for *xyxy, conf, cls in results.xyxy[0]:
        bbox = [int(xyxy[0]), int(xyxy[1]) , int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]
        bbox_tuple = (bbox,conf,cls)
        bbox_xywh.append(bbox_tuple)
        # confs.append([conf.item()])
    
    # Update the DeepSORT tracker
    tracks = deepsort.update_tracks(bbox_xywh, frame = frame)

    # Process tracking results (outputs will contain bounding boxes and object IDs)

    for track in tracks:
        if not track.is_confirmed():
            continue
       
        
        track_id = track.track_id
        tlwh = track.to_tlwh()

        x1, y1, w, h = [int(x) for x in tlwh]
        x2 = x1 + w
        y2 = y1 + h

        print(x1, y1, w, h)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)  # Green box

        # Draw track ID
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


    # for output in outputs:
    #     x1, y1, x2, y2, track_id = output
   

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
