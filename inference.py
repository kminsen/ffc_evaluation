import cv2
import torch

# load yolo model
# load video 
# using cv2 extract the frames
# model inference with frames as input
# i dont think i need deepsort for the first approach

def isStaff(result):
    detection = result.xyxy[0].shape[0]
    if detection <= 0:
        return False
    return True


if __name__=='__main__':
    model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp6/weights/best.pt', source='local')

    video_path = 'assignment/sample.mp4'

    cap = cv2.VideoCapture(video_path)

    staff_frames = []
    count = 0

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        result = model(frame)
        
        if isStaff(result):
            cx = (result.xyxy[0][0][0] + result.xyxy[0][0][2]) / 2
            cy = (result.xyxy[0][0][1] + result.xyxy[0][0][3]) / 2

            with open('results.txt','a+') as result_file:
                staff_frames.append([count,(cx,cy)])
                result_file.write(f'{count}th frame, (x,y):({cx}, {cy})\n')

        count += 1
    print(len(staff_frames))
