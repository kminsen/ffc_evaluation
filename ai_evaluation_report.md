# AI Evaluation Test Report
- [AI Evaluation Test Report](#ai-evaluation-test-report)
  - [1. Introduction](#1-introduction)
  - [2. Problem Statement](#2-problem-statement)
  - [3. Data Description](#3-data-description)
  - [4. Approach](#4-approach)
    - [4.1 Preprocessing](#41-preprocessing)
    - [4.2 Model Selection](#42-model-selection)
    - [4.3 Training/Finetuning](#43-trainingfinetuning)
    - [4.4 Detection (1. detection.py and self written code)](#44-detection-1-detectionpy-and-self-written-code)
  - [5. Results](#5-results)
    - [5.1 Training Metrics Performance Metrics](#51-training-metrics-performance-metrics)
    - [5.2 Test Metrics](#52-test-metrics)
  - [6. Conclusion](#6-conclusion)
  - [8. References](#8-references)
  - [9. Appendix (if needed)](#9-appendix-if-needed)

## 1. Introduction
- **Objective**: To train a AI model's ability to detect staff members (individuals wearing a staff badge) in a video and identify the frames where the staff members are detected.

## 2. Problem Statement
- **Task**: Provided with a 53-second video at 25 frames per second (fps), the objective is to identify the specific frames within the video where an employee wearing a staff badge appears. During the interview, you will be given a test video and will be required to run your detection process on it.
- **Challenges**:
  - Variability in lighting and video quality.
  - Motion and occlusion of badges.
  - Differentiating between staff and non-staff individuals.
- **Assumptions made**
  - As long as the staff badge is not seen(occluded), that person detected is no longer considered as a staff.

## 3. Data Description
- **Video File**: `sample.mp4` (53 seconds)
- **Resolution**:  960x720 pixels
- **Frame Rate**:  25 frames per second

## 4. Approach

### 4.1 Preprocessing
- **Frame Extraction**: 
  - Extracted frames from the video at a rate of 25 frames per second for analysis.
- **Data Annotation**: 
  - Manually create bounding box for groundtruth images using labelImg.

  
### 4.2 Model Selection
- **Model Used**: YOLOv5 Object Detection Model
  - **Speed and Efficiency**: YOLOv5 models, especially the smaller versions like YOLOv5s, are lightweight and can run efficiently on less powerful hardware.
  - **Active Community**: YOLOv5 has a large and active community, with frequent updates and improvements. This means there’s plenty of support available, from tutorials to forums where you can ask questions.
  - **High Accuracy**: YOLOv5 achieves competitive accuracy with a good balance between precision and recall. It’s capable of detecting objects with high confidence, even in challenging scenarios.
  - **Pre-trained Weights**: Pretrained on the COCO dataset, YOLOv5 models generalize well across various types of objects and environments.
  

### 4.3 Training/Finetuning 
- **Dataset**: images/frames extracted from `sample.mp4`. Train-val-test split ratio is **8:1:1**.
- **Training Parameters**: epoch:50, other parameters are left to default.

### 4.4 Detection (1. detection.py and self written code)
- confidence threshold of 25 and iou thresold of 45
1. **detect.py**:
  - **Important Note 1** : Bounding box coordinates extracted from detect.py is normalized, hence to get xy coordinate first need to multiply the normalized coordinate with the dimension of image.
  - **Important Note 2** : detect.py uses rectangular inference, hence the exact size of image will vary when it is being fed to the model. So calculating the xy coordinate of the staff includes, extracting the frame width and height in detect.py and multiply it will the normalized bounding box coordinate. Refer to reference number 2 on **RECTANGULAR INFERENCE**
  - **Output** consists of:
    1. **predictions.csv**: containing the index of frame that detected the staff, coordinate of the staff in the frame, confidence interval.  
    2. **video**: video with detected frames.
  
2. **inference.py**
   - This is a self written script mainly using Pytorch and CV2 libraries, it extracts the frames from video, and perform inference using fine-tuned yolov5 model.
    - **Output** consists of:
      1. **result.txt** : containing the index of frames within the video that detected staff,and xy coordinates


## 5. Results

### 5.1 Training Metrics Performance Metrics
- ![Confusion Matrix](./yolov5/runs/train/exp7/confusion_matrix.png)
- ![F1-curve](./yolov5/runs/train/exp7/F1_curve.png)
- ![PR-Curve](./yolov5/runs/train/exp7/PR_curve.png)
- ![Loss](./yolov5/runs/train/exp7/results.png)

### 5.2 Test Metrics
- Test is done using val.py on test set.
- Model shows sign of overfitting, with a TP and FP of 1 in confusion matrix, too confident in detecting staff.
- personally i believe its the lack of data, to enable the model to effectively distinguish between the pocket and staff badge
- pretrained yolov5 trained on coco dataset and the images resolution and view is quite different with the datast too

- ![Confusion Matrix](./yolov5/runs/val/run7_test/confusion_matrix.png)
- ![F1-curve](./yolov5/runs/val/run7_test/F1_curve.png)
- ![PR-Curve](./yolov5/runs/val/run7_test/PR_curve.png)

## 6. Conclusion
- **Summary**
- the finetuned model is great at detecting staff, in fact it is too 'confident' as there are lots of False Postives.
- A sign of overfitting due to the scarcity of dataset that is used to finetune the model. 
- In my opinion, larger dataset is required to increase the model performance of the model.


## 8. References
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [GitHubIssue: RECTANGULAR INFERENCE](https://github.com/ultralytics/yolov3/issues/232)
- [GitHubIssue: Difference between val and detect](https://github.com/ultralytics/yolov5/issues/8584)
- [GithubIssue: Varying image size](https://github.com/ultralytics/yolov5/issues/9046)
  
## 9. Appendix (if needed)
- **Frame Examples**: Add images of test, labels vs prediction and to show that the model is indeedly overfitting.
- ![GroundTruth when Testing](./yolov5/runs/val/run7_test/val_batch0_labels.jpg)
- ![Predictions when Testing](./yolov5/runs/val/run7_test/val_batch0_pred.jpg)

- **Commands**: 
  - **Training**:
    - ```python train.py --img 640 --epochs 50 --data custom.yaml --weights yolov5s.pt```
  - **Testing using val.py**:
    - ```python val.py --weights runs/train/exp7/weights/best.pt --data data/custom_val.yaml --save-hybrid --name test --conf-thres 0.25 --iou-thres 0.6```
  - **Detect using images**:
    - ```python detect.py --weights runs/train/exp7/weights/best.pt --source ../dataset/images/test/*.jpg --save-csv```
  - **Detect using video**:
    - ```python detect.py --weights runs/train/exp7/weights/best.pt --source ../assignment/sample.mp4 --video --save-csv```
  - **Detect using video (self-written)**:
    - ```python inference.py ```



 