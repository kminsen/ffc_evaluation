import cv2
import os
import torch



if __name__=='__main__':
    if not os.path.exists('allframes'):
        os.makedirs('allframes')
    vidcap = cv2.VideoCapture('assignment/sample.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("allframes/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    # path ="frames"
    
    # file_list = []
    # for root,dirs,files in os.walk(path):
    #     for file in files:
    #         file_list.append(file[:-4])
    # print(file_list)
    # for file in file_list:
    #     with open(f'annotated_staff_frames/{file}.txt', 'w') as fp:
    #         pass

