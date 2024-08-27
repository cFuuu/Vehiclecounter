import os
os.add_dll_directory("C:/Users/mason/anaconda3//Library/bin") # 一定要加，不然虛擬環境讀不到。
import cv2
import numpy as np

interval = 1
frame_count = 0
frame_index = 0
inputvideo = "D:/Harry/ITS/Vehiclecounter/Video/image.mp4"
outputvideo = "D:/Harry/ITS/Vehiclecounter/Image/%d.jpg"

cap = cv2.VideoCapture(inputvideo)
if cap.isOpened():
    success = True
else:
    success = False
    print("reading failed")
while(success):
    success, frame = cap.read()
    #frame = frame[0:140,:]
    if frame is not None:
        np.uint8(frame)
        #print(np.dtype(frame))
        if success is False:
            print("---> the %d frame reading failed" % frame_index)
            break
        print("---> Now reading the %d frame:" % frame_index, success)
        print(inputvideo)
        print(outputvideo)
        if frame_index % interval == 0 :
            cv2.imwrite(outputvideo % frame_count, frame)
            frame_count += 1
        frame_index += 1
    else: pass
