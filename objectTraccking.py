import cv2
import time
import numpy as np
import imutils
from imutils.video import FPS
import pyautogui


'''
"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
'''
tracker =[]
tracker.append(cv2.TrackerCSRT_create())
#tracker.append(cv2.TrackerKCF_create())
#tracker.append(cv2.TrackerMIL_create())
#tracker.append(cv2.TrackerBoosting_create())
#tracker.append(cv2.TrackerTLD_create())
#tracker.append(cv2.TrackerMOSSE_create())

cam = cv2.VideoCapture(0)
fps = None
intBB = None 
clr = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(0, 255, 255),(255, 0, 255)]
while True:
    _,frame = cam.read()
    frame = cv2.flip(frame,1)
    if frame is None:
        break

    frame = imutils.resize(frame, width=1920, height=1080)
    (H,W) = frame.shape[:2]

    if intBB is not None:
        i=0
        for t in tracker:
            (success, box) = t.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),clr[i], 2)
            pyautogui.moveTo((x + (x+w))/2, (y + (y+h))/2, duration=0.01)
            i+=1

        fps.update()
        fps.stop()

        print(fps.fps())

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)

    if key == ord("s"):
        intBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        for t in tracker:           
            t.init(frame, intBB)
        fps = FPS().start()
    elif key==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

        
        
    


