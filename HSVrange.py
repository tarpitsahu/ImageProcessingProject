import cv2
def nothing(x):
    pass
cam = cv2.VideoCapture(0)

cv2.namedWindow("Control")

cv2.createTrackbar("L-H","Control",0,180,nothing)
cv2.createTrackbar("L-S","Control",0,255,nothing)
cv2.createTrackbar("L-V","Control",0,255,nothing)
cv2.createTrackbar("U-H","Control",0,180,nothing)
cv2.createTrackbar("U-S","Control",0,255,nothing)
cv2.createTrackbar("U-V","Control",0,255,nothing)

print(pyautogui.size())



while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    frame = cv2.GaussianBlur(frame,(5,5),0)

    lh = cv2.getTrackbarPos("L-H","Control")
    ls = cv2.getTrackbarPos("L-S","Control")
    lv = cv2.getTrackbarPos("L-V","Control")
    uh = cv2.getTrackbarPos("U-H","Control")
    us = cv2.getTrackbarPos("U-S","Control")
    uv = cv2.getTrackbarPos("U-V","Control")
    
    
    f = cv2.inRange(frame,(lh,ls,lv),(uh,us,uv))
    #f = cv2.bitwise_not(f)

    
    k = cv2.waitKey(1)
    if(k ==27):
        break
    cv2.imshow("Frame",frame)
    cv2.imshow("mask",f)
    



cam.release()



