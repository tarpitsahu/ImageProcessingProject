import cv2
import numpy as np
import dlib
import imutils

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2
	
	@param      background_img    The background image
	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
	@param      x                 x location to place the top-left corner of our overlay
	@param      y                 y location to place the top-left corner of our overlay
	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
	@return     Background image with overlay on top
	"""
	
	bg_img = background_img.copy()
	
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img



cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nose = cv2.imread("noseT.png",-1)
nose1 = cv2.imread("nose.png",-1)
print(nose1.shape)


b,g,r,a = cv2.split(nose)

nose = cv2.merge([b,g,r])


c=0
while True:
    _,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
        landmarks = predictor(gray,face)

        for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                
                #cv2.putText(frame,str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                #cv2.circle(frame,(x,y),1,(0,255,255),-1)
        
        x1 = landmarks.part(31).x
        x2 = landmarks.part(35).x
        y1 = landmarks.part(30).y
        y2 = landmarks.part(55).y

        yy1 = landmarks.part(63).y
        yy2 = landmarks.part(67).y
        
        if(abs(x2-x1)>20):
                print("in if")
                nose = imutils.resize(nose, height=50,width=50)
                mask = imutils.resize(a, height=50,width=50)
        else:
                print("in else")
                nose = imutils.resize(nose, height=20,width=20)
                mask = imutils.resize(a,  height=20,width=20)
        if(abs(yy1-yy2)>10):
                if(c<30 or True):
                        cv2.putText(frame,str("Happy Birthday To You Tanish Bhai!!!"), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,2)
                        cv2.putText(frame,str("May God Bless You!!"), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,2)
                        
                
                c+=1
        if c>60:
                c=0
                
        #mask = cv2.medianBlur(a,5)
        
        not_mask = cv2.bitwise_not(mask)
        h,w,_=nose.shape
        y1=y1-20
        x1=x1-5
        temp = frame[y1:y1+h,x1:x1+w]
        print(temp.shape,nose.shape,mask.shape,not_mask.shape)
        try:
                temp = cv2.bitwise_and(temp,temp,mask = not_mask)
                temp1 = cv2.bitwise_and(nose,nose,mask = mask)
                frame[y1:y1+h,x1:x1+w] = cv2.add(temp,temp1)
        except:
                pass
                
        
    cv2.imshow("Face",frame)
    k = cv2.waitKey(1)
    if(k==27):
        break

cam.release()
cv2.destroyAllWindows()
