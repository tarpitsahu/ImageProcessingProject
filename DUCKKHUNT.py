import cv2
import numpy as np
import random
import math
from PIL import Image

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
i=0
blue = np.uint8([[[255, 0, 0]]]) #here insert the bgr values which you want to convert to hsv
hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsvBlue)

lowerLimit = hsvBlue[0][0][0] - 10, 100, 100
upperLimit = hsvBlue[0][0][0] + 10, 255, 255

print(upperLimit)
print(lowerLimit)


bg = cv2.imread('duckbg.jpg')
bg = cv2.resize(bg,(640,480),interpolation = cv2.INTER_NEAREST)

duck = cv2.imread('duck2.png',-1)
duck = cv2.resize(duck,(50,50),interpolation = cv2.INTER_NEAREST)


x=random.randint(100, 600)
y=random.randint(100, 300)
cx = 0
cy = 0
total =0
point=0
#(height 480,width 640, 3)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    
    
    
    #frame = cv2.GaussianBlur(frame,(7,7),0)
    f = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #obj======4
    '''b = Image.fromarray(frame)
    duckk = Image.fromarray(duck)
    b.paste(duckk,(x,y))
    frame = np.array(b)
    '''
    
    frame = overlay_transparent(frame, duck, x,y)
    bg2 = overlay_transparent(bg, duck, x,y)
    #cv2.circle(frame,center=(x+25,y+25),radius=15,color=(0,0,255),thickness=-1)
    mask = cv2.inRange(f, (105, 100, 100), (120, 255, 255))

    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        pass
    else:
        segmented = max(cnts, key=cv2.contourArea)
        cx = 0
        cy = 0
        for p in segmented:
            cx += p[0][0]
            cy += p[0][1]
            #cv2.drawContours(frame, [p], -1, (0, 255, 0), 2)
        cx = int(cx/len(segmented))
        cy = int(cy/len(segmented))
        
        #cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)

    if(math.sqrt((x+25-cx)**2 + (y+25-cy)**2)<25):
        point+=1
        i=0
        while(y<400):
            y=y+10
            print("Y:",y)
            ret,new_f = cam.read()
            new_f = cv2.flip(new_f,1)
            new_f = overlay_transparent(new_f, duck, x,y)
            cv2.imshow("Duck Hunt",new_f)
            bg3 = overlay_transparent(bg.copy(), duck, x,y)
            cv2.imshow("Duck Hunt Orig", bg3)



            
            cv2.waitKey(10)
            
        x=random.randint(50, 300)
        y=random.randint(50, 300)

    if(i>50):
        total+=1
        x=random.randint(50, 300)
        y=random.randint(50, 300)
        i=0
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text="Point:"+str(point),org=(0,480),fontFace=font,fontScale=2,color=(0,0,255),thickness=2)
    cv2.putText(bg2,text="Point:"+str(point),org=(0,480),fontFace=font,fontScale=2,color=(0,0,255),thickness=2)
    cv2.imshow("Duck Hunt", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Mas111k", f)
    cv2.imshow("Duck Hunt Orig", bg2)
    
    print(i)
    i+=1
    k = cv2.waitKey(1)
    if(k==27):
        break


cam.release()

    
