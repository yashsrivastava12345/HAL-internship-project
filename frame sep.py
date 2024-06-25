import cv2 
import os 
cam = cv2.VideoCapture("faces_video.mp4") 

currentframe=0
while(True): 
	ret,frame = cam.read()
	

	if ret: 
		name = str(currentframe) + '.jpg'
		print ('Creating...' + name) 
		cv2.imwrite(name, frame) 
		currentframe += 1
	else: 
		break
cam.release() 
cv2.destroyAllWindows() 

