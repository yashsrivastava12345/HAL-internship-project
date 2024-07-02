import cv2 
import os 
cam = cv2.VideoCapture("D:/New folder/Face Detection/5198159-uhd_3840_2160_25fps.mp4")
path="D:\\frames"
currentframe=0
while(True): 
	ret,frame = cam.read()
	

	if ret: 
		name =path +"\\"+ str(currentframe) + '.jpeg'
		print ('Creating...' + name) 
		cv2.imwrite(name, frame) 
		currentframe += 1
	else: 
		break
cam.release() 
cv2.destroyAllWindows() 

