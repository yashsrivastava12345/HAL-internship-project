from yoloface import face_analysis as face
import numpy
import cv2
cap = cv2.VideoCapture(r'faces_video.mp4')
path=r'D:\Face Detection\New folder'
currentframe=0
enable_detection=True
while True: 
    ret, frame = cap.read()
    if ret:
        name = path+"\\"+str(currentframe) + '.jpg'
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

    __,box,conf=face.face_detection(enable_detection,image_path=str(path),frame_arr=frame,frame_status=True,model='full')
    output_frame=face.show_output(img=frame,face_box=box,frame_status=True)
    print(box)

    key=cv2.waitKey(0)
    if key ==ord('v'): 
        break 
cap.release()
cv2.destroyAllWindows()

