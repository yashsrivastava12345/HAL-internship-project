from deepface import DeepFace
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
lst=[r'faces_video.mp4',r'854097-hd_1920_1080_25fps.mp4',r'3253272-uhd_3840_2160_25fps.mp4',r'4265036-uhd_3840_2160_30fps.mp4']
cap = cv2.VideoCapture(lst[2])
detector = FaceDetector(minDetectionCon=0.7, modelSelection=1)
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))
    imge=cv2.resize(img,(800,650))
    cv2.imshow("Image", imge)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# Example usage
img_path = ("196.jpeg")
db_path = "D:\\New folder\\Face Detection\\DB"
img = ""
# Perform face recognition using DeepFace
try:
    dfs = DeepFace.find(img_path=img_path, db_path=db_path)
    for i in dfs:
        for j in i:
            img=dfs[0][j].to_string(index=False)
            break
        break
    print(img)
    lst=[i for i in img.split(".")]
    list1=[]
    count=0
    for i in lst:
        if(count%2==0):
            list1.append(i)
        else:
            continue
        count+=1

    for i in list1:
        imge = cv2.imread(i+".png")
        if imge is not None:
            cv2.imshow("My image", imge)
            cv2.waitKey(0)
        else:
            print("1Image not found!")
except ValueError:
    print("2Image not found!")
#print(dfs)
#print(dfs[[identity]])


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

