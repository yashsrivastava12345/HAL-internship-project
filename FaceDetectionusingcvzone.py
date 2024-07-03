import cvzone
from deepface import DeepFace
from cvzone.FaceDetectionModule import FaceDetector
from skimage.metrics import structural_similarity as ssim
import cv2
import os
lst=[r'faces_video.mp4',\
     r'854106-hd_1920_1080_25fps.mp4',\
     r'3253272-uhd_3840_2160_25fps.mp4',\
     r'4265036-uhd_3840_2160_30fps.mp4',\
     r'D:/New folder/Face Detection/6550420-uhd_3840_2160_25fps.mp4',\
     r'D:/New folder/Face Detection/5198159-uhd_3840_2160_25fps.mp4',\
     r'D:/New folder/Face Detection/5199627-uhd_3840_2160_25fps.mp4',\
     r'D:\\New folder\\Face Detection\\3026357-uhd_3840_2160_30fps.mp4']
cap = cv2.VideoCapture(lst[6])
path="D:\\New folder\\Face Detection\\Frames\\croped images"
detector = FaceDetector(minDetectionCon=0.7, modelSelection=1)
face=1
while True:
    success, img = cap.read()
    check=0
    if success:
        img, bboxs = detector.findFaces(img, draw=False)
        if bboxs:
            for bbox in bboxs:
                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score = int(bbox['score'][0] * 100)
                if(score>=85):
                    crop=img[y-250:y+h+100, x-250:x+w+100]
                    name=path+"\\"+str(face)+".jpeg"
                    cv2.imwrite(name,crop)
                    face+=1
                    check+=1
                    lstdir=os.listdir(path)
                    for i in lstdir:
                        j=cv2.imread(i)
                        #print(i)
                        crop2=cv2.cvtColor(cv2.resize(crop,[100,100]),cv2.COLOR_BGR2GRAY)
                        compimg=cv2.cvtColor(cv2.resize(j if j is not None else crop,[100,100]),cv2.COLOR_BGR2GRAY)
                        print(j if j is not None else "hello")
                        similarity_index, _ = ssim(crop2, compimg, full=True)
                        if():
                            pass
                        #print(similarity_index)
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                cvzone.cornerRect(img, (x, y, w, h))
        imge=cv2.resize(img,(800,650))
        cv2.imshow("Image", imge)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

