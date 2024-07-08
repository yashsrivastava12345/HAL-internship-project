import cv2
from cvzone.FaceDetectionModule import FaceDetector as cz
import cvzone as cz1
from deepface import DeepFace as df
import dlib as db
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ss
import os
from win32api import GetSystemMetrics
import numpy
height,width=GetSystemMetrics(0),GetSystemMetrics(1)
path=[r'Video\\faces_video.mp4',\
     r'Video\\854106-hd_1920_1080_25fps.mp4',\
     r'Video\\3253272-uhd_3840_2160_25fps.mp4',\
     r'Video\\4265036-uhd_3840_2160_30fps.mp4',\
     r'Video\\6550420-uhd_3840_2160_25fps.mp4',\
     r'Video\\5198159-uhd_3840_2160_25fps.mp4',\
     r'Video\\5199627-uhd_3840_2160_25fps.mp4',\
     r'3026357-uhd_3840_2160_30fps.mp4',\
      
     r'D:\\New folder (2)\\Video\\ankur_video.mp4',\
     r'D:\\New folder (2)\\Video\\prachi_video.mp4',\
     r'D:\\New folder (2)\\Video\\they_video.mp4',\
     r'D:\\New folder (2)\\Video\\we_video.mp4',\
     "D:\\New folder (2)\\DB",\
     r'D:\\New folder(2)\\Frames\\croped images']

cap=cv2.VideoCapture(path[6])
dete=db.get_frontal_face_detector()
predct=db.shape_predictor('shape_predictor_68_face_landmarks.dat')
def image_match(video_frame_path, directory_path, threshold=0.08):
    if(video_frame_path is not None):
        cv2.imshow('test2',video_frame_path)
        frame = cv2.cvtColor(video_frame_path, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(frame, None)
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                image_path = os.path.join(directory_path, filename)
                print(image_path)
                directory_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                kp2, des2 = orb.detectAndCompute(directory_image, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)

                good_matches = [m for m in matches if m.distance < threshold * len(kp1)]

                if len(good_matches) > 10: 
                    return True 

    return False

while (True):
    run,img=cap.read()
    if(run):
        faces=dete(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        print(len(faces))
        for face in faces:
            if face is not None:
                X1,Y1,X2,Y2=face.left(),face.top(),face.right(),face.bottom()
                crop=img[Y1-250:Y2+250,X1-250:X2+250]
                #print(X1,Y1,X2,Y2)
                #cv2.imshow('crop',cv2.resize(crop,[100,100]))
                matched=image_match(crop,path[(len(path)-2)])
                if matched:
                    cz1.putTextRect(img, f'Matched', (X1,Y1))
                    cv2.rectangle(img,(X1,Y1),(X2,Y2),(0,255,0),2)
                    pass
                else:
                    cz1.putTextRect(img, f'Miss Matched', (X1,Y1))
                    cv2.rectangle(img,(X1,Y1),(X2,Y2),(0,255,0),2)
                
            else:
                continue
            pass
        imge=cv2.resize(img,[height,width])
        cv2.imshow("Test",imge)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
        pass
    else:
        print("Video not avilable")
        break
