import cv2
from cvzone.FaceDetectionModule import FaceDetector as cz
import cvzone as cz1
from deepface import DeepFace as df
import dlib as db
from PIL import Image
import imagehash
import skimage as ski
from skimage.metrics import structural_similarity as ss
import os
from win32api import GetSystemMetrics
lst=[[]]
height,width=GetSystemMetrics(0),GetSystemMetrics(1)
path=[r'Video\\faces_video.mp4',\
     r'Video\\854106-hd_1920_1080_25fps.mp4',\
     r'Video\\3253272-uhd_3840_2160_25fps.mp4',\
     r'Video\\4265036-uhd_3840_2160_30fps.mp4',\
     r'Video\\6550420-uhd_3840_2160_25fps.mp4',\
     r'Video\\5198159-uhd_3840_2160_25fps.mp4',\
     r'Video\\5199627-uhd_3840_2160_25fps.mp4',\
     r'3026357-uhd_3840_2160_30fps.mp4',\
     r'Video\\gettyimages-1214537099-640_adpp.mp4',\
     "DB",\
     r'D:\\New folder(2)\\Frames\\croped images']

cap=cv2.VideoCapture(path[5])
dete=db.get_frontal_face_detector()
predct=db.shape_predictor('shape_predictor_68_face_landmarks.dat')
def image_match(video_frame_path, directory_path, threshold=0.07):
    frame = cv2.cvtColor(video_frame_path, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame, None)
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(directory_path, filename)
            directory_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            kp2, des2 = orb.detectAndCompute(directory_image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            good_matches = [m for m in matches if m.distance < threshold * len(kp1)]
            #print(threshold * len(kp1),good_matches)
            if len(good_matches) > 10: 
                return True 

    return False
cnt=[]
while (True):
    run,img=cap.read()
    if(run):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=dete(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        for face in faces:
            if face is not None:
                X1,Y1,X2,Y2=face.left(),face.top(),face.right(),face.bottom()
                crop = img[max(0, Y1 - 250):min(gray.shape[0], Y2 + 250),max(0, X1 - 250):min(gray.shape[1], X2 + 250)]
                matched=image_match(crop,path[(len(path)-2)])
                if face in lst:
                    cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                    cv2.putText(img, 'Matched', (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    lst=[]
                    pass
                elif (matched):
                    cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                    cv2.putText(img, 'Matched', (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    lst.append(face)
                else:
                    cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                    cv2.putText(img, 'Mismatched', (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)                
            else:
                continue
            pass
        #lst=[]
        imge=cv2.resize(img,[height,width])
        cv2.imshow("Test",imge)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        pass
    else:
        print('video not avilable')
        break
cv2.destroyAllWindows()
