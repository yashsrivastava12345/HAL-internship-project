import cv2
import dlib
import os
from win32api import GetSystemMetrics
height,width=GetSystemMetrics(0),GetSystemMetrics(1)
video_paths =[r'Video\\5199627-uhd_3840_2160_25fps.mp4']#[r'Video/gettyimages-1214537099-640_adpp.mp4']#
# [r'Video\\gettyimages-1305120210-640_adpp.mp4'][r'Video\\gettyimages-1350896260-640_adpp.mp4']
directory_path = "DB"
cap = cv2.VideoCapture()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def image_match(video_frame_path, directory_path, threshold=0.65):
    if video_frame_path is not None:
        frame_gray = cv2.cvtColor(video_frame_path, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(frame_gray, None)
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory_path, filename)
                directory_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                kp2, des2 = orb.detectAndCompute(directory_image, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                good_matches = [m for m in matches if m.distance < threshold * len(kp1)]

                if len(good_matches) > 10:
                    return [True,image_path]  
    return [False,None] 
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    dete=0
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue
    
    while True:
        ret, img = cap.read()
        
        if not ret:
            print(f"End of video: {video_path}")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            crop = img[max(0, y1 - 250):min(gray.shape[0], y2 + 250),
                       max(0, x1 - 250):min(gray.shape[1], x2 + 250)]
            matched = image_match(crop, directory_path)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if matched[0] or dete >=1:
                cv2.putText(img, 'Matched', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                dete+=1
            else:
                cv2.putText(img, 'Mismatched', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('Video', cv2.resize(img,[height,width]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()



''' r'Video\\faces_video.mp4',
    r'Video\\854106-hd_1920_1080_25fps.mp4',
    r'Video\\3253272-uhd_3840_2160_25fps.mp4',
    r'Video\\4265036-uhd_3840_2160_30fps.mp4',
    r'Video\\6550420-uhd_3840_2160_25fps.mp4',
    r'Video\\5198159-uhd_3840_2160_25fps.mp4',
    r'Video\\5199627-uhd_3840_2160_25fps.mp4',
    r'3026357-uhd_3840_2160_30fps.mp4',
    r'Video\\ankur_video.mp4',
    r'Video\\prachi_video.mp4',
    r'Video\\they_video.mp4',
    r'Video\\we_video.mp4'
]'''
