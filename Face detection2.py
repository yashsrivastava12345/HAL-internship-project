import cv2
from win32api import GetSystemMetrics
import numpy as np
width,height=GetSystemMetrics(0),GetSystemMetrics(1)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
lst=[r'faces_video.mp4',r'854097-hd_1920_1080_25fps.mp4',r'3253272-uhd_3840_2160_25fps.mp4',r'4265036-uhd_3840_2160_30fps.mp4']
video_capture = cv2.VideoCapture(lst[0])
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return faces
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame
    frame=cv2.resize(video_frame,(width,height))
    cv2.imshow("My Face Detection Project", frame)  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()


