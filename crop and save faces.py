import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os

# List of video files
lst = [r'faces_video.mp4', r'854097-hd_1920_1080_25fps.mp4', r'3253272-uhd_3840_2160_25fps.mp4', r'4265036-uhd_3840_2160_30fps.mp4']

# Select the video file to process
video_path = lst[3]

# Path to save cropped images
save_path = "D:\\New folder\\Face Detection\\Frames\\cropped_images"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Initialize video capture object
cap = cv2.VideoCapture(video_path)

# Initialize FaceDetector object
detector = FaceDetector(minDetectionCon=0.7, modelSelection=1)

# Frame index for image filenames
frame_index = 0
face=0
# Process each frame in the video
while True:
    # Read frame-by-frame
    success, img = cap.read()

    # If no more frames, break the loop
    if not success:
        break

    # Detect faces in the frame
    img, bboxs = detector.findFaces(img, draw=False)

    # If faces are detected
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            face_img = img[y:h, x:w]
            save_name=save_path+"\\"+str(face)+".jpeg"
            print(f"Face dimensions: x={x}, y={y}, w={w}, h={h}")
            if face_img is not None:
                cv2.imwrite(save_name, face_img)
                face+=1
            else:
                print("Empty face image detected.")


    # Resize image for display
    img_resized = cv2.resize(img, (800, 650))

    # Display the image
    cv2.imshow("Image", img_resized)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
