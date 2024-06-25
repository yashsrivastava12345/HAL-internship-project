import cv2

def detect_faces_opencv(image_path, cascade_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), max_size=(300, 300), roi=None):
    """
    Detects faces in the given image using OpenCV's Haar cascades or LBP cascades.

    Args:
    - image_path (str): Path to the input image.
    - cascade_path (str): Path to the XML file containing the Haar/LBP cascade classifier.
    - scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
    - min_neighbors (int): Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    - min_size (tuple): Minimum possible object size.
    - max_size (tuple): Maximum possible object size.
    - roi (tuple or None): Region of interest (ROI) within the image for detection.

    Returns:
    - List of tuples, each containing bounding box coordinates (x, y, width, height) of a detected face.
    """
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size, maxSize=max_size)

    # Apply ROI if specified
    if roi is not None:
        x, y, w, h = roi
        faces = [face for face in faces if x <= face[0] <= x + w and y <= face[1] <= y + h]

    return faces

# Example usage:
image_path = 'path/to/image.jpg'
cascade_path = 'path/to/haarcascade_frontalface_default.xml'
detected_faces = detect_faces_opencv(image_path, cascade_path)

# Process the detected faces...
