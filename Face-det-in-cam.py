import cv2
import os

# Get user supplied values
imagePath = 'C:/Users/Admin/Documents/Programing/Voice-and-face-recognition/og.jpeg'
cascPath = 'C:/Users/Admin/Documents/Programing/Voice-and-face-recognition/haarcascade_frontalface_default.xml'
#cv2.namedWindow("Faces found", cv2.WINDOW_FREERATIO)


# Create the haar cascade
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
#image = cv2.imread(imagePath)
cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame
        , (x, y), (x+w, y+h), (0, 255, 0), 2)

    #frame = cv2.resize(frame,(480,852))
    cv2.imshow("Faces found", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

