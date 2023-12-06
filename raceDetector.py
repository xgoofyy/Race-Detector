import pathlib, cv2, os
from deepface import DeepFace

# Datasets for face and eyes
face_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
eye_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye_tree_eyeglasses.xml"

# Face and eye classifiers
faceCLF = cv2.CascadeClassifier(str(face_cascade_path))
eyeCLF = cv2.CascadeClassifier(str(eye_cascade_path))

# Webcam output
camera = cv2.VideoCapture(0)

# Variables
faceDetection = False
eyeDetection = False
delay = 0
img_counter = 1

while True:
    # Reads frame to look for face and eyes
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCLF.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    eyes = eyeCLF.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    # If a face is detected, a box will be drawn around it, and faceDetection = True
    for (x, y, width, height) in faces:
        faceDetection = True
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

    # If eyes are detected, it will set eyeDetection to True
    for (x, y, width, height) in eyes:
        eyeDetection = True

    # For every 50 iterations of the while loop
    delay += 1
    if (delay % 50 == 0):
        # If a face is detected, a screen shot of the frame is taken
        if faceDetection:
            # Creation of a new .png file of the frame
            newImg = (f"frame_{img_counter}.png")
            cv2.imwrite(newImg, frame)

            # The .png file is read and stored as img
            img = cv2.imread(f"frame_{img_counter}.png")
            
            # DeepFace.analyze analyzes img regardless of face detection, and will detect race
            info = DeepFace.analyze(img, actions = "race", enforce_detection = False)
            
            # The next img created will be named differently from the previous one
            img_counter += 1


        # Prints detection for face, eyes, and race
        print("Face detected: " + str(faceDetection))
        faceDetection = False

        print("Eyes detected: " + str(eyeDetection))
        eyeDetection = False

        race = info[0]["dominant_race"]
        print(f"Race: {race}")

    # Displays the video output
    cv2.imshow("camera", frame)

    # If q is pressed, it stops the video output
    if cv2.waitKey(1) == ord("q"):
        
        i = 1
        while True:
            # Checks for any existing frame_(number).png files, if it exists, it will be deleted
            if os.path.exists(f"frame_{i}.png"):
                os.remove(f"frame_{i}.png")
                i += 1

            # If there are no more remaining frame_(number).png files, it stops checking for images
            else:
                break

        # Stop face detection and analysis
        break

# Confirms the program is completed, releases the camera from the program, and stops any cv2 output windows
print("Done!")
camera.release()
cv2.destroyAllWindows()
