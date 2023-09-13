import cv2
from FaceRecognitionModule import encoding_image, recognize_face
from FaceDetectionModule import FaceDetector
import numpy as np

cam = cv2.VideoCapture(0)
detector = FaceDetector()

labels = ['Bill Gates', 'Elon Musk', 'Mark Zuckerburg']

# encoded face processing in the database
encoded_list = np.load('encoded_list.npy')

frame_count = 0

# Delay time each recognition
RECOGNITION_DELAY = 60

name = 'None'

while True:
    ret, frame = cam.read()
    frame, bounding_box = detector.findFaces(frame,name=name)
    cv2.imshow("My cam", frame)

    if len(bounding_box) == 0:
        name = 'None'
    # get all coordinates of faces in camera
    if frame_count > RECOGNITION_DELAY and name == 'None':

        # frame, bounding_box = detector.findFaces(frame, draw=False)
        for i in range(len(bounding_box)):
            id, face_coordinate, detection_score = bounding_box[i]
            face_x, face_y, face_w, face_h = face_coordinate

            # cv2.putText(frame, 'name', (face_x, face_y), cv2.FONT_HERSHEY_PLAIN, 2 , (255, 0, 255), 5)

            # get image of the face
            face_img = frame[max(0, face_y - 100): face_y + face_h + 50, max(0, face_x - 50): face_x + face_w + 50]
            # cv2.imshow("My face", face_img)

            # encoding the face
            encoded_face = encoding_image(face_img)
            print(encoded_face)

            # recognize face and get the label
            if len(encoded_face) > 0:
                face_index = recognize_face(encoded_face, encoded_list)
                if face_index != -1:
                    print(labels[face_index])
                    name = labels[face_index]
                else:
                    print('Can\'t recognize the face')

        frame_count = 0
    frame_count += 1
    cv2.waitKey(1)

