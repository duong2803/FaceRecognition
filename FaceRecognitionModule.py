import cv2
import face_recognition
import os
import numpy as np

PATH = './Resources/Images/'
PATH_TEST = './Resources/Test/'

labels = ['Bill Gates', 'Elon Musk', 'Mark Zuckerburg']


def encoding_image(image):
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        encoded = face_recognition.face_encodings(rgbImage)[0]
        # print(encoded)
        return encoded
    except:
        print("No face found")
        return []


def recognize_face(encoded_face, encoded_list):

    distances = []
    for encoded_img in encoded_list:
        print(f'encoded_img: {encoded_img}')
        distances.append(face_recognition.compare_faces(encoded_img, [encoded_face]))

    print(f'distances: {distances}')
    try:
        label_idx = distances.index([True])
        return label_idx
    except:
        return -1


def main():
    encoded_list = []
    for file in os.listdir(PATH):
        f = os.path.join(PATH, file)
        img = cv2.imread(f)
        encoded_img = encoding_image(img)
        encoded_list.append(encoded_img)
    encoded_list = np.array(encoded_list)
    print(encoded_list)
    np.save("encoded_list", encoded_list)
    for file in os.listdir(PATH_TEST):
        t = os.path.join(PATH_TEST, file)
        test_img = cv2.imread(t)
        rgb_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        encoded_test = face_recognition.face_encodings(rgb_test_img)

        print(f'encoded_test: {encoded_test}')
        distances = []
        for encoded_img in encoded_list:
            distances.append(face_recognition.compare_faces(encoded_img, encoded_test))
        label_idx = distances.index([True])

        cv2.putText(test_img, labels[label_idx], (30, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 5)
        cv2.imshow("Image", test_img)
        while True:
            key = cv2.waitKey(33)
            if key == 27:
                break



if __name__ == "__main__":
    main()
