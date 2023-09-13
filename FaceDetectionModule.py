import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)


class FaceDetector:
    def __init__(self, detectCon=0.5, model_selection=0):

        self.detectCon = detectCon
        self.model_selection = model_selection
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(detectCon, model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, name='None',draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    # cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                    #             (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    #             2, (255, 0, 255), 2)

                    cv2.putText(img, f'{name}',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs


def main():
    detector = FaceDetector()
    while True:
        ret, frame = cap.read()
        frame, bboxs = detector.findFaces(frame)
        cv2.imshow("My cam", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
