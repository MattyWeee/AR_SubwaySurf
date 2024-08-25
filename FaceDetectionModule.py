import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
    
    def findFaces(self, img, highlight=set(), draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        if draw and results.detections:
            for id, detection in enumerate(results.detections):
                if draw: self.mpDraw.draw_detection(img, detection)
                self.bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = detection.location_data.relative_bounding_box
                bbox_start = (int(bbox.xmin*w), int(bbox.ymin*h))
                bbox_dims = (int(bbox.width*w), int(bbox.height*h))
                bbox_end = (bbox_start[0]+bbox_dims[0], bbox_start[1]+bbox_dims[1])
                cv2.rectangle(img, bbox_start, bbox_end, (255, 0, 255))
                self.drawCorners(img, bbox_start, bbox_dims, bbox_end)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', bbox_start, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
    
    def drawCorners(self, img, bbox_start, bbox_dims, bbox_end, t=5):
        x1, y1 = bbox_start
        x2, y2 = bbox_start[0]+bbox_dims[0], bbox_start[1]
        x3, y3 = bbox_start[0], bbox_start[1]+bbox_dims[1]
        x4, y4 = bbox_end
        l = int((bbox_dims[0])/5)
        cv2.line(img, (x1, y1), (x1+l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1+l), (255, 0, 255), t)
        cv2.line(img, (x2, y2), (x2-l, y2), (255, 0, 255), t)
        cv2.line(img, (x2, y2), (x2, y2+l), (255, 0, 255), t)
        cv2.line(img, (x3, y3), (x3+l, y3), (255, 0, 255), t)
        cv2.line(img, (x3, y3), (x3, y3-l), (255, 0, 255), t)
        cv2.line(img, (x4, y4), (x4-l, y4), (255, 0, 255), t)
        cv2.line(img, (x4, y4), (x4, y4-l), (255, 0, 255), t)

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        detector.findFaces(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()