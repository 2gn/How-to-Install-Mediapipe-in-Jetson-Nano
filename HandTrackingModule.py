from mediapipe import solutions
from time import time
from cv2 import (
    circle as cv2_circle,
    cvtColor as cv2_cvtColor,
    COLOR_BGR2RGB as cv2_COLOR_BGR2RGB,
    VideoCapture as cv2_VideoCapture,
    CAP_PROP_FRAME_WIDTH as cv2_CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT as cv2_CAP_PROP_FRAME_HEIGHT,
    FONT_HERSHEY_SIMPLEX as cv2_FONT_HERSHEY_SIMPLEX,
    FONT_HERSHEY_PLAIN as cv2_FONT_HERSHEY_PLAIN,
    FILLED as cv2_FILLED,
    putText as cv2_putText,
    imshow as cv2_imshow,
    waitKey as cv2_waitKey,
    destroyAllWindows as cv2_destroyAllWindows,
)

mpHands = solutions.hands
drawing_utils = solutions.drawing_utils

LOG_TO_STDOUT = False

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.hands = mpHands.Hands(
            mode,
            maxHands,
            detectionCon,
            trackCon
        )
    
    def drawSegments(self, img, results):
        for handLms in results:
            drawing_utils.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )
    
    def drawBones(self, img, cxys):
        for cxy in cxys:
            cv2_circle(img, cxy, 15, (255, 0, 255), cv2_FILLED)

    def findHands(self, img, draw=True):
        results = self.hands.process(
            # convert image to grayscale so that it can easily be processed
            cv2_cvtColor(img, cv2_COLOR_BGR2RGB)
        ).multi_hand_landmarks
        return img, results

    def findPosition(self, img, results, handNo=0, draw=True):
        lmList = []
        cxys = []
        myHand = results[handNo] if results else None
        if results:
            for id, lm in enumerate(myHand.landmark):
                stdlog(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                stdlog(id, cx, cy)
                lmList.append([id, cx, cy])
                cxys.append((cx, cy))
        # return list of landmarks as a list 'lmList'
        # cx and cy are positions of points to be drawn on the image.
        return lmList, cxys

def stdlog(*value):
    if LOG_TO_STDOUT:
        print(value)

def main():
    pTime = 0
    draw = True
    cap = cv2_VideoCapture(0)
    dispW = 640
    dispH = 480
    font=cv2_FONT_HERSHEY_SIMPLEX
    cap.set(cv2_CAP_PROP_FRAME_WIDTH,dispW)
    cap.set(cv2_CAP_PROP_FRAME_HEIGHT,dispH)
    detector = handDetector()

    while True:
        _, img = cap.read()
        img, results = detector.findHands(img)
        lmList, cxys = detector.findPosition(img, results)

        if results and draw:
            detector.drawSegments(img, results)
            detector.drawBones(img,cxys)

        if len(lmList) != 0:
            stdlog(lmList[4])

        stdlog(lmList[4] if len(lmList) > 0 else "no hands detected")
        cTime = time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2_putText(
            img,
            str(int(fps)),
            (10, 70),
            cv2_FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3
        )
        cv2_imshow("Image", img)
        if cv2_waitKey(1) == 'q':
            break
    cap.release()
if __name__ == "__main__":
    main()
    cv2_destroyAllWindows()