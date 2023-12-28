#!/usr/bin/env python
# coding: utf-8

import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import cvzone

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [20, 50], invert=True)

    idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    ratioList = []
    blinkCounter = 0
    eyeClosed = False
    color = (255, 0, 255)

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            for id in idList:
                cv2.circle(img, face[id], 5, color, cv2.FILLED)

            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)

            if len(ratioList) > 3:
                ratioList.pop(0)
                
            ratioAvg = sum(ratioList) / len(ratioList)

            if ratioAvg < 35 and not eyeClosed:
                blinkCounter += 1
                color = (0, 200, 0)
                eyeClosed = True
            elif ratioAvg >= 35:
                eyeClosed = False

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

            imgPlot = plotY.update(ratioAvg, color)
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, img], 2, 1)

        cv2.imshow("Image", imgStack)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
