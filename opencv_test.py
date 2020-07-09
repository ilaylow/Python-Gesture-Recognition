import cv2
import numpy as np
from statistics import mean

cap = cv2.VideoCapture(0)

def threshold(imageArray):
    balanceAr = []
    newAr = imageArray
    for eachRow in imageArray:
        for eachPix in eachRow:   # For each pixel we are calculating a mean value
            avgNum = mean(eachPix[:2])
            balanceAr.append(avgNum)  #append it to the balanceAr

    balance = mean(balanceAr)  #calculate the average pixel value from the array of values
    for eachRow in newAr:
        for eachPix in eachRow:
            if mean(eachPix[:2]) > balance:
                for i in range(3):
                    eachPix[i] = 255
            else:
                for i in range(2):
                    eachPix[i] = 0
                eachPix[2] = 255

    return newAr

lower_skin = np.array([66, 40, 25])
upper_skin = np.array([120, 99, 95])

while True:
    ret, frame = cap.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Region Of Interest', (0, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # cv2.line(frame, (0,0), (1000,1000), (255, 255, 255), 30) # (frame for image, start, end, color in BGR)
    cv2.rectangle(frame, (15,90), (300, 380), (0,255,0), 5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define region of interest for our hand to fit in
    #roi = frame[15:300, 90:380]
    roi = frame[90:380, 15:300]

    #Using gaussian blur for clearer imaging (Test)
    blur_roi = cv2.GaussianBlur(roi, (5,5), 0)


    #Threshold testing
    ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Skin Detection, we convert to HSV
    hsv_skin = cv2.cvtColor(blur_roi, cv2.COLOR_BGR2HSV)

    min_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")

    hand_mask = cv2.inRange(hsv_skin, min_HSV,  max_HSV)

    # Get Hand Contours
    contours, val1 = cv2.findContours(hand_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Don't draw contours if they are smaller than a specific area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(roi, contour, -1, (0, 255, 0), 3)

    cv2.imshow('roi', roi)
    #cv2.imshow('hand', hand_mask)
    cv2.imshow('frame', frame)

    #print(roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
