import cv2
import os
import numpy as np
from statistics import mean

cap = cv2.VideoCapture(0)

IMAGE_TYPES = ['fist', 'okay', 'palm', 'peace']

DATADIR = r"C:\Users\chuen\Desktop\Python\image_recognition_basics\Python-Gesture-Recognition\extracted_images"

counter = 0
index = 0

while True:
    ret, frame = cap.read()

    roi = frame[90:380, 15:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if counter == 10:
        counter = 0
        index += 1

        if index == 4:
            break

        print(f"CAPTURING {IMAGE_TYPES[index]} NOW!!!")

    curr_type = IMAGE_TYPES[index]

    # Press key "t" to take a picture
    if cv2.waitKey(1) & 0xFF == ord('t'):
        path = os.path.join(DATADIR, curr_type)
        img_name = curr_type + str(counter) + '.png'
        img_path = path + '\\' + img_name
        print(img_name)
        print(img_path)

        IMG_SIZE = 100
        resize_image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(img_path, roi)

        counter += 1

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # A Frame for the region of interest
    cv2.rectangle(frame, (15,90), (300, 380), (0,255,0), 5)

    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)

cap.release()
cv2.destroyAllWindows()
