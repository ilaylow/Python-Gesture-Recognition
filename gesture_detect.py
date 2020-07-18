import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)

IMG_SIZE = 70

IMAGE_TYPES = ['fist', 'okay', 'palm', 'peace']

model = tf.keras.models.load_model(r"C:\Users\chuen\Desktop\Python\image_recognition_basics\128x2-CNN-GestureRecog.model")

def prepare(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#predict = model.predict([prepare(r"C:\Users\chuen\Desktop\Python\image_recognition_basics\Python-Gesture-Recognition\test_images\palm_test.jpg")])

"""predict = predict[0]
predict = [float(e) for e in predict]
print(predict)
img_index = predict.index( (max(predict)))
print(img_index)

print(IMAGE_TYPES[int(img_index)])"""

while True:
    ret, frame = cap.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Region Of Interest', (0, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # cv2.line(frame, (0,0), (1000,1000), (255, 255, 255), 30) # (frame for image, start, end, color in BGR)
    cv2.rectangle(frame, (15,90), (300, 380), (0,255,0), 5)

    # Define region of interest for our hand to fit in
    #roi = frame[15:300, 90:380]
    roi = frame[90:380, 15:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    resize_img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    predict = model.predict([resize_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])

    predict = list(predict[0])
    img_index = predict.index((max(predict)))

    print(predict)
    print(img_index)
    print(IMAGE_TYPES[img_index])

    cv2.putText(roi, IMAGE_TYPES[img_index], (100, 100), font, 2, (0, 255, 0), 5, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
