import cv2
import os
import numpy as np
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model('ASL_Model.h5')
scaler = StandardScaler()
def process(img):
    img = preprocess(img)
    img = np.reshape(img, (1, 784))
    data = scaler.fit_transform(img)
    data = np.resize(data, (1, 28, 28, 1))

    prediction = model.predict(data)
    print("Predicted: " + str(prediction))

def preprocess(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data_arr = np.asarray( img_gray )
    cropped_data = data_arr#[100:300, 50:250]
    cv2.imshow('test', cropped_data)
    rsimage = cv2.resize(cropped_data, (28,28))
    cv2.imshow('blah', rsimage)
    return rsimage


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
count = 0;

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    cv2.rectangle(frame, (50, 100), (250, 300), (0, 255, 0), 2)

    if key == 27: # exit on ESC
        break

    elif key == 112: # key is 'p'

        process(frame)

    count += 1


cv2.destroyWindow("preview")
vc.release()