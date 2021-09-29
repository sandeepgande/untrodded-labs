import tensorflow
import cv2 as cv
import numpy as np
from tensorflow.keras.models import model_from_json
import time

json_file = open('C:/Users/gv/MLPROJECTS/untrodden-labs-task/casting_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("C:/Users/gv/MLPROJECTS/untrodden-labs-task/casting_model/model.h5")

prev_frame_time = 0
new_frame_time = 0

capture = cv.VideoCapture(0)
while True:
    isTrue,frame = capture.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    frame_resized = cv.resize(gray, (50, 50),interpolation = cv.INTER_LINEAR)
    frame_resized = frame_resized.reshape(1,50,50,1)
    frame_resized = frame_resized/255.0
    
    #display frame rate
#     font = cv.FONT_HERSHEY_SIMPLEX
#     new_frame_time = time.time()
#     fps = 1/(new_frame_time-prev_frame_time)
#     prev_frame_time = new_frame_time
#     fps = int(fps)
#     fps = 'frame rate : '+str(fps)
#     cv.putText(frame, fps, (7, 70), font, 1, (100, 255, 0), 3, cv.LINE_AA)
    
    prediction = model.predict(frame_resized)
    if np.argmax(prediction) == 1 and np.max(prediction) >= 0.75:
        print('<defect type: no>')
    else: 
        pass

    cv.imshow('video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()