from creating_classes import preprocess_image
from keras.models import load_model
import numpy as np
import cv2

model = load_model('model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        cv2.imshow('Finger Count', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    hand_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(hand_contour)
    
    if w * h < frame.shape[0] * frame.shape[1] * 0.5:
        roi = frame[y:y+h, x:x+w]

        img = preprocess_image(roi)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Fingers: {predicted_class}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Finger Count', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()