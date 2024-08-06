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
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Preprocess the frame
    img = preprocess_image(frame)

    # Predict using the model
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Display the predicted number of fingers
    cv2.putText(frame, f'Fingers: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Finger Count', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()