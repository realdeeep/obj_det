import cv2
import numpy as np
from keras.models import load_model

# Load the trained CNN model
model = load_model('CNNmodel.keras')

# Create a dictionary mapping label indices to characters
label_mapping = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y',25: 'Z'}

# Open the webcam (0 indicates the default camera)
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the frame for prediction
    img = cv2.resize(gray_frame,(28, 28))  # Resize the grayscale frame to match the input size of your model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(img)

    # Get the predicted label index
    predicted_label = np.argmax(predictions)

    # Get the corresponding character from the label mapping
    sign_character = label_mapping[predicted_label]

    # Display the frame with the predicted character
    cv2.putText(frame, sign_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()