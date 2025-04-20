from keras.models import model_from_json
import cv2
import numpy as np

# Load model architecture and weights
json_file = open("sign-lang-detection.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("sign-lang-detection.h5")

# Function to extract features from the image
def extract_features(image):
    image = cv2.resize(image, (150, 150))             # Resize to match model input
    image = np.array(image).reshape(1, 150, 150, 1)    # Reshape for model
    return image / 255.0                               # Normalize

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the labels from 0-9 and a-z
label = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle for the ROI
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)

    # Crop and preprocess the ROI
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = extract_features(cropframe)

    # Predict the sign language
    pred = model.predict(cropframe)
    prediction_label = label[np.argmax(pred)]  # Label with highest probability

    # Show the predicted label and accuracy
    accuracy = "{:.2f}".format(np.max(pred) * 100)
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    cv2.putText(frame, f'{prediction_label}  {accuracy}%', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit on pressing 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
