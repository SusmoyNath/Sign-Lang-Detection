# Sign Language Detection

This project aims to detect hand gestures representing different letters and numbers in American Sign Language (ASL) using a deep learning model. It uses a pre-trained model to recognize gestures captured through a webcam and provides a real-time visual output along with speech feedback.

## Features

- **Real-time Hand Gesture Recognition**: Detects ASL hand gestures in real time from a webcam feed.
- **Speech Feedback**: Provides spoken output of the detected gesture using **pyttsx3**.
- **User-friendly Interface**: Visual feedback with the predicted sign and accuracy shown on the screen.
- **Support for ASL Characters**: The model is capable of recognizing both numeric (0-9) and alphabetic (A-Z) gestures.

## Requirements

Before running this project, you need to install the necessary dependencies.

### 1. Python

Ensure you have Python 3.6 or later installed. You can download Python from [here](https://www.python.org/downloads/).

### 2. Install Dependencies

You need to install the following Python packages:

- **OpenCV** for video capturing and image processing
- **TensorFlow** for deep learning model
- **pyttsx3** for text-to-speech conversion

You can install all required dependencies by running:

```bash
pip install opencv-python tensorflow pyttsx3 numpy
```

### 3. Additional Tools

Ensure you have a working webcam for capturing the hand gestures.

---

## Project Structure

Here’s a quick overview of the project structure:

```
Sign-Language-Detection/
├── model/
│   ├── sign-lang-detection.json
│   └── sign-lang-detection.h5
├── realtimedetection.py
├── README.md
└── requirements.txt
```

- `model/sign-lang-detection.json`: The architecture of the pre-trained model.
- `model/sign-lang-detection.h5`: The weights of the pre-trained model.
- `realtimedetection.py`: The main script to run the real-time hand gesture detection.
- `README.md`: The documentation for the project (this file).
- `requirements.txt`: A list of all Python packages required for the project.

---

## How to Use

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/Sign-Language-Detection.git
```

### 2. Set Up the Environment

Navigate to the project directory and install the required dependencies:

```bash
cd Sign-Language-Detection
pip install -r requirements.txt
```

### 3. Run the Real-Time Detection

To start the real-time hand gesture recognition, run the following command:

```bash
python realtimedetection.py
```

This will open the webcam window, where the system will start detecting the ASL hand gestures.

---

## How It Works

1. **Model Architecture**: The model is a Convolutional Neural Network (CNN) trained to recognize 36 ASL characters (0-9 and A-Z). The model's architecture and weights are stored in `sign-lang-detection.json` and `sign-lang-detection.h5` respectively.

2. **Webcam Feed**: The `cv2.VideoCapture` function is used to capture real-time video feed from the webcam.

3. **Hand Gesture Preprocessing**: The hand gestures are first cropped from the video feed, converted to grayscale, and resized to 48x48 pixels. These are then passed through the trained model.

4. **Prediction**: The model predicts the hand gesture and outputs the corresponding letter or number along with the confidence (accuracy).

5. **Speech Output**: The predicted label is spoken using the **pyttsx3** library.

---

## Example Output

Upon detecting a gesture, the following happens:

- A rectangle is drawn around the detected hand gesture.
- The predicted label (e.g., 'A', 'B', '1', etc.) is displayed at the top of the screen.
- The accuracy of the prediction is shown alongside the label.
- The predicted gesture is spoken aloud using text-to-speech.

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pyttsx3'`

Solution: Install the `pyttsx3` module using the following command:

```bash
pip install pyttsx3
```

### Error: `ValueError: Input 0 of layer is incompatible`

Solution: Ensure that the image is resized correctly and matches the input shape expected by the model (e.g., 48x48 pixels for grayscale images).

---

## Contributions

Feel free to contribute to this project. You can do so by:

- Reporting bugs or issues
- Submitting pull requests for improvements or features

---

## Acknowledgements

- OpenCV: For video capture and image processing.
- TensorFlow: For building and training the deep learning model.
- pyttsx3: For text-to-speech conversion.
- The dataset used for training the model (if applicable).

---

This README provides an overview of the project, installation instructions, and usage guidelines. Feel free to modify it according to your specific requirements or additions you’ve made to the project.
