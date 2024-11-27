import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Define the TensorFlow Lite model path and class names
TF_MODEL_FILE_PATH = 'model_fundus.tflite'  # The default path to the saved TensorFlow Lite model
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']  # Replace with actual class names

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the frame
def preprocess_frame(frame, target_size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, target_size)  # Resize to the target size
    img_array = np.array(img).astype(np.float32)

    # Normalize the image if the model expects it (e.g., values between 0 and 1)
    img_array /= 255.0

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Get the model's input shape
input_shape = input_details[0]['shape'][1:3]  # Assuming input shape (height, width)

# Access the webcam
cap = cv2.VideoCapture(0)  # Open the default camera (change the index if needed)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame, input_shape)

    # Ensure the input tensor's type matches the model's expected type
    input_data = input_data.astype(input_details[0]['dtype'])

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class index and name
    predicted_class_index = np.argmax(output_data[0])
    predicted_class_name = class_names[predicted_class_index]

    # Display the prediction on the frame
    cv2.putText(
        frame,
        f"Prediction: {predicted_class_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the frame with prediction
    cv2.imshow("Webcam Feed", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()