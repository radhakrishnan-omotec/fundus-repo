import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
import gradio as gr
from tensorflow.keras.preprocessing.image import img_to_array

# Classes dictionary
classes_dict = {
    0: 'Mild',
    1: 'Moderate',
    2: 'No DR',
    3: 'Proliferative DR',
    4: 'Severe'
}

# Load TFLite model and allocate tensors
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Predict Diabetic Retinopathy stage for a single frame
def predict_frame(frame, interpreter, classes_dict):
    # Resize and preprocess the frame
    frame_resized = cv2.resize(frame, (180, 180))
    img_array = img_to_array(frame_resized)
    processed_img = preprocess_input(img_array)
    input_data = np.expand_dims(processed_img, axis=0).astype(np.float32)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_class = classes_dict[predicted_index]
    confidence = output_data[0][predicted_index]

    return predicted_class, confidence

# Process video and make predictions
def process_video(video_path, tflite_model_path, classes_dict):
    interpreter = load_tflite_model(tflite_model_path)
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Make prediction on the current frame
        predicted_class, confidence = predict_frame(frame, interpreter, classes_dict)
        predictions.append(f"{predicted_class} (Confidence: {confidence:.2f})")

    cap.release()
    return predictions

# Gradio wrapper
def predict_video(video):
    tflite_model_path = r"C:\Users\OMOLP049\Documents\AnushkaManoj\fundus-repo-main\model_fundus.tflite"
    predictions = process_video(video, tflite_model_path, classes_dict)  # Removed .name
    return "\n".join(predictions)


# Main function for Gradio UI
def main():
    # Create Gradio interface
    io = gr.Interface(
        fn=predict_video,
        inputs=gr.Video(label="Upload Retinal Fundus Video"),
        outputs=gr.Textbox(label="DR Stage Predictions for Video "),
        title="Portable Fundus Imaging Device for DR Screening",
        
        theme=gr.themes.Soft(),
    )
    io.launch(share=True)


if __name__ == "__main__":
    main()
