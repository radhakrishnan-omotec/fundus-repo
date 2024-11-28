
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg19 import preprocess_input
import gradio as gr

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

# Prediction function using TFLite model
def predict_with_tflite(image_path, interpreter, classes_dict):
    # Load and preprocess image
    img = load_img(image_path, target_size=(180, 180))
    img_array = img_to_array(img)
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

    return image_path, f"{predicted_class} (Confidence: {confidence:.2f})"

# Gradio wrapper
def predict(image):
    tflite_model_path = r"C:\Users\OMOLP049\Documents\AnushkaManoj\fundus-repo-main\model_fundus.tflite"
    interpreter = load_tflite_model(tflite_model_path)
    return predict_with_tflite(image.name, interpreter, classes_dict)

# Main function for Gradio UI
def main():
    # Create Gradio interface
    io = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload Retinal Fundus Image", file_types=["image"]),
        outputs=[
            gr.Image(label="Uploaded Image"),
            gr.Textbox(label="Diabetic Retinopathy Stage Prediction"),
        ],
        title="Portable Fundus Imaging Device for DR Screening",
        
        theme=gr.themes.Base(
            primary_hue="blue",  # Primary color of the UI
            secondary_hue="green",  # Accent color
            neutral_hue="purple",  # Background and other elements
            text_size="lg",  # Increase text size
        ),
    )
    io.launch(share=True)


if __name__ == "__main__":
    main()
