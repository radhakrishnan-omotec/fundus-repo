import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

# Function to make predictions
def prediction(path, model, classes_dict):
    """
    This function takes an image file path, loads a pre-trained model,
    preprocesses the image, and makes a prediction using the model.

    Parameters:
    - path: The path to the input image.
    - model: The pre-trained model for classification.
    - classes_dict: A dictionary mapping class indices to human-readable labels.

    Returns:
    - img: The original input image.
    - predicted_class: The predicted class label.
    """

    # Load and preprocess the image
    img = load_img(path, target_size=(180, 180))  # Resizing the image to the model input size
    img_arr = img_to_array(img)                   # Convert image to array
    processed_img_arr = preprocess_input(img_arr) # Preprocess image for model compatibility

    # Expand image dimensions to match the input shape for the model (batch_size, height, width, channels)
    img_exp_dim = np.expand_dims(processed_img_arr, axis=0)

    # Make predictions using the model
    prediction_probabilities = model.predict(img_exp_dim)
    pred_class_index = np.argmax(prediction_probabilities)
    predicted_class = classes_dict[pred_class_index]
    predicted_probability = prediction_probabilities[0][pred_class_index]

    # Display the predicted class and probability
    print(f"Predicted Class: {predicted_class} with probability {predicted_probability:.2f}")

    # Plot the input image with the prediction
    plt.imshow(img)
    plt.axis('off')  # Remove axis
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

    return img, predicted_class

# Function to load the model and perform prediction
def predict(image):
    """
    This function takes an image object from the user interface,
    loads a pre-trained model, and makes a prediction.

    Parameters:
    - image: The image object passed from the user interface.

    Returns:
    - img: The processed image.
    - predicted_class: The predicted class label.
    """

    # Load the pre-trained model (ensure the path to your model is correct)
    model = load_model("/content/drive/MyDrive/1.ALL-RESEARCH/0-MANVY & ADVIK/CNN_drunksober_classification_model.keras")  # Adjust the path accordingly

    # Extract the file path from the image object
    path = image.name

    # Define your class dictionary mapping indices to class names
    print("classes_dict_cleaned : ",classes_dict_cleaned)

    # Call the prediction function
    img, predicted_class = prediction(path, model, classes_dict_cleaned)

    return img, predicted_class

# Main code to call the prediction function
if __name__ == "__main__":
    # Example path to an image
    image_path = "/content/drive/MyDrive/1.ALL-RESEARCH/0-MANVY & ADVIK/DRUNK/drunk10.png"

    # Assuming image is an object similar to what would be passed by a UI (for testing)
    class ImageObject:
        def __init__(self, name):
            self.name = name

    # Create an image object
    image = ImageObject(image_path)

    # Call the predict function
    img, predicted_class = predict(image)

    # Optionally, print the predicted class for logging
    print(f"Final Predicted Class: {predicted_class}")