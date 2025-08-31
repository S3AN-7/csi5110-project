import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import label_mapping

model = tf.keras.models.load_model("models/adam_100_da_best.h5")

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0) / 255.0
    return image, image_array

def predict(model, image_array):
    predictions = model.predict(image_array)
    return predictions[0]

def visualize_predictions(original_image, probabilities):
    plt.figure(figsize=(10, 8))  # Increased figure height for better spacing

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis("off")

    # Show probabilities
    plt.subplot(1, 2, 2)
    y_positions = np.arange(len(class_names))
    plt.barh(y_positions, probabilities, align="center")

    # Increase spacing
    plt.yticks(y_positions, class_names, fontsize=8)  # Reduce font size for better readability
    plt.gca().invert_yaxis()  # Keep highest probability at the top
    plt.xlabel("Probability")
    plt.title("Class Predictions")

    plt.tight_layout()
    plt.savefig("logs/plots/pred.png")

test_image = "data/test/five of hearts/4.jpg"
original_image, image_array = preprocess_image(test_image)
probabilities = predict(model, image_array)
print(probabilities)

key = [k for k, v in label_mapping.items() if v == np.argmax(probabilities)]
print(f"Predicted label: {key}")

class_names = list(label_mapping.keys())
visualize_predictions(original_image, probabilities)