import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import label_mapping
import os
import random

model = tf.keras.models.load_model("models/adam_100_da_best.keras")

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0) / 255.0
    return image, image_array

def predict(model, image_array):
    predictions = model.predict(image_array)
    return predictions[0]

calib_image_path = "./data/test/"

# Step 1: Collect all image paths
all_image_paths = []

for sub_dir_name in os.listdir(calib_image_path):
    sub_dir_path = os.path.join(calib_image_path, sub_dir_name)
    if os.path.isdir(sub_dir_path):
        for filename in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, filename)
            all_image_paths.append(file_path)

# Step 2: Shuffle all image paths
random.shuffle(all_image_paths)

correct = 0
# Step 3: Process a limited number of random images
for cnt, file_path in enumerate(all_image_paths[:10]):
    original_image, image_array = preprocess_image(file_path)
    probabilities = predict(model, image_array)
    key = [k for k, v in label_mapping.items() if v == np.argmax(probabilities)]
    card_type = os.path.basename(os.path.dirname(file_path))
    print(file_path)
    print(f"True label: {card_type}, Predicted label: {key}")
    if (card_type == key[0]):
        correct += 1

print(f"Final accuracy: {correct/cnt}")    