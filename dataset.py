import tensorflow as tf
import os
import kaggle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_folder = "data/"

# Check if the folder exists before downloading
if not os.path.exists(data_folder):
    os.makedirs(data_folder, exist_ok=True)
    # Kaggle API authentication
    # Ensure to store kaggle.json downloaded from your account in ~/.config/kaggle
    # Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /home/pranav/.config/kaggle/kaggle.json'
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('gpiosenka/cards-image-datasetclassification', path=data_folder, unzip=True)
    print("Dataset downloaded successfully.")
else:
    print("Folder already exists and is not empty. Skipping download.")

df = pd.read_csv('./data/cards.csv')
df["filepaths"] = df["filepaths"].apply(lambda x: f"./data/{x}")

label_encoder = LabelEncoder()
df["encoded_labels"] = label_encoder.fit_transform(df["labels"])
label_mapping = dict(zip(label_encoder.classes_,range(len(label_encoder.classes_))))

# First split: 60% train, 40% temporary (val + test)
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["encoded_labels"], random_state=42)

# Second split: 20% validation, 20% test (from the 40% temp)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["encoded_labels"], random_state=42)

# Checking the sizes
# print(len(train_df), len(val_df), len(test_df))

IMG_SIZE = (128,128)
BATCH_SIZE=32

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="filepaths", y_col="labels", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

val_generator = test_datagen.flow_from_dataframe(
    val_df, x_col="filepaths", y_col="labels", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col="filepaths", y_col="labels", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")