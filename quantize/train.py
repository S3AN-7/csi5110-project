import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import train_generator, val_generator, test_generator
from model import model
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import vitis_quantize


# Set up directories
checkpoint_dir = "models"
os.makedirs(checkpoint_dir, exist_ok=True)
plot_dir = "logs/plots"
os.makedirs(plot_dir, exist_ok=True)

# Uncomment 21-49 to train
# # Callbacks for saving best model and early stopping
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(checkpoint_dir, "adam_100_da_best.keras"),
#         monitor='val_accuracy',
#         save_best_only=True
#     )
# ]

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=callbacks, verbose=1)

# # Plot Training & Validation Loss
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss over Epochs")
# plt.legend()
# plt.grid()
# plt.savefig("logs/plots/adam_100_da_loss.png")

# # Plot Training & Validation Accuracy
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Accuracy over Epochs")
# plt.legend()
# plt.grid()
# plt.savefig("logs/plots/adam_100_da_accuracy.png")

model = tf.keras.models.load_model("models/adam_100_da_best.keras")

loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Quantize and store AI models
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=train_generator)
quantized_model.save("models/adam_100_da_best_quantized.h5")

quantized_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loss, accuracy = quantized_model.evaluate(test_generator, verbose=1)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
