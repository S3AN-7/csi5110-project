import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pynq_dpu import DpuOverlay


def visualize_predictions(original_image, probabilities, class_names):
    plt.figure(figsize=(10, 8))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Show probabilities
    plt.subplot(1, 2, 2)
    y_positions = np.arange(len(class_names))
    plt.barh(y_positions, probabilities, align="center")
    plt.yticks(y_positions, class_names, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Probability")
    plt.title("Class Predictions")

    plt.tight_layout()
    plt.savefig("pred.png")


def main(img_path):
    # Load overlay and model
    overlay = DpuOverlay("dpu.bit")
    overlay.load_model("adam_100_da_best_quantized.xmodel")

    # Load labels and encoder
    df = pd.read_csv('cards.csv')
    label_encoder = LabelEncoder()
    df["encoded_labels"] = label_encoder.fit_transform(df["labels"])
    class_names = label_encoder.classes_

    dpu = overlay.runner

    input_tensor = dpu.get_input_tensors()[0]
    output_tensor = dpu.get_output_tensors()[0]
    input_dim = tuple(input_tensor.dims)
    output_dim = tuple(output_tensor.dims)
    fix_point = input_tensor.get_attr("fix_point")
    scale = 2 ** fix_point

    # Load and preprocess image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: could not load image {img_path}")
        return

    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = (image / 255.0 * scale).round().astype(np.int8)

    input_data = [np.empty(input_dim, dtype=np.int8)]
    input_data[0][0] = image

    output_data = [np.empty(output_dim, dtype=np.int8)]

    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    probabilities = output_data[0][0]
    class_idx = probabilities.argmax()
    print("Your card is:", class_names[class_idx])

    visualize_predictions(original_image, probabilities, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a playing card image.")
    parser.add_argument("image_path", help="Path to the test image")
    args = parser.parse_args()
    main(args.image_path)