import os
import cv2
import numpy as np
from PIL import Image


def fruits360_parsing_ver2():
    try:
        print("Parsing Start")
        image = []
        label = []

        filepath_fruit360 = "C:\\Users\\eng\\Desktop\\PyTorch_data\\fruits360\\split_data"

        # Define a list of classes and their corresponding labels
        classes = ['apple', 'banana', 'orange', 'strawberry']
        class_labels = {class_name: i for i, class_name in enumerate(classes)}

        # File parse
        for class_name in classes:
            for idx in range(2500):
                img = Image.open(filepath_fruit360 + "/" + class_name + "/" + class_name + " (" +
                                 str(idx+1) + ").jpg").convert("RGB")
                img = np.array(img)
                img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
                img = np.transpose(img, (2, 0, 1))
                image.append(img)
                label.append(class_labels[class_name])
        # Image & label to ndarray
        image_np = np.array(image, dtype=np.uint8)
        label_np = np.array(label, dtype=np.uint8)

        # Image normalization (0.0~1.0)
        image_np = np.divide(image_np, 255.0, dtype=np.float32)

        # Shuffle the data
        shuffled_indices = np.random.permutation(len(image_np))
        image_np = image_np[shuffled_indices]
        label_np = label_np[shuffled_indices]

        # Split the data
        split = int(0.8 * len(image_np))
        X_train, X_test = image_np[:split], image_np[split:]
        y_train, y_test = label_np[:split], label_np[split:]

        return [X_train, y_train], [X_test, y_test], classes

    except Exception as e : print("Exception :", e)


if __name__ == "__main__":
    fruits360_parsing()
