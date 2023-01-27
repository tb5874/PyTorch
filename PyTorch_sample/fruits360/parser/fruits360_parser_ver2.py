import os
import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

def fruits360_parsing_ver2():
    try:
        print("Parsing Start")
        image = []
        label = []
        inference_image = []


        # Define a list of classes and their corresponding labels
        classes = ['apple', 'banana', 'orange', 'strawberry']
        class_labels = {class_name: i for i, class_name in enumerate(classes)}


        # Train Test Dataset : -->
        # File parse
        filepath_fruit360 = "C:\\Users\\eng\\Desktop\\PyTorch_data\\fruits360\\train_test"
        for class_name in classes:
            for idx in range(2500):
                img = Image.open(filepath_fruit360 + "/" + class_name + "/" + class_name + " (" + str(idx+1) + ").jpg").convert("RGB")
                numpy_img = np.array(img)
                resize_numpy_img = cv2.resize(numpy_img, (100, 100), interpolation=cv2.INTER_LINEAR)
                transpose_numpy_img = np.transpose(resize_numpy_img, (2, 0, 1))
                image.append(transpose_numpy_img)
                label.append(class_labels[class_name])
                img.close()

        # Image & label to ndarray
        image_np = np.array(image, dtype=np.uint8)
        label_np = np.array(label, dtype=np.uint8)
        # show
        if (False):
            transpose_data = np.transpose(image_np[10], (1, 2, 0)) # For PLT : row col channel
            plt.imshow(transpose_data)
            plt.show()

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
        # Train Test Dataset : <--


        # Inference Dataset : -->
        # File parse
        filepath_fruit360 = "C:\\Users\\eng\\Desktop\\PyTorch_data\\fruits360\\inference"
        for idx in range(15):
            img = Image.open(filepath_fruit360 + "/" + "infer" + " (" + str(idx+1) + ").jpg").convert("RGB")
            numpy_img = np.array(img)
            resize_numpy_img = cv2.resize(numpy_img, (100, 100), interpolation=cv2.INTER_LINEAR)
            transpose_numpy_img = np.transpose(resize_numpy_img, (2, 0, 1))
            inference_image.append(transpose_numpy_img)
            img.close()

        # Image & label to ndarray
        image_np = np.array(inference_image, dtype=np.uint8)
        # show
        if (False):
            transpose_data = np.transpose(image_np[10], (1, 2, 0)) # For PLT : row col channel
            plt.imshow(transpose_data)
            plt.show()

        # Image normalization (0.0~1.0)
        image_np = np.divide(image_np, 255.0, dtype=np.float32)

        X_infer = image_np
        # Inference Dataset : <--

        return [X_train, y_train], [X_test, y_test], X_infer, classes

    except Exception as e : print("Exception :", e)


if __name__ == "__main__":
    fruits360_parsing()
