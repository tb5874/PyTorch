import os
import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

# Setting : -->
train_test_path         = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_data/fruits360/train_test/"
inference_path          = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_data/fruits360/inference/"

#numpy_cache_path        = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_cache/numpy/size_100by100/"
numpy_cache_path        = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_cache/numpy/size_224by224/"

each_class_count        = 2500
total_inference_count   = 15

raw_resize              = (224, 224)
# Setting : <--

def fruits360_parsing(train_flag):
    try:
        print("Parsing Start")
        image = []
        label = []
        inference_image = []

        # Define a list of classes and their corresponding labels
        # train_test/<folder name>
        classes = ['apple', 'banana', 'orange', 'strawberry']

        # Train Test Dataset : -->
        if (train_flag):
            # File parse
            filepath_fruit360 = train_test_path
            for class_name in classes:
                print("Train :", class_name)
                for idx in range(each_class_count):
                    img = Image.open( filepath_fruit360 + class_name + "/" + class_name + " (" + str(idx+1) + ").jpg" ).convert("RGB")
                    numpy_img = np.array(img)
                    resize_numpy_img = cv2.resize(numpy_img, raw_resize, interpolation=cv2.INTER_LINEAR)
                    transpose_numpy_img = np.transpose(resize_numpy_img, (2, 0, 1))
                    image.append(transpose_numpy_img)
                    label.append( classes.index(class_name) )
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

            # NumPy Save
            np.save(numpy_cache_path + "X_train.npy", X_train)
            np.save(numpy_cache_path + "y_train.npy", y_train)
            np.save(numpy_cache_path + "X_test.npy", X_test)
            np.save(numpy_cache_path + "y_test.npy", y_test)
        else:
            # NumPy Load
            if os.path.exists(numpy_cache_path + "X_train.npy") and\
                os.path.exists(numpy_cache_path + "y_train.npy") and\
                os.path.exists(numpy_cache_path + "X_test.npy") and\
                os.path.exists(numpy_cache_path + "y_test.npy"):
                X_train = np.load(numpy_cache_path + "X_train.npy")
                y_train = np.load(numpy_cache_path + "y_train.npy")
                X_test = np.load(numpy_cache_path + "X_test.npy")
                y_test = np.load(numpy_cache_path + "y_test.npy")
            else:
                raise Exception("Not exist file\n")
        # Train Test Dataset : <--


        # Inference Dataset : -->
        # File parse
        filepath_fruit360 = inference_path
        for idx in range(total_inference_count):
            img = Image.open( filepath_fruit360 + "infer" + " (" + str(idx+1) + ").jpg" ).convert("RGB")
            numpy_img = np.array(img)
            resize_numpy_img = cv2.resize(numpy_img, raw_resize, interpolation=cv2.INTER_LINEAR)
            transpose_numpy_img = np.transpose(resize_numpy_img, (2, 0, 1))
            inference_image.append(transpose_numpy_img)
            img.close()

        # Image to ndarray
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

