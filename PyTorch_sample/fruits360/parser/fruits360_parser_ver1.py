import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

data_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_data/"

filepath_fruits360 = data_path + "fruits360/split_data/"

data_class = ["apple", "banana", "orange", "strawberry"]

def fruits360_parsing_ver1():
    try:
        print("Parsing Start")

        train_image = []
        train_lable = []
        test_image = []
        test_lable = []
        classes = []

        for class_idx in range(len(data_class)):
            for idx in range(2500):
                img = Image.open(filepath_fruits360 + "train/" + data_class[class_idx] + "/" + data_class[class_idx] + " (" + str(idx+1) + ").jpg").convert('RGB')
                numpy_img  = np.array(img)
                numpy_tans = np.transpose(numpy_img, (2, 0, 1))
                train_image.append( numpy_tans ) # For PyTorch : channel row col 
                train_lable.append( class_idx )
                img.close()

        for class_idx in range(4):
            for idx in range(300):
                img = Image.open(filepath_fruits360 + "test/" + data_class[class_idx] + "/" + data_class[class_idx] + " (" + str(idx+1) + ").jpg").convert('RGB')
                numpy_img  = np.array(img)
                numpy_tans = np.transpose(numpy_img, (2, 0, 1))
                test_image.append( numpy_tans ) # For PyTorch : channel row col 
                test_lable.append( class_idx )
                img.close()

        # To NumPy
        train_image = np.array(train_image, dtype=np.uint8)
        train_lable = np.array(train_lable, dtype=np.uint8)
        test_image = np.array(test_image, dtype=np.uint8)
        test_lable = np.array(test_lable, dtype=np.uint8)

        # Image Normalize : 0 ~ 1
        train_image = np.array(train_image/255.0, dtype=np.float32)
        test_image = np.array(test_image/255.0, dtype=np.float32)

        # Shuffle Train
        shuffle_idx = np.arange(4000)
        np.random.shuffle(shuffle_idx)
        train_image = train_image[shuffle_idx]
        train_lable = train_lable[shuffle_idx]

        # Shuffle Test
        shuffle_idx = np.arange(1200)
        np.random.shuffle(shuffle_idx)
        test_image = test_image[shuffle_idx]
        test_lable = test_lable[shuffle_idx]

        print("Parsing Done\n")

        # Result : -->
        # train_image (0.0 ~ 1.0, float32, NumPy array) [ N, 3, row, col ]
        # train_lable (0 ~ 255, uint8, NumPy array) [ N ]
        # test_image (0.0 ~ 1.0, float32, NumPy array) [ N, 3, row, col ]
        # test_lable (0 ~ 255, uint8, NumPy array) [ N ]
        # classes (string, Python list)
        # Result : <--

        return [train_image, train_lable], [test_image, test_lable], classes

    except Exception as e : print("Exception :", e)