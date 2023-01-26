import os

import matplotlib.pyplot as plt
import numpy as np

def cifar10_parsing():
    try:
        print("Parsing Start")

        print("Parsing Done\n")

        # Result : -->
        # train_image(0.0 ~ 1.0, float32, NumPy array) [ N, 3, row, col ]
        # train_lable(0 ~ 255, uint8, NumPy array) [ N ]
        # test_image(0.0 ~ 1.0, float32, NumPy array) [ N, 3, row, col ]
        # test_lable(0 ~ 255, uint8, NumPy array) [ N ]
        # classes(string, Python list)
        return [train_image, train_lable], [test_image, test_lable], classes
        # Result : <--

    except Exception as e : print("Exception :", e)