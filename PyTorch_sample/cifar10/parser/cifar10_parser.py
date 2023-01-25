import os

import matplotlib.pyplot as plt
import numpy as np

data_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_data/cifar10/"

def cifar10_parsing():
    try:
        print("Parsing Start")

        # Only for cifar10 Hard-Coding : -->
        train_count = 5
        test_count = 1

        lable_size = 1
        image_size = (3 * 32 * 32)
        image_count = 1000
        class_count = 10

        unit_size = lable_size + image_size
        # Only for cifar10 Hard-Coding : <--
        
        train_lable = []
        train_image = []
        test_lable = []
        test_image = []

        # Dataset : train
        for idx in range(train_count):
            filepath_cifar10 = data_path + "cifar-10-batches-bin/data_batch_" + str(idx+1) +".bin"
            binary_cifar10 = open(filepath_cifar10,'rb')
            numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
            for sub_idx in range(image_count * class_count):

                # [ index (0000) ~ index (lable_size(3072byte) + image_size(1byte)) ]
                # [ index (0000) ~ index (3072) ] = [ 3073 byte ]
                # [ index (3073) ~ index (6145) ] = [ 3073 byte ]
                # ...
                unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]

                # lable
                train_lable.append( unit[0] )

                # image
                train_image.append( unit[1:].reshape(3, 32, 32) ) # For PyTorch : channel row col 

                # show
                if (False):
                    transpose_data = np.transpose(train_image[sub_idx], (1, 2, 0)) # For PLT : row col channel
                    plt.imshow(transpose_data)
                    plt.show()

            binary_cifar10.close()

        # Dataset : test
        for idx in range(test_count):
            filepath_cifar10 = data_path + "cifar-10-batches-bin/test_batch.bin"
            binary_cifar10 = open(filepath_cifar10,'rb')
            numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
            for sub_idx in range(image_count * class_count):

                unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]

                test_lable.append(unit[0])
                
                test_image.append(unit[1:].reshape(3,32,32))

            binary_cifar10.close()

        # To NumPy
        train_image = np.array(train_image, dtype=np.uint8)
        train_lable = np.array(train_lable, dtype=np.uint8)
        test_image = np.array(test_image, dtype=np.uint8)
        test_lable = np.array(test_lable, dtype=np.uint8)

        # Image Normalize : 0 ~ 1
        train_image = np.array(train_image/255.0, dtype=np.float32)
        test_image = np.array(test_image/255.0, dtype=np.float32)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print("Parsing Done\n")
        return [train_image, train_lable], [test_image, test_lable], classes

    except Exception as e : print("Exception :", e)