import os

import cv2
import numpy as np
import matplotlib as plt

def webcam_parser():
    try:
        print("Webcam Start")

        # Fake Dataset
        if(False):
            X_infer = np.zeros((100,3,224,224), dtype=np.float32)
            return X_infer

        inference_image = []

        # '0' is camera index
        cap = cv2.VideoCapture(0)

        # Video Stream Check
        if not cap.isOpened():
            raise Exception("Opening video stream or file")

        # Loop
        while True:
            # Get Frame
            ret, frame = cap.read()

            # Return code check
            if not ret:
                cap.release()
                raise Exception("Reading the frame")

            # Show
            cv2.imshow("Frame", frame)

            # Frame to NumPy
            numpy_frame = np.array(frame)

            # Transpose : Col Row Channel -> Channel Row Col
            transpose_numpy_frame = np.transpose(numpy_frame, (2, 1, 0))
            inference_image.append(transpose_numpy_frame)

            # 27 is esc
            if cv2.waitKey(1) == 27:
                break

        # Camera Release
        cap.release()

        # Destroy Show Window
        cv2.destroyAllWindows()

        # Frame to ndarray
        image_np = np.array(inference_image, dtype=np.uint8)

        # Show PLT
        if (False):
            transpose_data = np.transpose(image_np[10], (1, 2, 0)) # For PLT : row col channel
            plt.imshow(transpose_data)
            plt.show()

        # Image Normalization (0.0~1.0)
        X_infer = np.divide(image_np, 255.0, dtype=np.float32)

        return X_infer

    except Exception as e : print("Exception :", e)

if __name__ == "__main__":
    webcam_parser()
