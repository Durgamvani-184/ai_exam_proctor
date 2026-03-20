import cv2
import numpy as np

def check_cameras():
    print("Checking camera indices 0 to 5...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i}: OPENED and CAPTURING (mean: {np.mean(frame):.2f})")
            else:
                print(f"Camera index {i}: OPENED but READ FAILED")
            cap.release()
        else:
            print(f"Camera index {i}: NOT OPENED")

if __name__ == "__main__":
    check_cameras()
