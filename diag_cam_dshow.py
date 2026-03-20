import cv2
import numpy as np

def check_cameras_dshow():
    print("Checking camera indices 0 to 5 with CAP_DSHOW...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} (DSHOW): OPENED and CAPTURING (mean: {np.mean(frame):.2f})")
            else:
                print(f"Camera index {i} (DSHOW): OPENED but READ FAILED")
            cap.release()
        else:
            print(f"Camera index {i} (DSHOW): NOT OPENED")

if __name__ == "__main__":
    check_cameras_dshow()
