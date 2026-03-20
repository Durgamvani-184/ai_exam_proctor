import cv2
import numpy as np

def scan_all_cameras():
    print("Scanning camera indices 0 to 10 with CAP_DSHOW...")
    found = False
    for i in range(11):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: OPENED and CAPTURING (mean: {np.mean(frame):.2f})")
                if np.mean(frame) > 5:
                    print(f"--- SUCCESS: Index {i} has valid image data! ---")
                    found = True
            else:
                print(f"Index {i}: OPENED but READ FAILED")
            cap.release()
        else:
            # print(f"Index {i}: NOT OPENED")
            pass
    
    if not found:
        print("No active cameras with non-black frames found.")

if __name__ == "__main__":
    scan_all_cameras()
