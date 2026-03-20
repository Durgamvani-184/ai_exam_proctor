import cv2
import numpy as np

def capture_test_frame():
    print("Attempting to capture frame with CAP_DSHOW...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Read a few frames to let camera warm up/auto-expose
        for _ in range(10):
            ret, frame = cap.read()
            
        if ret:
            print(f"Captured frame! Mean brightness: {np.mean(frame):.2f}")
            cv2.imwrite("test_frame.jpg", frame)
            print("Saved to test_frame.jpg")
        else:
            print("Failed to capture frame.")
        cap.release()
    else:
        print("Failed to open camera 0 with CAP_DSHOW.")

if __name__ == "__main__":
    capture_test_frame()
