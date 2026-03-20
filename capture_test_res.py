import cv2
import numpy as np

def capture_test_resolution():
    print("Attempting to capture frame with CAP_DSHOW and 640x480 resolution...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Read a few frames to let camera warm up
        for _ in range(15):
            ret, frame = cap.read()
            
        if ret:
            print(f"Captured frame! Shape: {frame.shape}, Mean brightness: {np.mean(frame):.2f}")
            cv2.imwrite("test_frame_res.jpg", frame)
            print("Saved to test_frame_res.jpg")
        else:
            print("Failed to capture frame.")
        cap.release()
    else:
        print("Failed to open camera 0 with CAP_DSHOW.")

if __name__ == "__main__":
    capture_test_resolution()
