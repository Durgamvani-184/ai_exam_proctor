import cv2
import numpy as np
import time

def capture_test_mjpg():
    print("Attempting to capture frame with CAP_DSHOW and MJPG format...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Set MJPG format
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Warming up for 20 frames...")
        for i in range(20):
            ret, frame = cap.read()
            if i % 5 == 0:
                if ret:
                    print(f"Frame {i}: Mean {np.mean(frame):.2f}")
                else:
                    print(f"Frame {i}: Read failed")
            time.sleep(0.05)
            
        if ret:
            print(f"Final Frame: Mean {np.mean(frame):.2f}")
            cv2.imwrite("test_frame_mjpg.jpg", frame)
            print("Saved to test_frame_mjpg.jpg")
        else:
            print("Failed to capture frame.")
        cap.release()
    else:
        print("Failed to open camera 0 with CAP_DSHOW.")

if __name__ == "__main__":
    capture_test_mjpg()
