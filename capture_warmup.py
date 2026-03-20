import cv2
import numpy as np
import time

def capture_extended_warmup():
    print("Attempting to capture frame with CAP_DSHOW and 50-frame warmup...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to nudge auto-exposure
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Usually 0.25 (auto) or 0.75 (auto)
        
        print("Warming up for 50 frames...")
        for i in range(50):
            ret, frame = cap.read()
            if i % 10 == 0:
                if ret:
                    print(f"Frame {i}: Mean {np.mean(frame):.2f}")
                else:
                    print(f"Frame {i}: Read failed")
            time.sleep(0.05)
            
        if ret:
            print(f"Final Frame: Mean {np.mean(frame):.2f}")
            cv2.imwrite("test_frame_warmup.jpg", frame)
            print("Saved to test_frame_warmup.jpg")
        else:
            print("Failed to capture frame.")
        cap.release()
    else:
        print("Failed to open camera 0 with CAP_DSHOW.")

if __name__ == "__main__":
    capture_extended_warmup()
