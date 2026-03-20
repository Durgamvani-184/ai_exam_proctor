import cv2
import numpy as np
import time

def force_camera_settings():
    print("Attempting to capture frame with CAP_DSHOW and forced settings...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Try to set brightness and exposure
        # These values vary by camera, but we can try common ranges
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 150) # 0-255 usually
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)    # -1 to -7 is common on Windows
        
        print("Settings applied. Capturing 20 frames...")
        for i in range(20):
            ret, frame = cap.read()
            if i % 10 == 0:
                if ret:
                    print(f"Frame {i}: Mean {np.mean(frame):.2f}")
                else:
                    print(f"Frame {i}: Read failed")
            time.sleep(0.05)
            
        if ret:
            print(f"Final Frame: Mean {np.mean(frame):.2f}")
            cv2.imwrite("test_frame_forced.jpg", frame)
            print("Saved to test_frame_forced.jpg")
        else:
            print("Failed to capture frame.")
        cap.release()
    else:
        print("Failed to open camera 0 with CAP_DSHOW.")

if __name__ == "__main__":
    force_camera_settings()
