import cv2
import numpy as np
import time

# Constants (keep original constants)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
OVERLAP_RATIO = 0.25
OUTPUT_WIDTH = 1024

def detect_objects(frame):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect objects
    boxes, weights = hog.detectMultiScale(frame, 
                                        winStride=(8, 8),
                                        padding=(8, 8),
                                        scale=1.05)
    
    # Draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

def blend_frames(left_frame, right_frame):
    # Keep original blending code
    h, w = left_frame.shape[:2]
    stitched_width = int(w * 1.75)
    stitched = np.zeros((h, stitched_width, 3), dtype=np.uint8)
    
    # Apply object detection to individual frames before blending
    left_frame = detect_objects(left_frame)
    right_frame = detect_objects(right_frame)
    
    # Copy left frame
    stitched[:, :w] = left_frame
    
    # Calculate overlap region
    overlap_width = int(w * OVERLAP_RATIO)
    right_start = w - overlap_width
    
    # Create blending mask
    mask = np.linspace(0, 1, overlap_width)
    mask = mask.reshape(1, -1, 1)
    mask = np.tile(mask, (h, 1, 3))
    
    # Blend overlap region
    left_overlap = left_frame[:, -overlap_width:]
    right_overlap = right_frame[:, :overlap_width]
    blended = (left_overlap * (1 - mask) + right_overlap * mask).astype(np.uint8)
    
    # Copy right frame and blended region
    stitched[:, right_start:right_start+overlap_width] = blended
    stitched[:, right_start+overlap_width:right_start+w] = right_frame[:, overlap_width:]
    
    # Resize to desired output width while maintaining aspect ratio
    aspect_ratio = stitched_width / h
    target_height = int(OUTPUT_WIDTH / aspect_ratio)
    stitched = cv2.resize(stitched, (OUTPUT_WIDTH, target_height))
    
    return stitched

# Main function remains the same
def main():
    left_cam = cv2.VideoCapture(0)
    right_cam = cv2.VideoCapture(1)
    
    if not left_cam.isOpened() or not right_cam.isOpened():
        print("Error: Could not open cameras")
        return
    
    # Set camera properties
    for cam in [left_cam, right_cam]:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    prev_stitched = None
    
    while True:
        frame_start_time = time.time()
        
        ret_left, left_frame = left_cam.read()
        ret_right, right_frame = right_cam.read()
        
        if not ret_left or not ret_right:
            break
        
        # Ensure frames are the right size
        left_frame = cv2.resize(left_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        right_frame = cv2.resize(right_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Apply minimal smoothing
        left_frame = cv2.GaussianBlur(left_frame, (3, 3), 0)
        right_frame = cv2.GaussianBlur(right_frame, (3, 3), 0)
        
        # Stitch frames (now includes object detection)
        stitched_frame = blend_frames(left_frame, right_frame)
        
        # Temporal smoothing
        if prev_stitched is not None:
            stitched_frame = cv2.addWeighted(prev_stitched, 0.3, stitched_frame, 0.7, 0)
        prev_stitched = stitched_frame.copy()
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 0.5:
            fps = int(frame_count / elapsed_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(stitched_frame, f"FPS: {fps}", (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(stitched_frame, f"Resolution: {stitched_frame.shape[1]}x{stitched_frame.shape[0]}", 
                    (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display
        cv2.imshow("Binocular Vision", stitched_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()