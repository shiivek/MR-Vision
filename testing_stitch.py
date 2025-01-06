import cv2
import numpy as np
import time

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
INTER_OCULAR_DISTANCE = 6.5  # cm
OVERLAP_RATIO = 0.3  # 30% overlap

def calculate_optical_flow(prev_frame, current_frame):
    return cv2.calcOpticalFlowFarneback(
        prev_frame, current_frame,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

def warp_frame(frame, flow, x_offset):
    h, w = frame.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Smooth flow fields for better stability
    flow_x = cv2.bilateralFilter(flow[:, :, 0], 9, 75, 75) - x_offset
    flow_y = cv2.bilateralFilter(flow[:, :, 1], 9, 75, 75)
    
    map_x = (map_x + flow_x).astype(np.float32)
    map_y = (map_y + flow_y).astype(np.float32)
    
    warped_frame = cv2.remap(frame, map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped_frame

def blend_frames(left_frame, right_frame, overlap_width):
    h, w = left_frame.shape[:2]
    
    # Calculate overlap region
    overlap_start = w - overlap_width
    
    # Create weight mask for blending
    x = np.linspace(0, 1, overlap_width)
    mask = x[np.newaxis, :, np.newaxis].repeat(h, axis=0)
    
    # Extract and blend overlap regions
    left_overlap = left_frame[:, overlap_start:w]
    right_overlap = right_frame[:, :overlap_width]
    blended_overlap = (left_overlap * (1 - mask) + right_overlap * mask).astype(np.uint8)
    
    # Construct final frame
    final_width = w * 2 - overlap_width
    stitched = np.zeros((h, final_width, 3), dtype=np.uint8)
    stitched[:, :overlap_start] = left_frame[:, :overlap_start]
    stitched[:, overlap_start:overlap_start + overlap_width] = blended_overlap
    stitched[:, overlap_start + overlap_width:] = right_frame[:, overlap_width:]
    
    return stitched

def main():
    # Initialize cameras
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
    
    # Calculate overlap width based on frame width
    overlap_width = int(FRAME_WIDTH * OVERLAP_RATIO)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    fps_update_interval = 0.5
    
    # Initialize previous frame for optical flow
    _, prev_right = right_cam.read()
    prev_right_gray = cv2.cvtColor(prev_right, cv2.COLOR_BGR2GRAY)
    
    while True:
        frame_start_time = time.time()
        
        # Capture frames
        ret_left, left_frame = left_cam.read()
        ret_right, right_frame = right_cam.read()
        
        if not ret_left or not ret_right:
            break
        
        # Convert right frame to grayscale for optical flow
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = calculate_optical_flow(prev_right_gray, right_gray)
        
        # Warp right frame
        warped_right = warp_frame(right_frame, flow, overlap_width)
        
        # Blend frames
        stitched_frame = blend_frames(left_frame, warped_right, overlap_width)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= fps_update_interval:
            fps = int(frame_count / elapsed_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(stitched_frame, f"FPS: {fps}", (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(stitched_frame, f"Resolution: {stitched_frame.shape[1]}x{stitched_frame.shape[0]}", 
                    (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display result
        cv2.imshow("Binocular Vision", stitched_frame)
        
        # Update previous frame
        prev_right_gray = right_gray.copy()
        
        # Frame rate control
        frame_time = time.time() - frame_start_time
        wait_time = max(1, int(1000/TARGET_FPS - frame_time*1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    
    # Cleanup
    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()