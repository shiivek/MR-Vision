import cv2
import numpy as np
import time

def calculate_optical_flow(prev_frame, current_frame):
    """Basic optical flow calculation"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,
        current_frame,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def warp_frame(frame, flow, x_offset):
    """Simple frame warping"""
    h, w = frame.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Basic flow adjustment
    flow_x = flow[:, :, 0].copy() - x_offset
    flow_y = flow[:, :, 1].copy()
    
    map_x = (map_x + flow_x).astype(np.float32)
    map_y = (map_y + flow_y).astype(np.float32)
    
    warped_frame = cv2.remap(frame, map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped_frame

def estimate_overlap(flow, frame1, frame2, overlap_width, prev_x_offset=None):
    """Basic overlap estimation"""
    h, w = flow.shape[:2]
    
    # Use center region for flow estimation
    center_flow = flow[:, w//4:3*w//4, 0]
    median_flow = np.median(center_flow)
    
    # Calculate x_offset
    x_offset = int((w / 2) + median_flow - (overlap_width / 2))
    
    # Simple temporal smoothing
    if prev_x_offset is not None:
        x_offset = int(0.7 * prev_x_offset + 0.3 * x_offset)
    
    # Ensure minimum overlap
    min_overlap = int(w * 0.3)
    x_offset = np.clip(x_offset, -min_overlap, w - min_overlap)
    
    return x_offset, overlap_width, flow

def create_simple_mask(height, width):
    """Creates a simple linear blending mask"""
    return np.linspace(0, 1, width)[np.newaxis, :].repeat(height, axis=0)

def blend_frames(frame1, warped_frame2, x_offset, overlap_width):
    """Simple frame blending"""
    h, w = frame1.shape[:2]
    stitched_width = max(w, x_offset + warped_frame2.shape[1])
    stitched_frame = np.zeros((h, stitched_width, 3), dtype=np.uint8)
    
    # Calculate overlap region
    overlap_start = w - overlap_width
    overlap_end = w
    
    # Create blending mask
    blend_mask = create_simple_mask(h, overlap_width)[:, :, np.newaxis]
    
    # Extract and blend overlap regions
    frame1_overlap = frame1[:, overlap_start:overlap_end]
    warped_overlap = warped_frame2[:, w - overlap_width - x_offset:w - x_offset]
    
    # Ensure same width for blending
    min_width = min(frame1_overlap.shape[1], warped_overlap.shape[1])
    frame1_overlap = frame1_overlap[:, :min_width]
    warped_overlap = warped_overlap[:, :min_width]
    blend_mask = blend_mask[:, :min_width]
    
    blended = (frame1_overlap * (1 - blend_mask) + 
              warped_overlap * blend_mask).astype(np.uint8)
    
    # Construct final frame
    stitched_frame[:, :overlap_start] = frame1[:, :overlap_start]
    stitched_frame[:, overlap_start:overlap_start + min_width] = blended
    
    # Add remaining part of warped frame
    remaining_start = overlap_start + min_width
    if remaining_start < stitched_width:
        remaining = warped_frame2[:, w - x_offset:]
        remaining_width = min(remaining.shape[1], stitched_width - remaining_start)
        stitched_frame[:, remaining_start:remaining_start + remaining_width] = \
            remaining[:, :remaining_width]
    
    return stitched_frame

def main():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open cameras")
        return
    
    # Set resolution
    frame_width = 640
    frame_height = 480
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    # Initialize
    ret1, frame1 = cap1.read()
    ret2, prev_frame2 = cap2.read()
    prev_gray2 = cv2.cvtColor(prev_frame2, cv2.COLOR_BGR2GRAY)
    
    overlap_width = int(frame_width * 0.3)
    prev_x_offset = None
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    fps_update_interval = 0.5  # Update FPS every 0.5 seconds
    
    while True:
        frame_start_time = time.time()
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Process frames
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = calculate_optical_flow(prev_gray2, gray2)
        
        # Estimate overlap and warp
        x_offset, overlap_width, _ = estimate_overlap(flow, frame1, frame2, overlap_width, prev_x_offset)
        warped_frame2 = warp_frame(frame2, flow, x_offset)
        
        # Blend frames
        stitched_frame = blend_frames(frame1, warped_frame2, x_offset, overlap_width)
        
        # Update previous values
        prev_gray2 = gray2.copy()
        prev_x_offset = x_offset
        
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= fps_update_interval:
            fps = int(frame_count / elapsed_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw FPS and resolution on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_color = (0, 255, 0)  # Green color
        cv2.putText(stitched_frame, f"FPS: {fps}", (10, 30), font, 0.7, fps_color, 2, cv2.LINE_AA)
        cv2.putText(stitched_frame, f"Resolution: {stitched_frame.shape[1]}x{stitched_frame.shape[0]}", 
                    (10, 60), font, 0.7, fps_color, 2, cv2.LINE_AA)
        
        # Display result
        cv2.imshow("Stitched Video", stitched_frame)
        
        # Frame rate control
        frame_time = time.time() - frame_start_time
        wait_time = max(1, int(1000/30 - frame_time*1000))  # Target 30 FPS
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()