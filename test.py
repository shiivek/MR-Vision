import cv2
import time
import numpy as np

def stack_side_by_side(frame1, frame2):
    """
    Stacks two frames side by side without resizing them, preserving
    their original resolutions. The shorter image is placed at the top
    and black padding is added to match the taller frame's height.
    """
    h1, w1, c1 = frame1.shape
    h2, w2, c2 = frame2.shape

    # Determine the combined frame size
    max_height = max(h1, h2)
    total_width = w1 + w2

    # Create a black canvas of the combined size
    combined_frame = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Place frame1 in the combined frame (left)
    combined_frame[:h1, :w1, :] = frame1

    # Place frame2 in the combined frame (right)
    combined_frame[:h2, w1:w1 + w2, :] = frame2

    return combined_frame

def main():
    # Open two cameras (adjust indices as needed)
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    if not cap1.isOpened():
        print("Could not open camera 1.")
        return
    if not cap2.isOpened():
        print("Could not open camera 2.")
        return

    # Retrieve each camera’s width and height (native resolution)
    width1  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width2  = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize for FPS calculation
    start_time = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to capture frame from one or both cameras.")
            break

        # Count frames for FPS calculation
        frames += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Update FPS every second
        if elapsed_time >= 1.0:
            fps = frames / elapsed_time
            frames = 0
            start_time = current_time

        # Overlay resolution and FPS on frame1
        cv2.putText(frame1,
                    f"Cam1 {width1}x{height1} | FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Overlay resolution and FPS on frame2
        cv2.putText(frame2,
                    f"Cam2 {width2}x{height2} | FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Stack frames side by side without resizing
        combined_frame = stack_side_by_side(frame1, frame2)

        # Show combined frame
        cv2.imshow("Two Cameras Side by Side", combined_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
