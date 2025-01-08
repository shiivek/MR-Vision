import cv2
import numpy as np
import time

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
OVERLAP_RATIO = 0.15
OUTPUT_WIDTH = 1024

# Initialize YOLO
def init_yolo():
    # Use YOLOv3 instead of v4
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_people(frame, net, output_layers):
    height, width = frame.shape[:2]
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Detect objects
    outputs = net.forward(output_layers)
    
    # Initialize lists
    boxes = []
    confidences = []
    centers = []
    
    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Increase confidence threshold to 0.7 (from 0.5)
            if class_id == 0 and confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                centers.append((center_x, center_y))
    
    # Apply stricter Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)
    
    # Initialize filtered lists
    filtered_centers = []
    filtered_confidences = []
    
    # Draw detections
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x, y, w, h = boxes[i]
            center_x, center_y = centers[i]
            confidence = confidences[i]
            
            # Add to filtered lists
            filtered_centers.append((center_x, center_y))
            filtered_confidences.append(confidence)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Add label with coordinates and confidence
            label = f"Person {len(filtered_centers)-1}: ({center_x},{center_y}) {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y-20), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return frame, filtered_centers, filtered_confidences

def blend_frames(left_frame, right_frame, net, output_layers):
    h, w = left_frame.shape[:2]
    stitched_width = int(w * 2 - (w * OVERLAP_RATIO))
    stitched = np.zeros((h, stitched_width, 3), dtype=np.uint8)
    
    # First do the blending without detection
    overlap_width = int(w * OVERLAP_RATIO)
    right_start = w - overlap_width
    
    # Create blending mask
    mask = np.linspace(0, 1, overlap_width)
    mask = mask.reshape(1, -1, 1)
    mask = np.tile(mask, (h, 1, 3))
    
    # Get overlap regions
    left_overlap = left_frame[:, -overlap_width:]
    right_overlap = right_frame[:, :overlap_width]
    
    # Blend overlap region
    blended = (left_overlap * (1 - mask) + right_overlap * mask).astype(np.uint8)
    
    # Copy frames and blended region
    stitched[:, :w] = left_frame
    stitched[:, right_start:right_start+overlap_width] = blended
    stitched[:, right_start+overlap_width:] = right_frame[:, overlap_width:]
    
    # Resize to desired output width
    aspect_ratio = stitched_width / h
    target_height = int(OUTPUT_WIDTH / aspect_ratio)
    stitched = cv2.resize(stitched, (OUTPUT_WIDTH, target_height))
    
    # Now apply detection on the stitched image
    stitched, centers, confidences = detect_people(stitched, net, output_layers)
    
    # Add detected coordinates
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 90  # Start below resolution text
    
    for i, (center, conf) in enumerate(zip(centers, confidences)):
        text = f"Person {i}: ({center[0]},{center[1]}) {conf:.2f}"
        cv2.putText(stitched, text, (10, y_offset + i*20), font, 0.5, (0, 255, 0), 1)
    
    return stitched

def main():
    # Initialize YOLO
    net, output_layers = init_yolo()
    
    left_cam = cv2.VideoCapture(1)
    right_cam = cv2.VideoCapture(0)
    
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
        
        # Stitch frames with person detection
        stitched_frame = blend_frames(left_frame, right_frame, net, output_layers)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 0.5:
            fps = int(frame_count / elapsed_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(stitched_frame, f"FPS: {fps}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(stitched_frame, f"Resolution: {stitched_frame.shape[1]}x{stitched_frame.shape[0]}", 
                    (10, 60), font, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Binocular Vision", stitched_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()