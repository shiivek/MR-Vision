import cv2

cap = cv2.VideoCapture(2)  # Replace 0 with the camera index you want to test
if not cap.isOpened():
    print("Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
