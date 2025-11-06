import cv2
import numpy as np

# ------------------------
# Webcam Setup
# ------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# ------------------------
# Ripeness Color Ranges in HSV
# ------------------------
# Adjust these ranges based on your lighting and camera
ripened_lower = np.array([0, 150, 100])      # Dark Red
ripened_upper = np.array([10, 255, 255])

turning_lower = np.array([10, 100, 100])     # Orange / Yellow
turning_upper = np.array([25, 255, 255])

unripened_lower = np.array([35, 50, 50])     # Green
unripened_upper = np.array([85, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each ripeness
    mask_ripened = cv2.inRange(hsv, ripened_lower, ripened_upper)
    mask_turning = cv2.inRange(hsv, turning_lower, turning_upper)
    mask_unripened = cv2.inRange(hsv, unripened_lower, unripened_upper)

    # Count non-zero pixels in each mask
    ripened_count = cv2.countNonZero(mask_ripened)
    turning_count = cv2.countNonZero(mask_turning)
    unripened_count = cv2.countNonZero(mask_unripened)

    # Determine tomato status
    if ripened_count > 5000:
        status = "Ripened"
    elif turning_count > 5000:
        status = "Turning"
    elif unripened_count > 5000:
        status = "Unripened"
    else:
        status = "Tomato not detected"

    # Display status on frame
    cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Tomato Ripeness Detection", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
