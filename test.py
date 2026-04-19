import cv2
from ultralytics import YOLO

# Load models
top_model = YOLO("models/lasttop.pt")
side_model = YOLO("models/lastside.pt")

# Camera streams (replace with your IPs)
cap_top = cv2.VideoCapture("http://192.168.1.11:8080/video")
cap_side = cv2.VideoCapture("http://192.168.1.12:8080/video")


def get_classes(result):
    names = result[0].names
    classes = result[0].boxes.cls.tolist() if result[0].boxes else []
    return [names[int(c)] for c in classes]


def combine_results(top_classes, side_classes):
    defects = []

    if "top_damaged" in top_classes:
        defects.append("Top Defect")

    if "side_damaged" in side_classes:
        defects.append("Side Defect")

    return "DEFECTIVE" if defects else "OK", defects


while True:
    ret1, frame_top = cap_top.read()
    ret2, frame_side = cap_side.read()

    if not ret1 or not ret2:
        break

    # Resize for speed
    frame_top = cv2.resize(frame_top, (640, 640))
    frame_side = cv2.resize(frame_side, (640, 640))

    # Run inference
    top_res = top_model(frame_top, verbose=False)
    side_res = side_model(frame_side, verbose=False)

    # Draw detections
    frame_top = top_res[0].plot()
    frame_side = side_res[0].plot()

    # Extract classes
    top_classes = get_classes(top_res)
    side_classes = get_classes(side_res)

    # Combine
    status, defects = combine_results(top_classes, side_classes)

    # Add status text
    cv2.putText(frame_top, f"TOP: {top_classes}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame_side, f"SIDE: {side_classes}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame_top, f"STATUS: {status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Combine both frames side by side
    combined = cv2.hconcat([frame_top, frame_side])

    cv2.imshow("Can Inspection System", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_top.release()
cap_side.release()
cv2.destroyAllWindows()