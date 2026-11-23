import cv2
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "label": label,
                    "class_id": class_id
                })

        return detections

    def draw_detections(self, frame, detections):

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]
            label = detection["label"]
            confidence = detection["confidence"]

            text = f"{label} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame