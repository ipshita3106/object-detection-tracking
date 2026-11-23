import cv2


class ObjectTracker:

    def __init__(self, model):
        self.model = model

    def track_objects(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        tracked_objects = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            if boxes.id is None:
                continue

            tracking_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for track_id, class_id, box, confidence in zip(
                tracking_ids,
                class_ids,
                coordinates,
                confidences
            ):

                label = self.model.names[class_id]

                x1, y1, x2, y2 = box

                tracked_objects.append({
                    "id": track_id,
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

                text = f"{label} ID:{track_id}"

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

        return frame, tracked_objects