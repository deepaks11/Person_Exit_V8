import cv2
import supervision as sv
import os
from polygon_test import PolygonTest

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class YoloObjectDetection:

    def __init__(self, q_img, model, line_zones, zones):

        self.frame = None
        self.person_flag = None
        self.intersect = None
        self.q_img = q_img.get()
        self.model = model
        self.line_zones = line_zones
        self.zones = zones
        self.detections = None
        self.count = None

    def predict(self):
        try:

            box_annotator = sv.BoxAnnotator(
                thickness=2,
            )
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=0)

            for result in self.model(source=self.q_img, classes=0, verbose=False):

                self.frame = result.orig_img
                self.detections = sv.Detections.from_ultralytics(result)

                labels = [

                    f"{self.model.names[class_id]}"
                    for box, mask, confidence, class_id, tracker_id, class_name
                    in self.detections
                ]
                box_annotator.annotate(
                    scene=self.frame,
                    detections=self.detections,

                )

                label_annotator.annotate(scene=self.frame, detections=self.detections, labels=labels)
                # return frame, detections
                self.polygon_test()
                return self.frame

        except Exception as er:
            print(er)

    def polygon_test(self):
        try:
            self.intersect, self.count = PolygonTest(self.detections, self.line_zones).point_polygon_test()

            if self.intersect:
                self.plots()
                self.person_flag = False
            else:
                self.plots()
                self.person_flag = True

        except Exception as er:
            print(er)

    def plots(self):
        try:
            if not self.detections:

                cv2.putText(
                    img=self.frame,
                    text=f"Person Not Present the Area",  # Shortened text
                    org=(400, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Changed font style
                    fontScale=1,  # Adjust font size
                    color=(0, 0, 255),
                    thickness=2  # Adjust thickness
                )
                cv2.polylines(self.frame, [self.zones], True, (0, 0, 255), 4)

            elif self.intersect:

                cv2.polylines(self.frame, [self.zones], True, (0, 220, 0), 4)

            else:

                cv2.putText(
                    img=self.frame,
                    text=f"Person Not Present In The Area",  # Shortened text
                    org=(400, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Changed font style
                    fontScale=1,  # Adjust font size
                    color=(0, 0, 255),
                    thickness=2  # Adjust thickness
                )
                cv2.polylines(self.frame, [self.zones], True, (0, 0, 255), 4)

        except Exception as er:
            print(er)
