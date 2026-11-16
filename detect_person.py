"""Detect whether a person is present in an image using YOLO.

Usage:
    uv run detect_person.py path/to/image.jpg [--conf 0.25] [--output out.jpg] [--show]

Exit codes: 0 = person found, 1 = no person found, 2 = error.
"""
import argparse
import sys

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class 0 is "person"
DEFAULT_MODEL = "yolo11n.pt"  # downloaded automatically on first run

_model = None


def get_model(name=DEFAULT_MODEL):
    global _model
    if _model is None:
        _model = YOLO(name)
    return _model


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_people(img, conf=0.25, model_name=DEFAULT_MODEL):
    """Return a list of (x1, y1, x2, y2, confidence) for people found in `img`."""
    results = get_model(model_name)(
        img, conf=conf, classes=[PERSON_CLASS_ID], verbose=False
    )
    boxes = results[0].boxes
    return [
        (*map(int, xyxy), float(c))
        for xyxy, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist())
    ]


def draw_boxes(img, detections):
    out = img.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Is there a person in the image?")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold (default: 0.25)")
    parser.add_argument("--output", help="save the annotated image to this path")
    parser.add_argument("--show", action="store_true",
                        help="display image with boxes (needs a display)")
    args = parser.parse_args()

    try:
        img = load_image(args.image)
        detections = detect_people(img, conf=args.conf)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if detections:
        print(f"Person detected ({len(detections)} found)")
    else:
        print("No person detected")

    if args.output or args.show:
        annotated = draw_boxes(img, detections)
        if args.output:
            if not cv2.imwrite(args.output, annotated):
                print(f"Error: could not write {args.output}", file=sys.stderr)
                return 2
            print(f"Saved annotated image to {args.output}")
        if args.show:
            cv2.imshow("result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0 if detections else 1


if __name__ == "__main__":
    sys.exit(main())
