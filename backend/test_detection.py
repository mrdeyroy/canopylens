"""
backend/test_detection.py
Proof-of-concept tree-crown detection script using DeepForest.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run individual tree crown detection on a forest image using DeepForest."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="Path to the input high-resolution forest image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Validate image path argument
    if not args.image_path:
        print("[ERROR] Missing required image_path argument.", file=sys.stderr)
        print("Usage: python backend/test_detection.py <path/to/image.jpg>", file=sys.stderr)
        sys.exit(1)

    image_path = Path(args.image_path)

    # 2. Validate image file exists
    if not image_path.exists():
        print(f"[ERROR] Image file not found: '{image_path}'", file=sys.stderr)
        sys.exit(1)

    if not image_path.is_file():
        print(f"[ERROR] Specified path is not a file: '{image_path}'", file=sys.stderr)
        sys.exit(1)

    # 3. Validate image can be read
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERROR] Failed to read image using OpenCV: '{image_path}'. Supported formats include JPG, PNG, TIFF.", file=sys.stderr)
        sys.exit(1)

    h, w, c = image_bgr.shape
    print(f"[INFO] Loaded image '{image_path}' successfully (Dimensions: {w}x{h}, Channels: {c}).")

    # 4. Load DeepForest model
    print("[INFO] Loading DeepForest model...")
    try:
        from deepforest import main as df_main
        model = df_main.deepforest()
        if hasattr(model, "use_release"):
            model.use_release()
        print("[INFO] DeepForest model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load DeepForest model: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Run inference
    print("[INFO] Running tree-crown detection...")
    try:
        predictions = model.predict_image(path=str(image_path))
    except Exception as e:
        print(f"[ERROR] Inference failed during model.predict_image: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure predictions is a DataFrame
    if predictions is None:
        predictions = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    elif not isinstance(predictions, pd.DataFrame):
        try:
            predictions = pd.DataFrame(predictions)
        except Exception as e:
            print(f"[ERROR] Unable to parse prediction output into DataFrame: {e}", file=sys.stderr)
            sys.exit(1)

    num_detections = len(predictions)
    print(f"[RESULT] Number of detections: {num_detections}")
    print(f"[RESULT] Prediction columns: {list(predictions.columns)}")

    # 6. Save raw predictions to data/predictions.csv
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = output_dir / "predictions.csv"

    try:
        predictions.to_csv(predictions_csv_path, index=False)
        print(f"[SUCCESS] Saved raw predictions to '{predictions_csv_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save predictions to '{predictions_csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 7. Create annotated output image
    annotated_image = image_bgr.copy()
    box_color = (0, 255, 0)  # Green for bounding boxes
    text_color = (255, 255, 255)  # White for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    for _, row in predictions.iterrows():
        try:
            xmin = int(round(float(row["xmin"])))
            ymin = int(round(float(row["ymin"])))
            xmax = int(round(float(row["xmax"])))
            ymax = int(round(float(row["ymax"])))
            score = float(row["score"]) if "score" in row else None

            # Draw rectangle
            cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), box_color, thickness)

            # Optionally draw score label
            label = f"{score:.2f}" if score is not None else "tree"
            label_size, _ = cv2.getTextSize(label, font, font_scale, 1)
            label_ymin = max(ymin, label_size[1] + 4)
            cv2.rectangle(
                annotated_image,
                (xmin, label_ymin - label_size[1] - 4),
                (xmin + label_size[0] + 4, label_ymin),
                box_color,
                -1,
            )
            cv2.putText(
                annotated_image,
                label,
                (xmin + 2, label_ymin - 2),
                font,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        except Exception as e:
            # Continue drawing remaining boxes if one box has malformed coordinates
            continue

    detection_result_path = output_dir / "detection_result.jpg"
    try:
        success = cv2.imwrite(str(detection_result_path), annotated_image)
        if not success:
            raise IOError("cv2.imwrite returned False.")
        print(f"[SUCCESS] Saved annotated detection result to '{detection_result_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save annotated image to '{detection_result_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print("[DONE] Tree detection proof of concept completed successfully.")


if __name__ == "__main__":
    main()
