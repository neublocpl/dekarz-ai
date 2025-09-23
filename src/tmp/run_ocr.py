import argparse
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from pathlib import Path


def create_elegant_visualization(image, results):
    """
    Draws elegant OCR results on the image.
    Uses a semi-transparent background for text for better readability.
    """
    output_image = image.copy()
    overlay = output_image.copy()

    if not results:
        return output_image

    for res in results:
        # This function now expects pre-cleaned data, but a safety check is good practice.
        try:
            box = np.array(res[0], dtype=np.int32)
            text, confidence = res[1]
        except (ValueError, TypeError) as e:
            print(
                f"⚠️ Warning: Skipping a result with invalid box format during visualization. Error: {e}. Data: {res}")
            continue

        cv2.polylines(output_image, [box], isClosed=True, color=(0, 255, 120), thickness=2)
        top_left = box.min(axis=0)
        box_width = box[:, 0].max() - top_left[0]
        font_scale = max(0.5, box_width / 150)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_pos = (top_left[0], max(0, top_left[1] - text_height - baseline))

        cv2.rectangle(
            overlay,
            (label_pos[0], label_pos[1] + baseline),
            (label_pos[0] + text_width, label_pos[1] - text_height - int(0.5 * baseline)),
            (30, 30, 30),
            -1
        )
        cv2.putText(
            output_image, text, (label_pos[0], label_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 240), 1, cv2.LINE_AA
        )

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
    return output_image


def process_image_in_tiles(ocr_engine, image, tile_size=1024, overlap=100):
    """
    Splits the image into overlapping tiles, runs OCR on each, and combines the results.
    """
    h, w, _ = image.shape
    all_results = []

    step_size = tile_size - overlap

    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]

            tile_results = ocr_engine.predict(tile)

            if tile_results and tile_results[0] is not None:
                for res in tile_results[0]:
                    try:
                        # FIX: This entire block is now wrapped in a try-except.
                        # This robustly handles any malformed coordinate (e.g., 'i')
                        # by skipping the entire invalid detection box.
                        box = res[0]
                        adjusted_box = []
                        for point in box:
                            # Attempt to convert and adjust each point.
                            adj_x = float(point[0]) + x
                            adj_y = float(point[1]) + y
                            adjusted_box.append([adj_x, adj_y])

                        # If successful, update the result with the new coordinates and add it.
                        res[0] = adjusted_box
                        all_results.append(res)
                    except (ValueError, TypeError) as e:
                        # If any coordinate is invalid (e.g., 'i'), skip this detection and warn the user.
                        print(f"⚠️ Warning: Skipping malformed coordinate data. Error: {e}. Data: {res}")
                        continue

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Perform OCR on an image or PDF and visualize the results.")
    parser.add_argument("input_path", type=str, help="Path to the input image or PDF file.")
    parser.add_argument("--lang", type=str, default="pl", help="Language for OCR (e.g., 'en', 'pl').")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: File not found at '{input_path}'")
        return

    print(f"Processing '{input_path.name}'...")
    try:
        if input_path.suffix.lower() == ".pdf":
            print("PDF detected, converting the first page at full resolution (300 DPI)...")
            images = convert_from_path(str(input_path), 300)
            if not images:
                print("Error: Could not extract any pages from the PDF.")
                return
            image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            print("Image file detected, loading...")
            image = cv2.imread(str(input_path))
            if image is None:
                print("Error: Could not read the image file.")
                return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    print(f"Initializing PaddleOCR for language: '{args.lang}'...")
    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang)

    print("Running OCR detection using tiling...")
    combined_results = process_image_in_tiles(ocr, image)

    if not combined_results:
        print("No text was detected.")
        return

    print(f"Detected {len(combined_results)} total text blocks across all tiles.")

    print("Creating visualization...")
    annotated_image = create_elegant_visualization(image, combined_results)

    output_filename = f"{input_path.stem}_ocr_result.png"
    cv2.imwrite(output_filename, annotated_image)
    print(f"✅ Success! Visualization saved to '{output_filename}'")


if __name__ == "__main__":
    main()