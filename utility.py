import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

# Types
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load annotation CSV into a DataFrame."""
    return pd.read_csv(csv_path)

def load_image(image_path: Path):
    """Load image with OpenCV and convert to RGB."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_image(img, title: str = "Image"):
    """Display an RGB image using matplotlib."""
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

def crop_face(img, bbox: BBox):
    """Crop face from image using pixel bbox (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]

def draw_bboxes(img, bboxes: List[BBox], color=(255, 0, 0), thickness: int = 2):
    """Return a copy of image with rectangles drawn (RGB coords)."""
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out

def show_image_with_all_bboxes(df: pd.DataFrame, image_root: Path, filename: str):
    """
    Load an image and display all bounding boxes associated with it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing columns: filename, x1, y1, x2, y2
    image_root : Path
        Root directory containing images.
    filename : str
        Image filename (can include subfolders).
    """
    # Select all annotations for this image
    rows = df[df["filename"] == filename]

    if rows.empty:
        raise ValueError(f"No annotations found for: {filename}")

    # Load image
    image_path = image_root / Path(filename)
    img = load_image(image_path)

    # Extract all bounding boxes
    bboxes = rows[["x1", "y1", "x2", "y2"]].values.tolist()

    # Draw all boxes
    img_with_boxes = draw_bboxes(img, bboxes, color=(255, 0, 0), thickness=3)

    # Display
    show_image(
        img_with_boxes,
        title=f"{Path(filename).name} ({len(bboxes)} faces)"
    )

# -----------------------
# Example usage (README)
# -----------------------
if __name__ == "__main__":
    # Configure paths
    CSV_PATH = Path("dataset/test_set.csv") # path to CSV
    IMAGE_ROOT = Path("dataset/images")  # folder with images

    # Load dataset
    df = load_dataset(CSV_PATH)

    # Pick a row (example: 9 th row)
    row = df.iloc[9]
    filename = row["filename"] 
    image_path = IMAGE_ROOT / filename

    # Load image
    img = load_image(image_path)

    # The CSV is assumed to have pixel bbox columns: x1, y1, x2, y2
    bbox = (row["x1"], row["y1"], row["x2"], row["y2"])

    # 1) Show whole image
    show_image(img, title=f"Full image: {Path(filename).name}")

    # 2) Show cropped face only
    face = crop_face(img, bbox)
    show_image(face, title="Cropped face")

    # 3) Show full image with the face bbox overlaid
    img_with_box = draw_bboxes(img, [bbox], color=(255, 0, 0), thickness=3)
    show_image(img_with_box, title="Image with bounding box")

    # 4) Show all faces in the image (if multiple)
    show_image_with_all_bboxes(df, IMAGE_ROOT, filename)