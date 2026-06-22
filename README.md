# Wider-Fair-Dataset

The project aims to create a subset of the WIDER FACE dataset [1] with annotations linked to protected attributes: **Ethnicity** and **Sex**.

The paper illustrates possible fairness evaluations of face detection models, although this is not the primary purpose of the work.

# Ethical Considerations

It is important to highlight that whenever we refer to ethnicity or sex, we actually refer to **perceived ethnicity** or **perceived sex**, based solely on the images in the dataset and **the judgment of a single human annotator**, as the original dataset does not contain any such information.

# The Dataset

Our dataset is a subset of the WIDER FACE dataset [1]. Since WIDER FACE contains many very small faces, we remove the smallest ones to make manual annotation easier.

## Process

- **Initial filtering:** Remove the smallest faces to facilitate manual annotation.

- **Annotation process:** Each face is annotated with the two sensitive attributes (**Sex** and **Ethnicity**) as well as two special labels used when the annotator cannot determine the category to which an individual belongs (**Undetermined** and **Other**).

- **Final filtering:** Remove any image containing at least one face tagged with a special label.

### Sex Categories

- Male
- Female

### Ethnicity Categories

- White
- Black
- Asian
- Indian

# Dataset Attributes

The dataset contains the following variables:

| Variable | Description |
|-----------|-------------|
| `id` | Row index or unique identifier of a face. |
| `filename` | Name of the image file associated with the sample. |
| `x1` | Left coordinate of the face bounding box (in pixels). |
| `y1` | Top coordinate of the face bounding box (in pixels). |
| `x2` | Right coordinate of the face bounding box (in pixels). |
| `y2` | Bottom coordinate of the face bounding box (in pixels). |
| `blur` | Blur level of the face image. Higher values indicate stronger blur. |
| `expression` | Facial expression score or category provided by the annotation source. |
| `illumination` | Illumination quality score indicating lighting conditions of the face. |
| `valid` | Indicator of validity. |
| `occlusion` | Degree of face occlusion (e.g., sunglasses, masks, hands, or other objects covering the face). |
| `pose` | Face pose score reflecting deviation from a frontal face orientation. |
| `original_width` | Width of the original image in pixels. |
| `original_height` | Height of the original image in pixels. |
| `relative_area` | Relative size of the face bounding box with respect to the original image area. |
| `Sex` | Annotated sex label. |
| `Ethnicity` | Annotated ethnicity label. |
| `Valid` | Binary indicator specifying whether the sample passed quality and filtering criteria. |
| `Sex_Ethnicity` | Combined demographic label formed by concatenating the `Sex` and `Ethnicity` attributes. |
| `area_bin` | Categorical bin derived from `relative_area`, grouping faces by relative size within the image. |

## Bounding Box Coordinates

The face bounding box is defined by:

- `(x1, y1)`: top-left corner
- `(x2, y2)`: bottom-right corner

```text
(x1, y1)  ┌─────────────┐
          │    Face     │
          │             │
          └─────────────┘ (x2, y2)
```

## Notes

- Image dimensions (`original_width`, `original_height`) correspond to the original image before any preprocessing.
- Quality-related variables (`blur`, `illumination`, `occlusion`, `pose`, `expression`, `valid`) originate from the WIDER FACE annotations.
- `relative_area` and `area_bin` can be used to analyze model performance across different face scales.

## Limitations

- The dataset is an **easier version** of the original WIDER FACE dataset. Therefore, it is not intended for benchmarking the absolute performance of a model, but rather for comparing the relative fairness performance of different models.

- The sensitive attributes rely on the judgment of a single annotator, which is a major limitation. Using multiple annotators and a majority-vote procedure would be a valuable extension of this work.

## Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the code
python src/main.py
```

## Core Functions

### Load the Annotation Dataset

```python
from pathlib import Path

df = load_dataset("dataset/test_set.csv")
print(df.head())
```

Loads the annotation CSV file into a Pandas DataFrame.

---

### Load an Image

```python
image_path = Path("dataset/images") / "0--Parade/0_Parade_marchingband_1_849.jpg"
img = load_image(image_path)
```

Loads an image using OpenCV and converts it from BGR to RGB.

---

### Display an Image

```python
show_image(img, title="Original Image")
```

Displays an RGB image using Matplotlib.

---

### Crop a Face from an Image

```python
row = df.iloc[0]

bbox = (
    row["x1"],
    row["y1"],
    row["x2"],
    row["y2"]
)

face = crop_face(img, bbox)
show_image(face, title="Cropped Face")
```

Extracts a face region using the bounding box coordinates.

---

### Draw Bounding Boxes

```python
bbox = (
    row["x1"],
    row["y1"],
    row["x2"],
    row["y2"]
)

img_with_box = draw_bboxes(img, [bbox])

show_image(
    img_with_box,
    title="Image with Bounding Box"
)
```

Draws one or more bounding boxes on an image.

---

### Display All Faces in an Image

```python
filename = df.iloc[0]["filename"]

show_image_with_all_bboxes(
    df,
    Path("dataset/images"),
    filename
)
```

Displays the image with every annotated face bounding box associated with that image.

---

## Complete Example

```python
# Configure paths
CSV_PATH = Path("dataset/test_set.csv")  # Path to CSV file
IMAGE_ROOT = Path("dataset/images")      # Folder containing images

# Load dataset
df = load_dataset(CSV_PATH)

# Pick a row (example: 9th row)
row = df.iloc[9]
filename = row["filename"]
image_path = IMAGE_ROOT / filename

# Load image
img = load_image(image_path)

# Create the bounding box
bbox = (row["x1"], row["y1"], row["x2"], row["y2"])

# 1) Show whole image
show_image(img, title=f"Full image: {Path(filename).name}")

# 2) Show cropped face only
face = crop_face(img, bbox)
show_image(face, title="Cropped face")

# 3) Show full image with the face bounding box overlaid
img_with_box = draw_bboxes(img, [bbox], color=(255, 0, 0), thickness=3)
show_image(img_with_box, title="Image with bounding box")

# 4) Show all faces in the image (if multiple)
show_image_with_all_bboxes(df, IMAGE_ROOT, filename)
```

# References

[1] Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016).
**WIDER FACE: A Face Detection Benchmark**.
In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5525–5533.
