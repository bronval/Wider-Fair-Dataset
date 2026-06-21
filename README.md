# Wider-Fair-Dataset

The project aims to primarily create a subset of the WiderFace (citation) with annotations linked to protected attributes : Ethnicity and Sex. 

In the paper, the possible evaluation of a model of face detection are illustrated even if they are not the main purpose of the work. 



# Ethical Considerations

It is important to highlight that whenever we refer to the ethnicity or the sex, we actually refer to the perceived ethnicity or sex,
based only on the images from the dataset and **one human annotator’s judgment**, as the original dataset does not
contain any of such information.


# The dataset

Our dataset is a subset of the WIDERFACE dataset (citation). As the WIDERFACE contains a lot of small faces, we remove the smallest faces from it in order to make it easier to annotated. 

## Process 

- Initial Filter : remove the smallest faces to facilitate manual annotation.

- Annotation process : each face is annotated with the 2 sensitives attributes (Sex and Ethnicity) and two specials annotation that are used when the annotator don't know to wich category an individual belong (Undetermined and Other).

- Final filtering : We remove any picture that contains a face tagged with a special annotation. 

### Sex Categories

- Male
- Female

### Ethnicity Categories

- White
- Black
- Asian
- Indian

## What is inside the dataset 

!Add table with composition of the dataset! 

# Dataset Attributes

The dataset contains the following variables:

| Variable          | Description                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| `id`      | Row index or unique identifier of a face.                            |
| `filename`        | Name of the image file associated with the sample.                                                    |
| `x1`              | Left coordinate of the face bounding box (in pixels).                                                 |
| `y1`              | Top coordinate of the face bounding box (in pixels).                                                  |
| `x2`              | Right coordinate of the face bounding box (in pixels).                                                |
| `y2`              | Bottom coordinate of the face bounding box (in pixels).                                               |
| `blur`            | Blur level of the face image. Higher values indicate stronger blur.                         |
| `expression`      | Facial expression score or category provided by the annotation source.                                |
| `illumination`    | Illumination quality score indicating lighting conditions of the face.                                |
| `invalid`         | Indicator of invalid face samples. |
| `occlusion`       | Degree of face occlusion (e.g., sunglasses, masks, hands, or other objects covering the face).        |
| `pose`            | Face pose score reflecting deviation from a frontal face orientation.                                 |
| `original_width`  | Width of the original image in pixels.                                                                |
| `original_height` | Height of the original image in pixels.                                                               |
| `relative_area`   | Relative size of the face bounding box with respect to the original image area.                       |
| `Sex`          | Annotated sex label.                                                                  |
| `Ethnicity`            | Annotated ethnicity label.                                                          |
| `Valid`           | Binary indicator specifying whether the sample passed quality and filtering criteria.                 |
| `Sex_Ethnicity`     | Combined demographic label formed by concatenating the `Sex` and `Ethnicity` attributes.                |
| `area_bin`        | Categorical bin derived from `relative_area`, grouping faces by relative size within the image.       |

## Bounding Box Coordinates

The face bounding box is defined by:

* `(x1, y1)`: top-left corner
* `(x2, y2)`: bottom-right corner

```text
(x1, y1)  ┌─────────────┐
          │    Face     │
          │             │
          └─────────────┘ (x2, y2)
```

## Notes

* Image dimensions (`original_width`, `original_height`) correspond to the original image before any preprocessing.
* Quality-related variables (`blur`, `illumination`, `occlusion`, `pose`, `expression`, `invalid`) originate from WIDERFACE annotation. 
* `relative_area` and `area_bin` can be used to analyze model performance across different face scales.


## Limitations

- The dataset is an **easiest version** of the original WIDERFACE. Thus, it is not designed for benchmark absolute performance of a model but uniquely relative performance between two models in term of fairness. 

- The sensitive attributes rely on a single annotator judgement and that is a major limitation. However a majority vote could be a good extension to our work.


## Installation

```
 1- Create venv : python -m venv .venv 
 2- Activate : .venv\Scripts\activate 
 3- Populate venv : pip install -r requirements.txt
 4- Running the code : python src/main.py
 ```

## Core Functions

### Load the annotations dataset

```python
from pathlib import Path

df = load_dataset("dataset/test_set.csv")
print(df.head())
```

Loads the annotation CSV into a Pandas DataFrame.

---

### Load an image

```python
image_path = Path("dataset/images") / "0--Parade/0_Parade_marchingband_1_849.jpg"
img = load_image(image_path)
```

Loads an image using OpenCV and converts it from BGR to RGB.

---

### Display an image

```python
show_image(img, title="Original Image")
```

Displays an RGB image using Matplotlib.

---

### Crop a face from an image

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

### Draw bounding boxes

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

### Display all faces in an image

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

    # Create the bbox
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
```
