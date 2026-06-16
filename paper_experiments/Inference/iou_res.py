import os
import pandas as pd
import yaml
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Training.utility import calculate_iou_2

def calculation_IOU(): 
    """
    ### DOCSTRINGS DONE BY CHATGPT
    Calculates the Intersection over Union (IoU) between ground truth bounding boxes and detected 
    bounding boxes for each image in the dataset. The function matches ground truth boxes with the 
    best detections based on IoU and computes the corresponding IoU values for each image. It then 
    stores the results, including class information for each matched 
    ground truth and its best detection, in a CSV file.

    The function performs the following steps:
    1. Reads ground truth and detection files for each image in the dataset.
    2. For each ground truth bounding box, it computes the IoU with all detection bounding boxes.
    3. The detection with the highest IoU for each ground truth is selected as the best match.
    4. The IoU value, along with additional information (class, occlusion, and relative area), is 
       recorded for each matched ground truth and detection pair.
    5. The results are saved in a CSV file containing the IoU values
       information for each matched ground truth and its best detection.

    Args:
        None

    Returns:
        None

    Saves:
        - A CSV file (`res_IOU.csv`) containing the calculated IoU values, along with associated 
          information (race, gender, occlusion, relative area) for each ground truth and its best 
          detection.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ### FILE HANDLING
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    GROUND_TRUTH_FOLDER = "test_groundtruths"
    DETECTION_FOLDER = "test_detections"
    DF_FOLDER = "metrics_df_test"
    model_name = config["saving_name"]
    DETECTIONS_PATH  = os.path.join(RES_FOLDER, GROUP_FOLDER,model_name,DETECTION_FOLDER)
    GROUND_TRUTH_PATH = os.path.join(RES_FOLDER, GROUP_FOLDER,model_name,GROUND_TRUTH_FOLDER)
    METRICS_DF_PATH = os.path.join(RES_FOLDER,GROUP_FOLDER, model_name, DF_FOLDER )
    

    data = []
    confidence_threshold = config["Recall"]["confidence_threshold"]
    ### For each images
    for gt_file in os.listdir(GROUND_TRUTH_PATH): 
        image_name = gt_file.replace(".txt", "")
        
        ### Collect each  gt in the image 
        gt_boxes = []
        with open(os.path.join(GROUND_TRUTH_PATH, gt_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                class_race = parts[0]
                class_gender = parts[1] 
                bbox = list(map(float, parts[2:6]))### [x1, y1, x2, y2]
                gt_boxes.append((class_race,class_gender, bbox))

        ### Collect each prediction in the image
        det_file = os.path.join(DETECTIONS_PATH, image_name + ".txt")
        det_boxes = []
        if os.path.exists(det_file):
            with open(det_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    confidence = float(parts[1])
                    if confidence < confidence_threshold : 
                        break
                    bbox = list(map(float, parts[2:]))  # [x1, y1, x2, y2]
                    det_boxes.append((confidence, bbox))

        ### Match each gt with the best detections possible
        matched_detections = set()  
        for gt_race, gt_gender, gt_bbox in gt_boxes:
            best_iou = 0
            best_det_idx = -1
            best_conf = 0

            for idx, (det_confidence, det_bbox) in enumerate(det_boxes):
                if idx in matched_detections:
                    continue 
                iou = calculate_iou_2(gt_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou.item()
                    best_det_idx = idx
                    best_conf = det_confidence

            matched_detections.add(best_det_idx)
            data.append([image_name, gt_race,gt_gender, best_iou, best_conf])
            
    ### Create dataframe
    metrics_csv_path = os.path.join(METRICS_DF_PATH, 'res_IOU.csv')
    df = pd.DataFrame(data, columns=["Image", "Race","Gender", "Best_Iou", "Best_Confidence"])
    df.to_csv(metrics_csv_path, index=False)
