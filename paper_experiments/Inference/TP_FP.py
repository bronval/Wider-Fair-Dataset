import os
import pandas as pd
import yaml
import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Training.dataset import *
from Training.utility import *

def calculation_TP_FP(split):
    """
    ### DOCSTRINGS DONE BY CHATGPT
    Calculates True Positive (TP) and False Positive (FP) values for object detection results 
    based on Intersection over Union (IoU) thresholds. This function compares the predicted 
    bounding boxes with ground truth boxes, and identifies which detections are True Positives 
    (TPs) and which are False Positives (FPs) for each image.

    For each image in the ground truth and detection datasets, the function computes the IoU 
    between predicted bounding boxes and ground truth boxes, and classifies each detection as 
    either a TP or FP. A detection is considered a TP if it has an IoU greater than a given 
    threshold with a corresponding ground truth box. Otherwise, it is considered an FP.

    Args:
        None

    Returns:
        None

    Saves:
        - A CSV file for each IoU threshold in the specified folder, containing the following columns:
            ["Images", "Race", "Gender", "Confidences", "TP or FP"]
          - "Images": The name of the image.
          - "Race": The race class label of the ground truth box.
          - "Gender": The gender class label of the ground truth box.
          - "Confidences": The confidence score of the detection (in percentage).
          - "TP or FP": The classification of the detection as a "TP" (True Positive) or "FP" (False Positive).
    """

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ### File handling
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    if split == 'test' : 
        GROUND_TRUTH_FOLDER = "test_groundtruths"
        DETECTION_FOLDER = "test_detections"
        DF_FOLDER = "metrics_df_test"
    else : 
        GROUND_TRUTH_FOLDER = "val_groundtruths"
        DETECTION_FOLDER = "val_detections"
        DF_FOLDER = "metrics_df_val"

    model_name = config["saving_name"]
    GROUND_TRUTH_PATH = os.path.join(RES_FOLDER, GROUP_FOLDER,model_name,GROUND_TRUTH_FOLDER)
    DETECTIONS_PATH  = os.path.join(RES_FOLDER, GROUP_FOLDER,model_name,DETECTION_FOLDER)
    METRICS_DF_PATH = os.path.join(RES_FOLDER,GROUP_FOLDER, model_name, DF_FOLDER )
    os.makedirs(METRICS_DF_PATH, exist_ok=True)

    
    iou_threshold = config["iou_threshold"]

    for thresh in iou_threshold : 
        data = []
        ### For each images
        for gt_file in os.listdir(GROUND_TRUTH_PATH):
            image_name = gt_file.replace(".txt", "")
            
            ### Collect each gt in the image 
            gt_boxes = []
            with open(os.path.join(GROUND_TRUTH_PATH, gt_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_race = parts[0]
                    class_gender = parts[1]
                    bbox = list(map(float, parts[2:6]))  # [x1, y1, x2, y2]
                    gt_boxes.append((class_race,class_gender, bbox))

            ### Collect each prediction in the image
            det_file = os.path.join(DETECTIONS_PATH, image_name + ".txt")
            det_boxes = []
            if os.path.exists(det_file):
                with open(det_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        class_name = parts[0]
                        confidence = float(parts[1])
                        if confidence <0.1 : 
                            break
                        bbox = list(map(float, parts[2:]))  # [x1, y1, x2, y2]
                        det_boxes.append((class_name, confidence, bbox))

            ### Match each gt with the best detections possible
            matched_detections = set()  
            for gt_race,gt_gender, gt_bbox in gt_boxes:
                best_iou = 0
                best_det_idx = -1

                for idx, (det_class, det_confidence, det_bbox) in enumerate(det_boxes):
                    if idx in matched_detections:
                        continue 
                    iou = calculate_iou_2(gt_bbox, det_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx

                if best_iou > thresh:
                    matched_detections.add(best_det_idx)
                    data.append([image_name, gt_race, gt_gender, f"{det_boxes[best_det_idx][1]*100:.0f}%", "TP"])
                
            ### Add unmatched detections as FP
            for idx, (det_class, det_confidence, det_bbox) in enumerate(det_boxes):
                if idx not in matched_detections:
                    data.append([image_name, None, None, f"{det_confidence*100:.0f}%", "FP"])

        ### Create dataframe
        metrics_csv_path = os.path.join(METRICS_DF_PATH, f'res_TP_FP{thresh}.csv')
        df = pd.DataFrame(data, columns=["Images", "Race", "Gender",  "Confidences", "TP or FP"])
        df.to_csv(metrics_csv_path, index=False)

