import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
import matplotlib.pyplot as plt
import statistics
from ruamel.yaml import YAML

def metrics_calculation(split):
    """
    ### DOCSTRINGS DONE BY CHATGPT
    Calculates and visualizes precision, recall, and F1-score metrics for object detection results, 
    based on True Positive (TP) and False Positive (FP) classifications, for each Intersection over 
    Union (IoU) threshold. This function processes the results of object detections, calculates 
    performance metrics, generates precision-recall curves, and plots precision, recall, and F1-score 
    against confidence levels for various thresholds. Additionally, it updates the configuration file 
    with the confidence threshold of intersection between precision and recall (PRBEP).

    The function performs the following steps:
    1. Reads and processes TP/FP results for different IoU thresholds from CSV files.
    2. Computes cumulative sums of TP and FP, calculates precision, recall, and F1-score.
    3. Identifies the best F1-score and the point where precision and recall intersect (PRBEP).
    4. Visualizes precision-recall points and plots the relationship between precision, recall, 
       and F1-score versus confidence levels.
    5. Saves the transformed results to CSV files.
    6. Updates the configuration file with the optimal confidence threshold for recall based on 
       the intersection of precision and recall.

    Args:
        None

    Returns:
        None

    Saves:
        - Transformed TP/FP results to CSV files (`transformed_res_TP_FP{thresh}.csv`).
        - Precision-Recall curves and plots saved in the `precision_recall` folder.
        - Updates the `config.yaml` file with the optimal confidence threshold for recall.
    """
    yaml = YAML()
    with open("config.yaml", "r") as f:
        config = yaml.load(f)

    ### FILE HANDLING
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]

    if split == 'test' : 
        GROUND_TRUTH_FOLDER = "test_groundtruths"
        DF_FOLDER = "metrics_df_test"
    else : 
        GROUND_TRUTH_FOLDER = "val_groundtruths"
        DF_FOLDER = "metrics_df_val"

    model_name = config["saving_name"]
    GROUND_TRUTH_PATH = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name, GROUND_TRUTH_FOLDER)
    METRICS_DF_PATH = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name, DF_FOLDER)
    GRAPH_FOLDER = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name, "precision_recall")
    os.makedirs(GRAPH_FOLDER, exist_ok=True)

    iou_threshold = config["iou_threshold"]
    intersection_confidence_list = []
    ap_values = {}
    for thresh in iou_threshold:
        metrics_csv_path = os.path.join(METRICS_DF_PATH, f"res_TP_FP{thresh}.csv")
        df = pd.read_csv(metrics_csv_path)

        total_bounding_boxes = sum(
            len(open(os.path.join(GROUND_TRUTH_PATH, gt_file)).readlines())
            for gt_file in os.listdir(GROUND_TRUTH_PATH)
            if os.path.isfile(os.path.join(GROUND_TRUTH_PATH, gt_file))
        )

        ### Transform Tp or fp into binary variable, add it to dataframe
        df["TP"] = (df["TP or FP"] == "TP").astype(int)
        df["FP"] = (df["TP or FP"] == "FP").astype(int)
        df["Confidences"] = df["Confidences"].str.rstrip("%").astype(float)
        df = df.sort_values(by="Confidences", ascending=False).reset_index(drop=True)
        df["Acc TP"] = df["TP"].cumsum()
        df["Acc FP"] = df["FP"].cumsum()
        df["Precision"] = df["Acc TP"] / (df["Acc TP"] + df["Acc FP"])
        df["Recall"] = df["Acc TP"] / total_bounding_boxes

        ### Compute F1-score
        df["F1-score"] = (2 * df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
        df["F1-score"] = df["F1-score"].fillna(0)

        max_f1_idx = df["F1-score"].idxmax()
        max_f1_confidence = df.loc[max_f1_idx, "Confidences"]
        max_f1_value = df.loc[max_f1_idx, "F1-score"]

        ### FIND PRBEP

        exceed_idx = (df["Recall"] > df["Precision"]).idxmax()  
        if exceed_idx > 0:
            prev_confidence = df.loc[exceed_idx - 1, "Confidences"]
            curr_confidence = df.loc[exceed_idx, "Confidences"]
            intersection_confidence = (prev_confidence + curr_confidence) / 2
            intersection_precision = df.loc[exceed_idx, "Precision"]
        else:
            ### placeholder for safety 
            intersection_confidence = df.loc[exceed_idx, "Confidences"]
            intersection_precision = df.loc[exceed_idx, "Precision"]


        intersection_confidence_list.append(intersection_confidence)

        metrics_csv_path = os.path.join(METRICS_DF_PATH, f"transformed_res_TP_FP{thresh}.csv")
        df.to_csv(metrics_csv_path, index=False)

        ### Plot Precision-Recall Points
        plt.figure(figsize=(8, 6))
        plt.scatter(df["Recall"], df["Precision"], color="blue", label="Precision-Recall Points")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Points (IoU Threshold = {thresh})")
        plt.legend(loc="best")
        plt.grid(True)
        if split == 'test':
            plt.savefig(os.path.join(GRAPH_FOLDER, f"precision_recall_curve{int(thresh*100)}.png"))
        plt.close() 

        average_precision = auc(df["Recall"], df["Precision"])
        ap_values[thresh] = average_precision
        ### Plot Precision, Recall, and F1-score vs Confidence Level
        plt.figure(figsize=(10, 6))
        plt.plot(df["Confidences"], df["Precision"], color="blue", label="Precision")
        plt.plot(df["Confidences"], df["Recall"], color="green", label="Recall")
        plt.scatter(intersection_confidence, intersection_precision, color="red", s=100, label=f"PRBEP at confidence {intersection_confidence :.2f}") 
        plt.axvline(x=max_f1_confidence, color="black", linestyle="--", label=f"max_F1 {max_f1_value:.2f} at confidence {max_f1_confidence:.2f}")
        plt.xlabel("Confidence Level")
        plt.ylabel("Metric Value")
        plt.title(f"Precision, Recall, and F1-score vs Confidence Level\n(AP{int(thresh*100)} = {average_precision:.4f})")
        plt.legend(loc="best")
        plt.grid(True)
        if split == 'test': 
            plt.savefig(os.path.join(GRAPH_FOLDER, f"precision_recall_f1_vs_confidence{int(thresh*100)}.png"))
        plt.close() 

    ap_df = pd.DataFrame.from_dict(ap_values, orient="index", columns=["AP"])
    ap_csv_path = os.path.join(METRICS_DF_PATH, "average_precision_values.csv")
    ap_df.to_csv(ap_csv_path)
    print(ap_df)

    if split == 'val':
    ### Safety check link to round errors
        mode_value = statistics.mode(intersection_confidence_list)
        print(mode_value)
        config["Recall"]["confidence_threshold"] = float(round(mode_value, 2)) / 100
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)



