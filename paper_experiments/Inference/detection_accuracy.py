import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import yaml
import os
import numpy as np

def bootstrapCI(df, threshold, group_name, n_iterations=1000, ci=95): 
    n_size = len(df)
    group_levels = df[group_name].unique()
    stats = {group: [] for group in group_levels}  

    for _ in range(n_iterations): 
        sample_df = df.sample(n=n_size, replace=True)
        detected = sample_df[sample_df["Best_Iou"] >= threshold].groupby(group_name).size()
        total = sample_df.groupby(group_name).size()
        detection_rates = detected / total

        for group in group_levels:
            stats[group].append(detection_rates.get(group, 0)) 

    
    ci_intervals = {}
    for group in group_levels:
        lower = np.percentile(stats[group], (100 - ci) / 2)
        upper = np.percentile(stats[group], 100 - (100 - ci) / 2)
        ### AVOID NAN VALUE
        lower = 0 if np.isnan(lower) else round(lower, 2)
        upper = 0 if np.isnan(upper) else round(upper, 2)

        ci_intervals[group] = (lower, upper)

    return ci_intervals 

def mean_CI_group(interval): 
    mean_ci_by_group = {}
    for group in interval[0.5].keys():  
        lower_vals = []
        upper_vals = []
        
        for iou in interval:
            lower, upper = interval[iou].get(group, (0, 0)) 
            lower_vals.append(lower)
            upper_vals.append(upper)

        mean_lower = np.mean(lower_vals)
        mean_upper = np.mean(upper_vals)

        mean_ci_by_group[group] = (round(mean_lower, 2), round(mean_upper, 2))

    return mean_ci_by_group

def compute_and_save_detection_rates(group_name):
    """
    ### DOCSTRINGS DONE BY CHAT GPT
    Computes detection rates and saves the results as CSV files and PNG tables.
    
    Args:
        group_name (str): The name of the group to calculate metrics for (e.g., "race", "gender").
        
    Saves:
        - CSV files with detection rates and confidence intervals for each group.
        - PNG images of tables summarizing recall details and disparity metrics.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ### FILE HANDLING
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    model_name = config["saving_name"]
    DF_FOLDER = "metrics_df_test"
    TABLE_FOLDER = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name, "table_metrics")
    os.makedirs(TABLE_FOLDER, exist_ok=True)

    METRICS_FOLDER = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name, DF_FOLDER)
    metrics_csv_path = os.path.join(METRICS_FOLDER, 'res_IOU.csv')

    df = pd.read_csv(metrics_csv_path)
    df_train = pd.read_csv(config['dataset']['train_set'])
    df_test = pd.read_csv(config['dataset']['test_set'])

    ### Mappings
    race_mapping = config["mappings"]["race"]
    reverse_race_mapping = {v: k for k, v in race_mapping.items()}
    
    gender_mapping = config["mappings"]["gender"]
    reverse_gender_mapping = {v: k for k, v in gender_mapping.items()}

    df["Race"] = df["Race"].map(reverse_race_mapping)
    df["Gender"] = df["Gender"].map(reverse_gender_mapping)

    ### NUmber of samples for training and test
    train_support = df_train.groupby(group_name).size()
    test_support = df_test.groupby(group_name).size()

    ### Calculation of recall across all the iou threshold, for each group
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    detection_rates = {}
    interval = {}
    for threshold in iou_thresholds:
        detected = df[df["Best_Iou"] >= threshold].groupby(group_name).size()
        total = df.groupby(group_name).size()
        detection_rates[round(threshold, 2)] = round((detected / total), 3)
        interval[round(threshold, 2)] = bootstrapCI(df, threshold, group_name)

    mean_ci_by_group = mean_CI_group(interval)
    detection_rates_df = pd.DataFrame(detection_rates).T
    detection_rates_df = detection_rates_df.fillna(0)
    detection_rates_df["Row Std"] = round(detection_rates_df.std(axis=1), 3) ### STD across groups at each iou threshold
    detection_rates_df.loc['Mean Recall Across IoU Threshold'] = round(detection_rates_df.mean(axis=0), 3)
    detection_rates_df.loc['Train Support'] = train_support
    detection_rates_df.loc['Test Support'] = test_support

    interval_df = pd.DataFrame(interval).T

    csv_filename = os.path.join(METRICS_FOLDER, f"detection_rates_{group_name}.csv")
    detection_rates_df.to_csv(csv_filename)
    csv_filename = os.path.join(METRICS_FOLDER, f"detection_rates_CI_{group_name}.csv")
    interval_df.to_csv(csv_filename)
    print(f"Saved CSV: {csv_filename}")

    ### Table with detail of recall by group, Mean of recall across Iou threshold (detected accuracy), Trainsuppoort, Test support
    df_detail_recall_by_group = detection_rates_df.drop(columns=["Row Std"])

    ### For the second table, calculate Detected Accuracy and Disparity
    summary_table_detected_rates_disparity = detection_rates_df.drop(columns=["Row Std"])
    summary_table_detected_rates_disparity = summary_table_detected_rates_disparity.drop(index=["Train Support", "Test Support", "Mean Recall Across IoU Threshold"])  # Remove support columns
    summary_table_detected_rates_disparity['Detected Accuracy'] = round(summary_table_detected_rates_disparity.mean(axis=1), 3)
    summary_table_detected_rates_disparity['Disparity'] = round(summary_table_detected_rates_disparity.std(axis=1), 3)
    summary_table_detected_rates_disparity = summary_table_detected_rates_disparity[['Detected Accuracy', 'Disparity']]

    ### Plotting the first table
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    tbl1 = table(ax, df_detail_recall_by_group, loc='center', cellLoc='center', colWidths=[0.1] * len(detection_rates_df.columns))
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(12)
    tbl1.scale(1.5, 1.5)

    for i, row in df_detail_recall_by_group.iterrows():

        if i in ["Train Support", "Test Support"]: 
            continue
        max_val_idx = row.idxmax()
        min_val_idx = row.idxmin()

        row_index_in_tbl = list(df_detail_recall_by_group.index).index(i) + 1

        tbl1[(row_index_in_tbl, df_detail_recall_by_group.columns.get_loc(max_val_idx))].set_facecolor('#90EE90')  # Green for max
        tbl1[(row_index_in_tbl, df_detail_recall_by_group.columns.get_loc(min_val_idx))].set_facecolor('#FF6347')  # Red for min

    png_filename = os.path.join(TABLE_FOLDER, f"table_recall_detail_support_{group_name}.png")
    plt.savefig(png_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved PNG: {png_filename}")

    ### Plotting the second table
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    tbl2 = table(ax, summary_table_detected_rates_disparity, loc='center', cellLoc='center', colWidths=[0.1] * len(summary_table_detected_rates_disparity.columns))
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(12)
    tbl2.scale(1.5, 1.5)

    png_filename = os.path.join(TABLE_FOLDER, f"table_detectionrates_disparity_{group_name}.png")
    plt.savefig(png_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved PNG: {png_filename}")

    csv_filename = os.path.join(METRICS_FOLDER, f"recall_detail_support_{group_name}.csv")
    df_detail_recall_by_group.to_csv(csv_filename)
    print(f"Saved CSV: {csv_filename}")

    ### Save the second table (Detected Accuracy and Disparity)
    csv_filename = os.path.join(METRICS_FOLDER, f"detectionrates_disparity_{group_name}.csv")
    summary_table_detected_rates_disparity.to_csv(csv_filename)
    print(f"Saved CSV: {csv_filename}")

    


