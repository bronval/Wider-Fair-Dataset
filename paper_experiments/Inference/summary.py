import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import yaml 
from matplotlib.lines import Line2D
import numpy as np

def format_group_name(group_folder):
    if group_folder.endswith('N'):
        group_folder = group_folder[:-1]
    return group_folder.replace("_", " ").title()

def load_and_clean_csv(path):
    df = pd.read_csv(path, index_col=0)
    # Drop unwanted rows
    df = df.drop(index=["Mean Recall Across IoU Threshold", "Train Support", "Test Support"], errors='ignore')
    return df

def get_train_support_row(path):
    df = pd.read_csv(path, index_col=0)
    return df.loc["Train Support"]

def compute_mean_std(dfs):
    stacked = pd.concat(dfs, axis=0, keys=range(len(dfs)))
    mean_df = stacked.groupby(level=1).mean() * 100
    std_df = stacked.groupby(level=1).std() * 100
    return mean_df, std_df

def format_with_std(mean, std):
    return mean.round(2).astype(str) + " $\\pm$ " + std.round(2).astype(str)

def format_latex_df(mean_df, std_df):
    latex_df = mean_df.copy()
    for col in mean_df.columns:
        latex_df[col] = format_with_std(mean_df[col], std_df[col])
    return latex_df

def add_train_support_row(latex_df, train_support_row):
    train_support_fmt = train_support_row.fillna(0).astype(int).astype(str)
    latex_df.loc["Train Support"] = train_support_fmt
    return latex_df

def bold_max_per_row(latex_df, mean_df, train_support_row):
    mean_plus_support = pd.concat([mean_df, pd.DataFrame([train_support_row], index=["Train Support"])])
    for idx, row in mean_plus_support.iterrows():
        max_val = row.max()
        max_cols = row[row == max_val].index
        for col in max_cols:
            if idx in latex_df.index:
                latex_df.at[idx, col] = "\\textbf{" + latex_df.at[idx, col] + "}"
    return latex_df

def add_aggregate_row(latex_df, mean_df, train_support_label="Train Support", aggregate_label="Average(across IoU)"):
    # Calculate mean across IoU (excluding train support)
    iou_only_df = mean_df.loc[mean_df.index != train_support_label] if train_support_label in mean_df.index else mean_df
    mean_across_iou = iou_only_df.mean()

    mean_row_fmt = mean_across_iou.round(2).astype(str)

    # Ensure index has the correct name for reset_index
    latex_df.index.name = "IoU"
    latex_df = latex_df.reset_index()

    # Use the actual column name, which should be 'IoU'
    train_support_idx = latex_df[latex_df['IoU'] == train_support_label].index[0]

    mean_row_df = pd.DataFrame([[aggregate_label] + mean_row_fmt.tolist()], columns=latex_df.columns)
    top = latex_df.iloc[:train_support_idx + 1]
    bottom = latex_df.iloc[train_support_idx + 1:]
    latex_df = pd.concat([top, mean_row_df, bottom], ignore_index=True)
    return latex_df, mean_across_iou

def bold_only_mean(value_str):
    # Split string at ' $\pm$ '
    parts = value_str.split(' $\\pm$ ')
    if len(parts) == 2:
        mean_part, std_part = parts
        # Bold only the mean part, leave std part as is
        return f"\\textbf{{{mean_part}}} $\\pm$ {std_part}"
    else:
        # If no ± found, just bold whole string as fallback
        return f"\\textbf{{{value_str}}}"

def bold_all_max_rows(latex_df, mean_df, train_support_row, mean_across_iou, train_support_label="Train Support", aggregate_label="Average(across IoU)"):
    mean_plus_support = pd.concat([
        mean_df,
        pd.DataFrame([train_support_row], index=[train_support_label]),
        pd.DataFrame([mean_across_iou], index=[aggregate_label])
    ])

    for idx, row in mean_plus_support.iterrows():
        max_val = row.max()
        max_cols = row[row == max_val].index
        for col in max_cols:
            try:
                val = latex_df.loc[latex_df['IoU'] == idx, col].values[0]
                if val != '-':
                    latex_df.loc[latex_df['IoU'] == idx, col] = bold_only_mean(val)
            except IndexError:
                pass  # skip if idx not found
    return latex_df


def clean_and_rename_columns(latex_df):
    latex_df = latex_df.replace('0', '-').fillna('-')
    cols = latex_df.columns.tolist()
    cols[-1] = "Aggregate(Disparity)"
    latex_df.columns = cols
    return latex_df

def generate_latex_table(latex_df, col_fmt):
    latex_str = latex_df.to_latex(
        escape=False,
        column_format=col_fmt,
        index=False,
        header=True
    )

    lines = latex_str.splitlines()

    ### Remove default \toprule, \midrule, \bottomrule lines if present
    lines = [line for line in lines if not line.strip() in ['\\toprule', '\\midrule']]

    ### Insert \hline before Train Support and Aggregate rows
    insert_indices = []
    for i, line in enumerate(lines):
        if "Train Support" in line or "Average(across IoU)" in line:
            insert_indices.append(i)
    for idx in reversed(insert_indices):
        lines.insert(idx, r"\hline")

    for i, line in enumerate(lines):
        if line.strip() in ['\\toprule', '\\midrule', '\\bottomrule']:
            lines[i] = r'\hline'
    for i, line in enumerate(lines):
        if ' & ' in line and not line.strip().startswith(r'\hline'):
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith(r'\hline'):
                lines.insert(i + 1, r'\hline')
            break
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\begin{tabular}'):
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith(r'\hline'):
                lines.insert(i + 1, r'\hline')
            break


    latex_str = "\n".join(lines)

    ### Wrap with table environment and formatting
    #latex_str = "\\begin{table}[h!]\n\\centering\n\\small\n" + latex_str + "\n\\end{table}"

    return latex_str

def extract_first_train_support_and_aggregate(latex_df, train_support_label="Train Support", aggregate_label="Average(across IoU)"):
    ### If index is a column, set it as index
    if 'IoU' in latex_df.columns:
        df = latex_df.set_index('IoU')
    else:
        df = latex_df.copy()
        if df.index.name != "IoU":
            df.index.name = "IoU"

    ### Select the three rows: first row, Train Support, Aggregate
    selected_rows = df.loc[[train_support_label, aggregate_label]]


    return selected_rows.reset_index()

def main(dfs, train_support_row):
    ### Compute mean and std
    mean_df, std_df = compute_mean_std(dfs)

    ### Format dataframe with mean ± std strings
    latex_df = format_latex_df(mean_df, std_df)

    ### Add Train Support row
    latex_df = add_train_support_row(latex_df, train_support_row)

    ### Add Aggregate row and get the mean values
    latex_df, mean_across_iou = add_aggregate_row(latex_df, mean_df)

    ### Bold max values including Aggregate row and Train Support again
    latex_df = bold_all_max_rows(latex_df, mean_df, train_support_row, mean_across_iou)

    ### Clean zeros and rename last column
    latex_df = clean_and_rename_columns(latex_df)

    small_table = extract_first_train_support_and_aggregate(latex_df)
    small_table.columns.values[0] = '~'
    ### Prepare column format string
    n_cols = latex_df.shape[1]
    col_fmt = "|l|" + "c" * (n_cols - 2) + "|c|"

    ### Generate latex table
    latex_big = generate_latex_table(latex_df, col_fmt)

    latex_small = generate_latex_table(small_table, col_fmt)

    return latex_big, latex_small

def global_performance(group): 
    """
    #### DOCSTRINGS DONE BY CAHT GPT
    Generates and saves performance plots for detection accuracy, average precision (AP), 
    and standard deviation (Disparity) across different IoU thresholds for a given group (Gender or Race).

    Args:
        group (str): The group for which performance metrics are generated. Can be 'Gender' or 'Race'.

    Creates:
        - Line plots showing mean detection accuracy, mean average precision (AP), 
          and mean disparity (Std) across IoU thresholds.
        - Saves the plots as PNG files.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ### FILE HANDLING
    CHECKPOINTS_FOLDER = "Checkpoints"
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    SUMMARY_FOLDER = "Global_summary" 
    num_subdirectories = len([d for d in os.listdir(os.path.join(CHECKPOINTS_FOLDER,GROUP_FOLDER))])
    DF_PATH = "metrics_df_test"
    
    if group == 'Gender': 
        DF_NAME = "detection_rates_Gender.csv"
        target_folder = ['global_sex', 'loso_female', 'loso_male']
    else: 
        DF_NAME = "detection_rates_Race.csv"
        target_folder = ['global_ethnicity', 'loeo_asian', 'loeo_black', 'loeo_indian']
    
    SAVING_PATH = os.path.join(RES_FOLDER, SUMMARY_FOLDER)
    os.makedirs(SAVING_PATH, exist_ok=True)

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(18, 10), sharex=False)

    for idx, folder in enumerate(target_folder): 
        dfs_detection_accuracy = []
        dfs_ap = []
        dfs_std = []
        seeds = [folder + "_" + str(i) for i in range(num_subdirectories)]
        
        for i, seed in enumerate(seeds):
            full_path = os.path.join(RES_FOLDER, folder, seed, DF_PATH, DF_NAME)
            df = pd.read_csv(full_path, index_col=0)  
            
            ### detection Accuracy 
            df_cleaned_detection_accuracy = df.drop(index=["Mean Recall Across IoU Threshold", "Train Support", "Test Support"])
            df_cleaned_detection_accuracy = df_cleaned_detection_accuracy.drop(columns="Row Std")
            dfs_detection_accuracy.append(df_cleaned_detection_accuracy)
            
            ### Standard Deviation (Disparity)
            df_std = df.drop(index=["Mean Recall Across IoU Threshold", "Train Support", "Test Support"])
            df_std = df_std['Row Std']
            dfs_std.append(df_std)

            ### Average Precision (AP)
            full_path = os.path.join(RES_FOLDER, folder, seed, DF_PATH, 'average_precision_values.csv')
            df_ap = pd.read_csv(full_path, index_col=0)  
            dfs_ap.append(df_ap)

        color = plt.cm.tab10(idx)
        
        # Aggregate detection Accuracy across seeds 
        df_combined = pd.concat(dfs_detection_accuracy, axis=0)
        aggregation_df = df_combined.groupby(df_combined.index).mean()  
        """ax1.plot(aggregation_df.index, aggregation_df.mean(axis=1), 
                 label=f'{folder}', marker='o', linestyle='-', color=color)"""
        
        # Aggregate AP across seeds
        df_combined_ap = pd.concat(dfs_ap, axis=0)
        aggregation_df = df_combined_ap.groupby(df_combined.index).mean()
        ax2.plot(aggregation_df.index, aggregation_df, 
                 label=f'{folder}', marker='o', linestyle='-', color=color)
        
        # Aggregate Std across seeds
        df_combined_std = pd.concat(dfs_std, axis=0)
        aggregation_df = df_combined_std.groupby(df_combined.index).mean() 
        ax3.plot(aggregation_df.index, aggregation_df, 
                 label=f'{folder}', marker='o', linestyle='-', color=color)

    """# Set plot labels, titles, and legends
    ax1.set_xlabel('IoU Threshold')
    ax1.set_ylabel('Aggregate Detection Accuracy')
    ax1.set_title(f'Aggregate detection Accuracy Across Seeds for {group}')
    ax1.legend()"""

    if group == 'Gender': 
        group_name = 'Sex'
    else : 
        group_name = 'Ethnicity'

    ax2.set_xlabel('IoU Threshold', fontsize=14)
    ax2.set_ylabel('Aggregate (Average Precision)', fontsize=14)
    ax2.set_title(f'Aggregate (Average Precision) for {group_name}', fontsize=16)
    ax2.legend()

    ax3.set_xlabel('IoU Threshold', fontsize=14)
    ax3.set_ylabel('Aggregate (Disparity)', fontsize=14)
    ax3.set_title(f'Aggregate (Disparity) for {group_name}', fontsize=16)
    ax3.legend()

    # Save the plots
    plt.tight_layout()
    plot_filename = os.path.join(SAVING_PATH, f"Aggregate_detection_Accuracy_STD_AP_across_seeds_{group}.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=600)
    plt.close()




def generate_summary():
    """
    #### DOCSTRINGS DONE BY CAHT GPT
    Generates and saves summary plots that visualize the trends and statistics 
    (mean, median, min, max) across different IoU thresholds for the given experimental group.

    Creates:
        - A plot showing trends across seeds in terms of recall statistics (median, min, max).
        - A plot showing the mean across all seeds for each IoU threshold.
        - A bar plot showing the mean recall for each class across the seeds.

    Saves:
        - Summary plots in the 'Results' folder in a directory named 'summary'.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ### FILE HANDLING
    CHECKPOINTS_FOLDER = "Checkpoints"
    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    SUMMARY_FOLDER = "summary" 
    num_subdirectories = len([d for d in os.listdir(os.path.join(CHECKPOINTS_FOLDER,GROUP_FOLDER))])
    seeds = [GROUP_FOLDER + "_" + str(i) for i in range(num_subdirectories)]
    DF_PATH = "metrics_df_test"
    
    if config["saving_folder"] in ["global_sex", "global_sexN" ,"loso_male", "loso_female", "loso_femaleN", "loso_maleN"]: 
        DF_NAME = "detection_rates_Gender.csv"
    else: 
        DF_NAME = "detection_rates_Race.csv"
    
    SAVING_PATH = os.path.join(RES_FOLDER, GROUP_FOLDER, SUMMARY_FOLDER)
    os.makedirs(SAVING_PATH, exist_ok=True)

    dfs = []
    train_support_row = None

    for i, seed in enumerate(seeds):
        full_path = os.path.join(RES_FOLDER, GROUP_FOLDER, seed, DF_PATH, DF_NAME)
        df = pd.read_csv(full_path, index_col=0)

        if train_support_row is None:
            train_support_row = df.loc["Train Support"]

        df_cleaned = df.drop(index=["Mean Recall Across IoU Threshold", "Train Support", "Test Support"])
        dfs.append(df_cleaned)

    latex_big, latex_small =main(dfs, train_support_row)

    latex_filename = os.path.join(SAVING_PATH, "Recap_Big.tex")
    with open(latex_filename, "w") as f:
        f.write(latex_big)
    
    latex_filename = os.path.join(SAVING_PATH, "Recap_small.tex")
    with open(latex_filename, "w") as f:
        f.write(latex_small)

    print("LaTeX table saved")

    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns="Row Std")


    df_combined = pd.concat(dfs, axis=0)

    ### Aggregate results Recall calculate across seeds.
    aggregation_df = pd.DataFrame()
    for column in df_combined.columns: 
        aggregation_df[f'Mean {column}'] = df_combined[column].groupby(df_combined.index).mean()  
    aggregation_df['IoU Threshold'] = aggregation_df.index

    for idx, column in enumerate(df_combined.columns):
        color = plt.cm.tab10(idx)
        plt.plot(aggregation_df['IoU Threshold'], aggregation_df[f'Mean {column}'], 
                label=f'{column}', marker='o', linestyle='-', color=color)
        
    plt.xlabel('IoU Threshold')
    plt.ylabel('Aggregate(Recall)')
    plt.title(f'Aggregate(Recall) in {format_group_name(GROUP_FOLDER)} experiments')
    plt.legend()
    plot_filename = os.path.join(SAVING_PATH, "Mean_recall_seeds")
    plt.savefig(
        plot_filename,
        dpi=600,                     # High resolution for print
        bbox_inches='tight',        # Removes all unnecessary whitespace
        pad_inches=0.1              # Minimal padding, keeps axes safe
    )

    plt.close()

    ### Aggregate recall5095 calculate across seeds 
    column_means = df_combined.mean()
    column_means = column_means.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    ax = column_means.plot(kind='bar', color='orange')
    plt.title(f'Aggregate(AvgRecall) across seeds in {format_group_name(GROUP_FOLDER)} experiments')
    plt.xlabel('')
    plt.ylabel('Aggregate(AvgRecall)')
    plt.xticks(rotation=45)

    for i, v in enumerate(column_means):
        ax.text(i, v , f'{v:.3f}', ha='center', va='bottom', fontsize=10, color= "black")

    plt.tight_layout()
    bar_plot_filename = os.path.join(SAVING_PATH, "Mean_Recall_5095_seeds")
   
    plt.savefig(
        bar_plot_filename,
        dpi=600,                     # High resolution for print
        bbox_inches='tight',        # Removes all unnecessary whitespace
        pad_inches=0.1              # Minimal padding, keeps axes safe
    )

    plt.close()


    