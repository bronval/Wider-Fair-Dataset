import os
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

def analyze_significant_differences_latex(
    results_dir="Results",
    latex_dir=os.path.join("viz_thesis", "Experimental", "Cross_MWU_latex"),
    indicator="Best_Iou"
):
    os.makedirs(latex_dir, exist_ok=True)

    def effective_sample_size(n1, n2):
        return (n1 * n2) / (n1 + n2) if (n1 + n2) > 0 else 0

    def save_latex_table(df, filename):
        if 'Model' in df.columns:
            df = df.drop(columns=['Model'])
        if 'Group_Comparison' in df.columns:
            df = df.rename(columns={'Group_Comparison': 'Groups'})

        latex = df.to_latex(index=False, float_format="%.2f", caption=None, label=None)
        lines = latex.splitlines()
        start_idx = next(i for i, line in enumerate(lines) if line.startswith(r'\begin{tabular}'))
        end_idx = next(i for i, line in enumerate(lines) if line.startswith(r'\end{tabular}'))
        tabular_lines = lines[start_idx:end_idx+1]
        tabular_str = "\n".join(tabular_lines)

        with open(os.path.join(latex_dir, filename), "w") as f:
            f.write(tabular_str)

    def analyze_groups(df, group_col, group_labels):
        results = []
        groups = {label: df[df[group_col] == code][indicator] for label, code in group_labels.items()}
        valid_groups = {g: vals for g, vals in groups.items() if len(vals) > 0}

        if len(valid_groups) < 2:
            return pd.DataFrame()

        group_names = list(valid_groups.keys())
        group_values = list(valid_groups.values())

        if len(group_names) > 2:
            stat, p = kruskal(*group_values)
            if p >= 0.05:
                return pd.DataFrame()

        pairs = combinations(valid_groups.keys(), 2)
        for g1, g2 in pairs:
            d1, d2 = valid_groups[g1], valid_groups[g2]
            stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
            ess = effective_sample_size(len(d1), len(d2))
            results.append({
                "Group_1": g1,
                "Group_2": g2,
                "p_value": pval,
                "ESS": ess,
                "Significant": pval < 0.05
            })
        return pd.DataFrame(results)

    model_run_results = defaultdict(lambda: defaultdict(list))

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for run_name in os.listdir(folder_path):
            model_path = os.path.join(folder_path, run_name, "metrics_df_test", "res_IOU.csv")
            if not os.path.exists(model_path):
                continue

            df = pd.read_csv(model_path)

            if "sex" in folder or "loso" in folder:
                group_labels = {"Male": 0.0, "Female": 1.0}
                group_col = 'Gender'
            else:
                group_labels = {
                    "White": 0.0,
                    "Black": 1.0,
                    "Asian": 2.0,
                    "Indian": 3.0
                }
                group_col = 'Race'

            res_df = analyze_groups(df, group_col, group_labels)
            if res_df.empty:
                continue

            for _, row in res_df.iterrows():
                key = tuple(sorted([row["Group_1"], row["Group_2"]]))
                model_run_results[folder][key].append(row)

    for model_name, group_data in model_run_results.items():
        merged_rows = []
        for group_pair, results_list in group_data.items():
            if len(results_list) == 3 and all(r["Significant"] for r in results_list):
                p_values = tuple(round(r["p_value"], 2) for r in results_list)
                merged_rows.append({
                    "Model": model_name,
                    "Group_Comparison": f"{group_pair[0]} vs {group_pair[1]}",
                    "p-values (Run 1, 2, 3)": f"({p_values[0]}, {p_values[1]}, {p_values[2]})"
                })

        if merged_rows:
            df_table = pd.DataFrame(merged_rows)
            filename = f"{model_name}_significant_differences.tex"
            save_latex_table(df_table, filename)
            print(f"Saved merged LaTeX table for model: {model_name} -> {filename}")


if __name__ == "__main__":
    analyze_significant_differences_latex()


