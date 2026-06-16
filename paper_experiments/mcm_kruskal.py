import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  
import os 
from scipy.stats import beta, kstest

def run_kruskal_mcm(group_column='Race', 
                    csv_path='Train_test_dataset/filtered_dataset_x1y1x2y2.csv',
                    limit_per_group=600, 
                    n_iterations=1000, 
                    latex_output_dir=os.path.join('viz_thesis', 'Experimental','MCM')):
    
    if group_column == 'Gender': 
        limit_per_group= 1200
    
    ### Generate output filename based on group_column
    os.makedirs(latex_output_dir, exist_ok=True)

    ### Load and filter dataset
    df = pd.read_csv(csv_path)
    excluded_classes = ['Undetermined', 'Other', 'Middle Eastern']
    df = df.groupby('filename').filter(
        lambda group: not (
            (group['Gender'] == 'Undetermined').any() or 
            (group['Race'].isin(excluded_classes)).any() or 
            (group['Valid'] == False).any()
        )
    )
    
    df['race_gender'] = df['Race'] + "_" + df['Gender']
    filename_groups = df.groupby('filename')
    all_filenames = list(filename_groups.groups.keys())

    kruskal_pvalues = []

    for _ in tqdm(range(n_iterations)):
        #### Greedy sampling
        group_counts = defaultdict(int)
        selected_filenames = set()
        np.random.shuffle(all_filenames)

        for filename in all_filenames:
            group = filename_groups.get_group(filename)
            current_counts = group[group_column].value_counts()
            if all(group_counts[g] + c <= limit_per_group for g, c in current_counts.items()):
                selected_filenames.add(filename)
                for g, c in current_counts.items():
                    group_counts[g] += c
            if all(c >= limit_per_group for c in group_counts.values()):
                break

        balanced_df = df[df['filename'].isin(selected_filenames)]
        grouped_data = balanced_df.groupby(group_column)['relative_area']

        if len(grouped_data) > 1:
            samples = [group.values for _, group in grouped_data]
            _, pval = kruskal(*samples)
            kruskal_pvalues.append(pval)
        else:
            kruskal_pvalues.append(np.nan)
    ### Compute median and proportion of significant
    valid_pvals = [p for p in kruskal_pvalues if not np.isnan(p)]
    proportion_significant = sum(p < 0.05 for p in valid_pvals) / len(valid_pvals)
    median_pval = np.median(valid_pvals)

    ### Create LaTeX table
    summary_df = pd.DataFrame({
        'Group Column': [group_column],
        'Iterations': [n_iterations],
        'Median p-value': [round(median_pval, 4)],
        'Proportion p < 0.05': [round(proportion_significant, 4)]
    })

    latex = summary_df.to_latex(index=False, float_format="%.4f", caption=None, label=None)
    output_filename = f"kruskal_summary_{group_column.lower()}.tex" 
    out_path = os.path.join(latex_output_dir, output_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  
    with open(out_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX summary saved to: {out_path}")

def fit_and_plot_beta_distributions(csv_path=r'Train_test_dataset\test_set.csv', group_column='Race', output_plot='viz_thesis/Experimental/beta_distributions.png'):
    df = pd.read_csv(csv_path)
    grouped_data = df.groupby(group_column)['relative_area']
    beta_params = {}

    ### Fit Beta distribution
    for group, values in grouped_data:
        a, b, loc, scale = beta.fit(values, floc=0, fscale=1)
        beta_params[group] = (a, b)
        print(f"Beta params for '{group}': a={a:.3f}, b={b:.3f}")

    ### KS test for goodness of fit
    for group, values in grouped_data:
        values = values[(values > 0) & (values < 1)]
        if len(values) == 0 or group not in beta_params:
            continue
        a, b = beta_params[group]
        cdf = lambda x: beta.cdf(x, a, b, loc=0, scale=1)
        stat, p_value = kstest(values, cdf)
        print(f" {group}: stat={stat:.4f}, p-value={p_value:.4f}")


    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    for group, (a, b) in beta_params.items():
        y = beta.pdf(x, a, b, loc=0, scale=1)
        plt.plot(x, y, label=f"{group} (a={a:.2f}, b={b:.2f})")

    plt.title(f'Beta Distributions by {group_column}', fontsize=16)
    plt.xlabel('Relative area', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.xlim(0, 0.25)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=600)
    plt.show()
    print(f" Saved to: {output_plot}")

def kde_iou_distribution_by_race(path):
    # Load data
    df = pd.read_csv(path)

    # Mapping race codes to labels
    race_mapping = {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian'
    }
    df['Ethnicity'] = df['Race'].map(race_mapping)

    # Filter invalid IoU values (optional, just in case)
    df = df[(df['Best_Iou'] >= 0) & (df['Best_Iou'] <= 1)]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="Best_Iou", hue="Ethnicity", common_norm=False, fill=False, alpha=0.4, linewidth=2)

    plt.title("")
    plt.xlabel("IoU")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig("viz_thesis/Experimental/KDE_IoU_by_Ethnicity.png", dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    lst = ['Gender', 'Race']
    for l in lst : 
        run_kruskal_mcm(group_column=l)
    fit_and_plot_beta_distributions()
    kde_iou_distribution_by_race(r"Results\global_ethnicity\global_ethnicity_0\metrics_df_test\res_IOU.csv")
