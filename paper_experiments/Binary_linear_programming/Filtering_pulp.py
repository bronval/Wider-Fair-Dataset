import pulp
import numpy as np
import pandas as pd 
import os 
from subdolder_creation import * 

def create_test_train_split(group_interest, df, test_ratio=0.15, validation_ratio=0.05, excluded_classes=['Undetermined', 'Other', 'Middle Eastern']): 
    """
    Aim : create a test_set and a train_set. In the test set we want to balance in term of group but also in term of difficulty of prediction (distribution of categorical variable across group and distribution of relative area across group.) 
    Input : 
        - group_interest (str) : the group that we want to create the test set for ('Gender' or 'Race')
        - df (dataframe) : the dataframe annotated
        - test_ratio (float) : proportion of bounding boxes to put in the test set
        - validation_ratio (float) : proportion of bounding boxes to put in the validation set
        - excluded_classes (list(str)) : group that we exclude from the analysis. 
    Return : 
        - train_set (dataframe) 
        - test_set  (dataframe)
    """
    excluded_dict = {
        "test": None,
        "validation": None
    }
    splits = list(excluded_dict.keys())

    ### Remove images where any row in a group_interest has an excluded gender or group_interest
    df_filtered = df.groupby('filename').filter(
        lambda group: not ((group['Gender'] == 'Undetermined').any() or 
                           (group['Race'].isin(excluded_classes)).any() or 
                           (group['Valid'] == False).any())
    )
    #create_subset_images(df_filtered)
    ### Create a new column combining Gender and Race values
    df_filtered['Gender_Race'] = df_filtered['Gender'] + "_" + df_filtered['Race']
    group_interest = 'Gender_Race'
    class_ratio = 1 / len(set(df_filtered[group_interest].unique()))

    targets_boxes_test = int(len(df_filtered) * test_ratio * class_ratio)
    targets_boxes_validation = int(len(df_filtered) * validation_ratio * class_ratio)

    ### Fix a target for each categorical variable on the test and validation sets
    categorical_attribute = ['expression', 'illumination', 'occlusion', 'pose', 'blur']
    target_categorical_value = {split :{cat: 0 for cat in categorical_attribute} for split in splits}
    for cat in categorical_attribute:
        min_count = df_filtered[df_filtered[cat] != 0].groupby(group_interest)[cat].value_counts().min()  
        target_categorical_value['test'][cat] = int((min_count / 2) * (test_ratio / (validation_ratio + test_ratio)))
        target_categorical_value['validation'][cat] = int((min_count / 2) * (validation_ratio / (validation_ratio + test_ratio)))

    ### Fix a target on relative area on the test and validation sets
    bins = np.linspace(df_filtered['relative_area'].min(), 0.4, 16)
    bin_labels = [i for i in range(15)]
    df_filtered['area_bin'] = pd.cut(df_filtered['relative_area'], bins=bins, labels=bin_labels, include_lowest=True)
    area_bin_proportions = df_filtered['area_bin'].value_counts(normalize=True).sort_index()
    
    ### ATTRIBUTES
    unique_group = df_filtered[group_interest].unique()
    unique_filename = df_filtered["filename"].unique()
    
    ### PRECOMPUTATION 
    box_count_per_filename_group = (
        df_filtered.groupby(['filename', group_interest])
        .size()
        .reset_index(name="count")
        .set_index(['filename', group_interest])
        ["count"]
        .to_dict()
    )

    categorical_count_per_filename_group = {}
    for cat in categorical_attribute:
        filtered_df = df_filtered[df_filtered[cat] != 0]
        counts = (
            filtered_df.groupby(['filename', group_interest])
            .size()
            .reset_index(name="count")
            .set_index(['filename', group_interest])
            ["count"]
            .to_dict()
        )
        
        for (filename, group), count in counts.items():
            categorical_count_per_filename_group[(filename, group, cat)] = count

    bin_counts_per_filename_group = {}
    for label in bin_labels:
        filtered_df = df_filtered[df_filtered['area_bin'] == label]
        counts = (
            filtered_df.groupby(["filename", group_interest])
            .size()
            .reset_index(name="count")
            .set_index(["filename", group_interest])
            ["count"]
            .to_dict()
        )
        for (filename, group), count in counts.items():
            bin_counts_per_filename_group[(filename, group, label)] = count
        
    ### MAX PROBLEM 
    prob = pulp.LpProblem("Test_Set_Balancing", pulp.LpMaximize)

    unique_group = df_filtered[group_interest].unique()

    x = {split: pulp.LpVariable.dicts(f"Select_{split}", unique_filename, cat="Binary") for split in splits}

    ### SLACK VALUES FOR TOLERANCE
    slack_test = 0.01
    slack_validation = 0.05
    slack_bin_test = 0.02
    slack_bin_validation = 0.02

    prob += pulp.lpSum([x[split][filename] for split in splits for filename in unique_filename])

    ### AUXILIARY VARIABLE 
    box_counts = {split : 
    {group_interest: pulp.LpVariable(f"BoxCount_{group_interest}_{split}", lowBound=0, cat="Integer") for group_interest in unique_group}
    for split in splits}

    categorical_variable_counts = {split :{
        cat: {group_interest: pulp.LpVariable(f"{split}_{cat}_{group_interest}", lowBound=0, cat="Integer") for group_interest in unique_group}
        for cat in categorical_attribute
        }
        for split in splits}


    bin_variable_counts = {split :{
        label :{group_interest: pulp.LpVariable(f"{split}_{group_interest}_{label}", lowBound=0, cat="Integer") for group_interest in unique_group}
        for label in bin_labels}
        for split in splits}

    ### CONSTRAINT 1: Definition of box count in function of x 
    for split in splits:
        for group in unique_group:
            prob += box_counts[split][group] == pulp.lpSum(
                box_count_per_filename_group.get((filename, group), 0) * x[split][filename] for filename in unique_filename
            )
            
    ### CONSTRAINT 2: Limit of box count for each group_interest 
    for split in splits:
        for group in unique_group:
            if split == 'test':
                prob += box_counts[split][group] <= targets_boxes_test * (1 + slack_test  )
                prob += box_counts[split][group] >= targets_boxes_test * (1 - slack_test  )
            elif split == 'validation':
                prob += box_counts[split][group] <= targets_boxes_validation * (1 + slack_validation)
                prob += box_counts[split][group] >= targets_boxes_validation * (1 - slack_validation)

    ### CONSTRAINT 3: Definition of categorical count in function of x 
    for split in splits:
        for cat in categorical_attribute:
            for group in unique_group:
                prob += categorical_variable_counts[split][cat][group] == pulp.lpSum(
    categorical_count_per_filename_group.get((filename, group, cat), 0) * x[split][filename] for filename in unique_filename
)

    ### CONSTRAINT 4 : Limit of categorical count for each group_interest 
    for split in splits:
        for cat in categorical_attribute: 
            for group in unique_group:
                prob += categorical_variable_counts[split][cat][group] <= target_categorical_value[split][cat] 
                prob += target_categorical_value[split][cat]  <= categorical_variable_counts[split][cat][group]

    ### CONSTRAINT 5: Definition of bin count in function of x
    for split in splits:
        for label in bin_labels:
            for group in unique_group:
                prob += bin_variable_counts[split][label][group] == pulp.lpSum(
                    bin_counts_per_filename_group.get((filename, group, label), 0) * x[split][filename] for filename in unique_filename
                )
    ### CONSTRAINT 6: Limit of each bin count
    for split in splits:
        for label in bin_labels:
            for group in unique_group:
                if split == 'test':
                    target_bin_count = int(area_bin_proportions[label] * targets_boxes_test)
                    prob += bin_variable_counts[split][label][group] <= target_bin_count * (1 + slack_bin_test)
                    prob += bin_variable_counts[split][label][group] >= target_bin_count * (1 - slack_bin_test)
                if split == 'validation' : 
                    target_bin_count = int(area_bin_proportions[label] * targets_boxes_validation)
                    prob += bin_variable_counts[split][label][group] <= target_bin_count * (1 + slack_bin_validation)
                    prob += bin_variable_counts[split][label][group] >= target_bin_count * (1 - slack_bin_validation)

    ### CONSTRAINT 7: Mutually exclusive file assignment
    for filename in unique_filename:
        prob += pulp.lpSum([x[split][filename] for split in splits]) <= 1
    
    prob.solve()

    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise ValueError("Optimization problem is infeasible! Please check constraints.")
    
    ### SOLUTION 
    selected_filenames = {split: [filename for filename in unique_filename if pulp.value(x[split][filename]) == 1] for split in splits}

    test_split = df_filtered[df_filtered["filename"].isin(selected_filenames["test"])]
    validation_split = df_filtered[df_filtered["filename"].isin(selected_filenames["validation"])]
    train_split = df_filtered[~df_filtered["filename"].isin(selected_filenames["test"] + selected_filenames["validation"])]

    return test_split, validation_split, train_split

def create_loro_split(train_set, excluded_dict, group_interest, upsample_minority=False, target_bbox=None , target_filename=None  ) : 
    """
    Aim : create subsplit. In the subsplit we want to ensure that each subsplit have the same number of bounding boxes and filenames (images)
    Input : 
        - train_set (dataframe) : train set resulting from create_test_train_split()
        - excluded_dict (dict (str: str)) : dict with keys equals to the name of subsplit and values equals to the group to excluded from the split
        - group_interest (str) : the group that we want to create the test set for ('Gender' or 'Race')
    Return : 
        - train_splits (dict (str: dataframe)) : dict with keys equals to the name of subsplit and values equals to the dataframe corresponding 
    """

    ### MAX
    prob = pulp.LpProblem("Train_Set_Balancing", pulp.LpMaximize)

    ### PRECOMPUTATION
    box_count_per_filename = (
        train_set.groupby(["filename"])
        .size()
        .reset_index(name="count")
        .set_index(["filename"])
        ["count"]
        .to_dict()
    )

    box_count_per_filename_group = (
        train_set.groupby(['filename', group_interest])
        .size()
        .reset_index(name="count")
        .set_index(['filename', group_interest])
        ["count"]
        .to_dict()
    )

    group_per_filename = train_set.groupby("filename")[group_interest].unique().to_dict()
 
    ### ATTRIBUTE
    unique_group = train_set[group_interest].unique()
    splits = list(excluded_dict.keys())
    unique_filenames = train_set["filename"].unique()

    ### DECISION VARIABLE 
    if upsample_minority == True :
        x = {split: pulp.LpVariable.dicts(f"Select_{split}", unique_filenames, cat="Integer", lowBound=0, upBound=10) for split in splits}
        z = {split: {filename: pulp.LpVariable(f"IsSelected_{split}_{filename}", cat="Binary") for filename in unique_filenames} for split in splits}
        
    else : 
        x = {split: pulp.LpVariable.dicts(f"Select_{split}", unique_filenames, cat="Binary") for split in splits}

    ### AUXILIARY VARIABLE
    box_counts = {group_interest: pulp.LpVariable(f"BoxCount_{group_interest}", lowBound=0, cat="Integer") for group_interest in unique_group}
    ### CONSTRAINT 1 : each loro/logo split need to have the same number of bounding box
    for split1 in x:
        for split2 in x:
            if split1 != split2:
                prob += pulp.lpSum([x[split1][filename] * box_count_per_filename.get((filename),0) for filename in unique_filenames]) == \
                        pulp.lpSum([x[split2][filename] * box_count_per_filename.get((filename),0) for filename in unique_filenames])
                
                ### CONSTRAINT 2 : each loro/logo split need to have the same number of image
                prob += pulp.lpSum([x[split1][filename]  for filename in unique_filenames]) == \
                        pulp.lpSum([x[split2][filename]  for filename in unique_filenames]) 
    
    ### CONSTRAINT 2 : in each loro/logo split we exclude any image that contain one exclude group 
    for split, excluded_races in excluded_dict.items():
        for filename in unique_filenames:
            groups = group_per_filename.get(filename, [])
            if any(excluded in groups for excluded in excluded_races):
                prob += x[split][filename] == 0 
    
    ### SPECIAL CONSTRAINT: Balance "global_gender" in terms of Male/Female if group_interest == 'Gender'

    if group_interest == "Gender":

        ### CONSTRAINT 3: Definition of box count in function of x 
        for group in unique_group:
            prob += box_counts[group] == pulp.lpSum(
                box_count_per_filename_group.get((filename, group), 0) * x["global_sex"][filename] for filename in unique_filenames
            )

        ### CONSTRAINT 4 : Impose equality of count
        for group1 in unique_group:
            for group2 in unique_group:
                if group1 != group2:
                    prob += box_counts[group2] == box_counts[group1]
    
    epsilon = 0.1

    if upsample_minority == True:

        ### CONSTRAINT 3: Definition of box count in function of x 
        for split in splits :
            for group in unique_group:
                prob += box_counts[group] == pulp.lpSum(
                    box_count_per_filename_group.get((filename, group), 0) * x[split][filename] for filename in unique_filenames
                )

            ### CONSTRAINT 4 : Impose equality of count
            for group1 in unique_group:
                for group2 in unique_group:
                    if group1 != group2:
                        prob += box_counts[group2] >= (1 - epsilon) * box_counts[group1]
                        prob += box_counts[group2] <= (1 + epsilon) * box_counts[group1]
        
        for split in splits:
            for filename in unique_filenames:
                prob += x[split][filename] >= z[split][filename]
                prob += x[split][filename] <= 10 * z[split][filename]  

    if target_filename and upsample_minority == True:
        # Constraint for the number of filenames (images)
        for split in splits:
            prob += pulp.lpSum(x[split][filename] for filename in unique_filenames) >= target_filename * (1 - epsilon)
            prob += pulp.lpSum(x[split][filename] for filename in unique_filenames) <= target_filename 

        # Constraint for the number of bounding boxes
        for split in splits:
            prob += pulp.lpSum([x[split][filename] * box_count_per_filename.get((filename), 0) for filename in unique_filenames]) >= target_bbox * (1 - epsilon)
            prob += pulp.lpSum([x[split][filename] * box_count_per_filename.get((filename), 0) for filename in unique_filenames]) <= target_bbox 
        
    elif target_filename and upsample_minority == False : 
        for split in splits:
            prob += pulp.lpSum(x[split][filename] for filename in unique_filenames) >= target_filename * (1 - 0.1)
            prob += pulp.lpSum(x[split][filename] for filename in unique_filenames) <= target_filename 

        # Constraint for the number of bounding boxes
        for split in splits:
            prob += pulp.lpSum([x[split][filename] * box_count_per_filename.get((filename), 0) for filename in unique_filenames]) >= target_bbox * (1 - 0.1)
            prob += pulp.lpSum([x[split][filename] * box_count_per_filename.get((filename), 0) for filename in unique_filenames]) <= target_bbox 


    if upsample_minority == True : 
       prob += pulp.lpSum(z[split][filename] for split in splits for filename in unique_filenames) ### maximize the diversity of selection 
    else : 
        prob += pulp.lpSum([x[split][filename] for filename in unique_filenames] for split in splits)

    ### SOLVER 
    prob.solve()
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise ValueError("Optimization problem is infeasible! Please check constraints.")
    
    ### SOLUTION 
    train_splits = {}
    for split in splits:
        upsample_counts = {
            filename: int(pulp.value(x[split][filename]))
            for filename in unique_filenames
            if pulp.value(x[split][filename]) != 0
        }
        
        split_train_set = train_set[train_set["filename"].isin(upsample_counts.keys())].copy()
        
        split_train_set["upsample"] = split_train_set["filename"].map(upsample_counts)
        
        train_splits[split] = split_train_set

    return train_splits

    