from Filtering_pulp import * 
from subdolder_creation import * 
import pandas as pd 
import os 
import seaborn as sns 

def validate_split(train_set, test_set, group):
    """Check if there is any intersection between train and test splits."""
    try:
        intersection = set(train_set['filename'].unique()).intersection(set(test_set['filename'].unique()))
        if intersection:
            raise ValueError(f"Error: {len(intersection)} overlapping files found in {group} split.")
    except Exception as e:
        print(f"Validation failed for {group}: {e}")
        return False
    return True

def run_splits_and_save():
    ### Note : As I forgot to put a seed on the creation of the splits, I remove the saving to df in order to be consistent with the results that are inside the master thesis. 
    # This randomness have no impact on test set, validation set but can have impact on train splits.
    ###
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_test_path = os.path.join(project_root, "Train_test_dataset")
    os.makedirs(train_test_path, exist_ok=True)

    ### ANNOTATE DATASET
    df = pd.read_csv('filtered_dataset_x1y1x2y2.csv')

    ### Create test, val, train splits
    test_set, validation_set, train_split = create_test_train_split("", df)
    train_split.to_csv(os.path.join(train_test_path, "train_split.csv"), index=False)
    validation_set.to_csv(os.path.join(train_test_path, "validation_set.csv"), index=False)
    test_set.to_csv(os.path.join(train_test_path, "test_set.csv"), index=False)

    if not validate_split(train_split, test_set, "Initial Train-Test"):
        print("Overlap detected in Initial Train-Test split!")
    if not validate_split(train_split, validation_set, "Initial Train-Validation"):
        print("Overlap detected in Initial Train-Validation split!")

    train_path = os.path.join(train_test_path, "train_split.csv")
    df = pd.read_csv(train_path)

    excluded_dict_race = {
        "global_ethnicity": [], 
        "loeo_indian": ["Indian"],
        "loeo_black": ["Black"],
        "loeo_asian": ["Asian"]
    }

    excluded_dict_race_black_indian = {
        "loeo_indianBlack" : ["Indian", "Black"]
    }

    excluded_dict_gender = {
        "global_sex": [], 
        "loso_male": ["Male"],
        "loso_female": ["Female"]
    }

    excluded_dict_race_upsample = {
        "global_ethnicityUpsample": []
    }

    try:
        ### RACE SPLIT (TEST SET 0.2)
        group = 'Race'
        
        train_splits = create_loro_split(df, excluded_dict_race, group)
        for split, d in train_splits.items():
            saving_path = os.path.join(train_test_path,f'train_set_{split}.csv')
            num_boxes = len(d)
            num_images = d['filename'].nunique()
            if not validate_split(d, test_set, f"Race Split {split}"):
                print(f"Overlap detected in Race split {split}")
            #d.to_csv(saving_path, index=False)

        print(num_boxes, num_images)

        ### RACE SPLIT (TEST SET 0.2) Black+Indian exclusion no upsampling
        train_splits = create_loro_split(df, excluded_dict_race_black_indian, group, upsample_minority=False, target_bbox=num_boxes, target_filename=num_images)
        for split, d in train_splits.items():
            saving_path = os.path.join(train_test_path, f'train_set_{split}.csv')
            if not validate_split(d, test_set, f"Gender Split {split}"):
                print(f"Overlap detected in Gender split {split}")
            #d.to_csv(saving_path, index=False)

        ### Gender split (TEST SET 0.15)
        group = 'Gender'
        train_splits = create_loro_split(df, excluded_dict_gender, group)
        for split, d in train_splits.items():
            saving_path = os.path.join(train_test_path, f'train_set_{split}.csv')
            if not validate_split(d, test_set, f"Gender Split {split}"):
                print(f"Overlap detected in Gender split {split}")
            #d.to_csv(saving_path, index=False)
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    run_splits_and_save()
    



