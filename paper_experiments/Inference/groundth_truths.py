import os
import yaml
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import sys 
import os 
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Training.dataset import ImageFolder
from Training.utility import convert_center_to_corners

def create_ground_truths_files(split): 
    """
    ### DOCSTRINGS DONE BY CHATGPT
    Generates ground truth files for the test dataset by extracting bounding box annotations 
    and metadata for each image. For each image in the test set, a corresponding text file is 
    created, containing the ground truth bounding boxes, class labels, and additional information 
    like occlusion and relative area.

    The function processes each image in the test dataset, extracts the bounding boxes and 
    corresponding metadata, and saves the results in text files with the following format:
        {class_name_ethnicity} {class_name_gender} {x1} {y1} {x2} {y2} {} {}

    Args:
        None

    Returns:
        None

    Saves:
        - Ground truth results in text files inside a specified directory under `Results`.
          Each file contains ground truth data in the format:
            {class_name_ethnicity} {class_name_gender} {x1} {y1} {x2} {y2} {placeholder} {placeholder}
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    if split == 'test':
        GROUND_TRUTH_FOLDER = "test_groundtruths"
        df = pd.read_csv(config['dataset']['test_set'])
    else : 
        GROUND_TRUTH_FOLDER = "val_groundtruths"
        df = pd.read_csv(config['dataset']['val_set'])

    model_name = config["saving_name"]
    SAVE_DIR = os.path.join(RES_FOLDER, GROUP_FOLDER,model_name,GROUND_TRUTH_FOLDER)
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR) ### SAFETY 
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = ImageFolder(config, phase=split)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=config["batch_size"], drop_last=False)
    val_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)

    for batch_idx, (images, targets, img_paths) in enumerate(tqdm(val_loader, desc="Generating Ground Truth Files")):
        for (img_idx, target_boxes) in enumerate(targets):
            image_filename = os.path.basename(img_paths[img_idx])
            txt_filename = os.path.join(SAVE_DIR, os.path.splitext(image_filename)[0] + ".txt")

            with open(txt_filename, "w") as f:
                for box in target_boxes:
                    class_name_ethnicity = box[-1]
                    class_name_gender = box[-2]  
                    x1, y1, x2, y2 = convert_center_to_corners(box)
                    target_rows = df[df['filename'] == img_paths[img_idx]]
                    for _, row in target_rows.iterrows():
                        x1_scaled = x1 * row['original_width']
                        y1_scaled = y1 * row['original_height']
                        x2_scaled = x2 * row['original_width']
                        y2_scaled = y2 * row['original_height']
                        
                        
                        tolerance = 1e-3
                        if (abs(int(x1_scaled) - row['x1']) < tolerance and 
                            abs(int(y1_scaled) - row['y1']) < tolerance and 
                            abs(int(x2_scaled) - row['x2']) < tolerance and 
                            abs(int(y2_scaled) - row['y2']) < tolerance):

                            occlusion = row['occlusion']
                            relative_area = row['relative_area']

                    f.write(f"{class_name_ethnicity} {class_name_gender} {x1} {y1} {x2} {y2} {_} {_}\n")
