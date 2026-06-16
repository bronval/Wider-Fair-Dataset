import os
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Training.utility import *
from Training.dataset import *


def process_dataset(model, data_loader, save_dir, small_grid_anchors, medium_grid_anchors, large_grid_anchors, config):
    """
    Runs inference on a dataset and saves detection results to text files.
    """
    device = next(model.parameters()).device
    confidence_threshold = config['NMS']['confidence_threshold']
    iou_threshold = config['NMS']['iou_threshold']
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for _, (images, _, img_paths) in enumerate(tqdm(data_loader, desc=f"Generating Detection Files {save_dir}")):
            images = images.to(device)
            predictions = model(images)

            pred_small, conf_small = decode_predictions(predictions[0], small_grid_anchors, config["grid_sizes"][0])
            pred_medium, conf_medium = decode_predictions(predictions[1], medium_grid_anchors, config["grid_sizes"][1])
            pred_large, conf_large = decode_predictions(predictions[2], large_grid_anchors, config["grid_sizes"][2])

            all_predictions = [pred_small, pred_medium, pred_large]
            all_confidences = [conf_small, conf_medium, conf_large]

            filtered_boxes, filtered_scores = apply_nms(all_predictions, all_confidences, iou_threshold, confidence_threshold)
            for img_idx, img_path in enumerate(img_paths):
                image_filename = os.path.basename(img_path)
                txt_filename = os.path.join(save_dir, os.path.splitext(image_filename)[0] + ".txt")

                with open(txt_filename, "w") as f:
                    for box, score in zip(filtered_boxes[img_idx], filtered_scores[img_idx]):
                        class_name = "face"  
                        x1, y1, x2, y2 = box
                        x1 = max(0, min(x1, 1))
                        y1 = max(0, min(y1, 1))
                        x2 = max(0, min(x2, 1))
                        y2 = max(0, min(y2, 1))

                        f.write(f"{class_name} {score:.3f} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}\n")

def create_detection_files():
    """
    Generates detection files for validation and test datasets.
    """

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    RES_FOLDER = "Results"
    GROUP_FOLDER = config["saving_folder"]
    DETECTION_FOLDER_TEST = "test_detections"
    DETECTION_FOLDER_VAL = "val_detections"
    model_name = config["saving_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    model = load_model_from_checkpoints(config).to(device).eval()
    
    anch = torch.tensor(config["yolo"]["anchors"], dtype=torch.float32) / config["img_h"]
    small_grid_anchors, medium_grid_anchors, large_grid_anchors = anch[6:9], anch[3:6], anch[0:3]
    
    base_save_path = os.path.join(RES_FOLDER, GROUP_FOLDER, model_name)
    
    for phase, folder in zip(['val', 'test'], [DETECTION_FOLDER_VAL, DETECTION_FOLDER_TEST]):
        dataset = ImageFolder(config, phase=phase)
        data_loader = DataLoader(
            dataset,
            batch_sampler=BatchSampler(RandomSampler(dataset), batch_size=config["batch_size"], drop_last=False),
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        process_dataset(model, data_loader, os.path.join(base_save_path, folder), 
                        small_grid_anchors, medium_grid_anchors, large_grid_anchors, config)
        print(f"Detection files saved in '{folder}'")

if __name__ == "__main__":
    create_detection_files()
