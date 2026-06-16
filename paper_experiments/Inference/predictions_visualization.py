from tqdm import tqdm  
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Training.dataset import *
from Training.utility import *


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def create_predictions_images() : 
    """
    ### DOCSTRINGS DONE BY CHATGPT
    This function processes a dataset using a trained model to generate predictions for 
    each image in the dataset. For each image, the model's predictions (bounding boxes and class 
    confidences) are visualized alongside the ground truth bounding boxes. The images with overlaid 
    predictions and ground truth are saved for further evaluation.

    The function performs the following steps:
    1. Loads a pre-trained model from a checkpoint file specified in the configuration (`config.yaml`).
    2. Loads the dataset and prepares it for processing by batching and applying the required transformations.
    3. For each batch of images:
        - The model generates predictions for small, medium, and large grid sizes.
        - Non-Maximum Suppression (NMS) is applied to filter out low-confidence predictions and 
          duplicate boxes.
        - The predictions and ground truth boxes are visualized and saved as images.
    
    Args:
        None

    Returns:
        None

    Saves:
        - Saves visualized images that show the model's predictions overlaid with the ground truth 
          bounding boxes. Each image is saved in the directory specified by the `filename`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = load_model_from_checkpoints(config)
    model.eval()

    validation_dataset = ImageFolder(config, phase='test' 
        )
    
    val_sampler = RandomSampler(validation_dataset)

    val_batch_sampler = BatchSampler(
        val_sampler,
        batch_size=config["batch_size"],
        drop_last=False  
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=validation_dataset.collate_fn  
    )
    
    anch = config["yolo"]["anchors"]
    default = torch.tensor(anch, dtype=torch.float32) / config["img_h"]
    
    small_grid_anchors = default[6:9].clone().detach().float()
    medium_grid_anchors = default[3:6].clone().detach().float()
    large_grid_anchors = default[0:3].clone().detach().float()

    
    with torch.no_grad():
        ### Save each image with prediction and ground
        for batch_idx, (images, targets, filename) in enumerate(tqdm(val_loader)):


            predictions = model(images)
            pred_small, conf_small = decode_predictions(predictions[0], small_grid_anchors, config["grid_sizes"][0])
            pred_medium, conf_medium = decode_predictions(predictions[1], medium_grid_anchors, config["grid_sizes"][1])
            pred_big, conf_big = decode_predictions(predictions[2], large_grid_anchors, config["grid_sizes"][2])

            predictions = [pred_small, pred_medium, pred_big]
            confidences = [conf_small, conf_medium, conf_big]

            filtered_boxes, filtered_scores = apply_nms(predictions, confidences, 0.4,config['Recall']['confidence_threshold'])
            
            visualize_batch_predictions_nms(images, filtered_boxes, filtered_scores, (config["img_h"], config["img_w"]),targets, batch_idx=batch_idx, filename = filename)

    