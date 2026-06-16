from Inference import * 
import os
from ruamel.yaml import YAML
import argparse

def process_checkpoints(mode, target_folder=None, target_checkpoint=None):
    CHECKPOINTS_DIR = "Checkpoints"
    DATASET_DIR = "Train_test_dataset"

    folders_to_process = (
        [target_folder] if target_folder else os.listdir(CHECKPOINTS_DIR)
    )

    for checkpoints_folder in folders_to_process:

        folder_path = os.path.join(CHECKPOINTS_DIR, checkpoints_folder)

        checkpoint_files = (
            [target_checkpoint] if target_checkpoint 
            else os.listdir(folder_path)
        )

        for checkpoint_model in checkpoint_files:
            if not checkpoint_model.endswith(".pth"):
                continue

            print(f"Processing: {checkpoint_model}")

            name_parts = checkpoint_model.replace(".pth", "").split("_")

            if len(name_parts) < 3:
                raise ValueError("Checkpoint name does not contain enough elements to extract 'saving_name'.")

            saving_name = "_".join(name_parts[-3:])

            config_path = "config.yaml"
            yaml = YAML()

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file '{config_path}' not found!")

            with open(config_path, "r") as f:
                config = yaml.load(f)

            cleaned_part = name_parts[-2].replace("N", "")
            config["dataset"]["train_set"] = os.path.join(
                DATASET_DIR, f"train_set_{name_parts[-3]}_{cleaned_part}.csv"
            )

            if "N" in name_parts[-2]:
                config["yolo"]["model"] = "Training\\yolov5n.yaml"
            else:
                config["yolo"]["model"] = "Training\\yolov5s.yaml"

            config["saving_folder"] = "_".join(name_parts[-3:-1])
            config["checkpoints"] = os.path.join(folder_path, checkpoint_model)
            config["saving_name"] = saving_name

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            print(f"Updated 'checkpoints': {config['checkpoints']}")
            print(f"Updated 'saving_name': {config['saving_name']}")
            print(f"Updated 'saving_folder': {config['saving_folder']}")

            if mode == "viz" or mode == 'all':
                create_predictions_images()

            elif mode == "predictions" or mode == 'all':
                create_ground_truths_files('test')
                create_ground_truths_files("val")
                create_detection_files()
                calculation_TP_FP('test')
                calculation_TP_FP('val')
                metrics_calculation('test')
                metrics_calculation('val')
                calculation_IOU()

            elif mode == "summary" or mode == 'all' : 
                if config["saving_folder"] in [
                    "global_sex", "global_sexN", 
                    "loso_male", "loso_female", 
                    "loso_femaleN", "loso_maleN"
                ]:
                    compute_and_save_detection_rates('Gender')
                else:
                    compute_and_save_detection_rates('Race')

        generate_summary()  

    global_performance('Gender')
    global_performance('Race')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YOLO checkpoints")
    parser.add_argument(
        "--mode", 
        choices=["viz", "predictions", "summary", "all"], 
        default="summary", 
        help="Operation mode"
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        help="Name of the checkpoint folder to process (e.g., global_sexN)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Specific checkpoint file to process inside the folder (e.g., model_epoch_10.pth)"
    )
    args = parser.parse_args()
    process_checkpoints(args.mode, args.folder, args.checkpoint)


    