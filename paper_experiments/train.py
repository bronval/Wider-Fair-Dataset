import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm  
import os
import yaml
from ruamel.yaml import YAML as RuamelYAML
import pandas as pd 
import argparse

from Training.yolo import *
from Training.dataset import *

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_model_architecture(yaml_path):
    yaml = RuamelYAML()
    with open(yaml_path, encoding="ascii", errors="ignore") as f:
        model_config = yaml.load(f)
    return model_config

def set_seed(seed=42):
    ### Partial reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_yolo(model, dataloader, val_dataloader,config, device, num_epochs ,indice = ''):
    model = model.to(device)
    model.train()  

    checkpoint_dir = os.path.join("checkpoints", config['saving_folder'], indice)
    os.makedirs(checkpoint_dir, exist_ok=True)  
    
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'checkpoint_path'])

    best_val_loss = float('inf')
    best_checkpoint_path = None

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train() 

        for batch_idx, (images, targets, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            ### Forward pass
            images = images.to(device)
            targets = [t.to(device) for t in targets] 
            optimizer.zero_grad()
            losses = model(images, targets)
            total_loss = sum(losses)*len(targets)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            print(f"Batch {batch_idx}/{(len(dataloader))}, Loss: {total_loss.item()}")

        ### Average epoch loss
        avg_train_loss = epoch_loss / (len(dataloader))
        

        ### Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():  
            for val_batch_idx, (val_images, val_targets, _) in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")):
                val_images = val_images.to(device)
                val_targets = [v.to(device) for v in val_targets] 
                val_losses = model(val_images, val_targets)
                total_val_loss = sum(val_losses) * len(val_targets)
                val_loss += total_val_loss.item()

            ### Average validation loss
            avg_val_loss = val_loss / len(val_dataloader)

        checkpoint_path = os.path.join(checkpoint_dir, f"yolo_epoch_{epoch+1}_{config['saving_folder']}_{indice}.pth")
        ### Save checkpoint
        if epoch > 20 : 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = checkpoint_path

        loss_df.loc[epoch] = [
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            checkpoint_path
        ]       
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    loss_csv_path = os.path.join(checkpoint_dir, f"training_logs_{indice}.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    
    return best_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO with custom dataset path and model type.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Name of the training dataset folder inside the dataset directory.")
    parser.add_argument("--model", type=str, required=True, choices=["S", "N"], help="Model type: 'S' for small, 'N' for nano.")
    args = parser.parse_args()

    config_path = 'config.yaml'
    DATASET_DIR = "Train_test_dataset"

    yaml = RuamelYAML()
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # Modify the config based on command-line input
    config['dataset']['train_set'] = os.path.join(DATASET_DIR, args.train_dataset)
    input_str = args.train_dataset.replace(".csv", "")
    parts = input_str.split("_")
    new_str = "_".join(parts[2:])
    config['saving_folder'] = new_str + ('N' if args.model == "N" else "")
    config['yolo']['model'] = os.path.join('Training' , "yolov5"+args.model.lower()+".yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    for i in range(3):
        set_seed(i)
        g = torch.Generator()
        g.manual_seed(i)

        train_dataset = ImageFolder(config, phase='train')
        validation_dataset = ImageFolder(config, phase='val')

        train_sampler = RandomSampler(train_dataset, generator=g)
        train_batch_sampler = BatchSampler(train_sampler, batch_size=config["batch_size"], drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=4,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            generator=g
        )

        val_sampler = RandomSampler(validation_dataset, generator=g)
        val_batch_sampler = BatchSampler(val_sampler, batch_size=config["batch_size"], drop_last=False)

        val_loader = DataLoader(
            validation_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=4,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=validation_dataset.collate_fn,
            generator=g
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = config["yolo"]["model"]
        epoch = config["yolo"]["epochs"]
        model_architecture = load_model_architecture(model_name)
        model = Yolo(model_architecture, config=config)

        best_checkpoint = train_yolo(model, train_loader, val_loader, config, device, epoch, indice=str(i))