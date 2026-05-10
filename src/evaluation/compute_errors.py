from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import torch

from src.utils.device import get_device
from src.utils.experiment import load_best_checkpoint
from src.utils.get_folders_utils import get_processed_folder, get_evaluation_results_main_folder
from src.utils.create_folders_utils import create_eval_results_folder

def create_evaluation_dataloaders(config):
    processed_data_folder = get_processed_folder(config)
    
    # load config
    window_size = config["training"]["window_size"]
    batch_size = config["evaluation"]["batch_size"]

    arrays = np.load(f"{processed_data_folder}/arrays.npz")

    train_dataset = TimeSeriesDataset(arrays["train"], window_size)
    val_dataset = TimeSeriesDataset(arrays["val"], window_size)
    actor2_test_dataset = TimeSeriesDataset(arrays["actor2_test"], window_size)
    actor1_test_dataset = TimeSeriesDataset(arrays["actor1_test"], window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    actor2_test_loader = DataLoader(actor2_test_dataset, batch_size, shuffle=False)
    actor1_test_loader = DataLoader(actor1_test_dataset, batch_size, shuffle=False)

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "actor2_test_loader": actor2_test_loader,
        "actor1_test_loader": actor1_test_loader
    }
    return data_loaders

def compute_errors_per_loader(model, dataloader):
    device = get_device()
    model.eval()

    forecast_errors = []
    recon_errors = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            # Case 1: MTAD-GAT (dict output)
            if isinstance(output, dict):
                pred = output["pred"]
                recon = output.get("recon", None)
            else:
                # Case 2: GDN (tensor output)
                pred = output
                recon = None

            # --- Forecast error ---
            f_err = torch.abs(pred - y)   # (B, N)
            forecast_errors.append(f_err.cpu().numpy())

            # --- Reconstruction error (if exists) ---
            if recon is not None:
                r_err = torch.abs(recon - x)  # (B, W, N)
                r_err_last = r_err[:, -1, :]  # align with forcast to become (B, N)
                recon_errors.append(r_err_last.cpu().numpy())

    forecast_errors = np.concatenate(forecast_errors, axis=0)

    if len(recon_errors) > 0:
        recon_errors = np.concatenate(recon_errors, axis=0)
    else:
        recon_errors = None

    return {
        "forecast": forecast_errors,   # shape [T, N]
        "reconstruction": recon_errors # shape [T, N] or None
    }

def compute_raw_errors_all_splits(config):
    """Compute raw errors for all splits and save them in eval results folder"""
    print("Computing raw errors for all splits...")

    model = load_best_checkpoint(config)

    data_loaders = create_evaluation_dataloaders(config)

    train_errors = compute_errors_per_loader(model, data_loaders["train_loader"])
    val_errors = compute_errors_per_loader(model, data_loaders["val_loader"])
    actor2_test_errors = compute_errors_per_loader(model, data_loaders["actor2_test_loader"])
    actor1_test_errors = compute_errors_per_loader(model, data_loaders["actor1_test_loader"])

    # Check once
    has_recon = train_errors["reconstruction"] is not None

    if has_recon:
        train = {"forecast": train_errors["forecast"], 
                 "reconstruction": train_errors["reconstruction"]}
        val = {"forecast": val_errors["forecast"], 
                 "reconstruction": val_errors["reconstruction"]}
        actor2_test = {"forecast": actor2_test_errors["forecast"], 
                 "reconstruction": actor2_test_errors["reconstruction"]}
        actor1_test = {"forecast": actor1_test_errors["forecast"], 
                 "reconstruction": actor1_test_errors["reconstruction"]}
    else:
        train = {"forecast": train_errors["forecast"]}
        val = {"forecast": val["forecast"]}
        actor2_test = {"forecast": actor2_test["forecast"]}
        actor1_test = {"forecast": actor1_test["forecast"]}
        
    raw_errros = {
        "train": train,
        "val": val,
        "actor2_test": actor2_test,
        "actor1_test": actor1_test,
    }

    eval_results_folder = create_eval_results_folder(config)

    np.savez(f"{eval_results_folder}/raw_errors.npz", raw_errros)
    
def normalize_raw_errors_all_splits(config):
    """
    Normalize raw errors using train statistics only (robust, per error type)
    """
    print("Normalizing raw errors for all splits...")
    
    eval_results_folder = get_evaluation_results_main_folder(config)

    # IMPORTANT: allow_pickle=True for dict
    data = np.load(f"{eval_results_folder}/raw_errors.npz", allow_pickle=True)

    errors = data["arr_0"].item()  # recover dict

    normalized = {}

    # Loop over error types (forecast, reconstruction)
    for error_type in errors["train"].keys():

        train_err = errors["train"][error_type]

        # --- robust stats ---
        median = np.median(train_err, axis=0)
        iqr = np.percentile(train_err, 75, axis=0) - np.percentile(train_err, 25, axis=0)

        # adaptive stabilization
        iqr = np.maximum(iqr, 1.0)

        def normalize(e):
            norm = (e - median) / iqr
            norm = np.clip(norm, -5, 5)
            return np.abs(norm)

        # Apply to all splits
        for split in ["train", "val", "actor2_test", "actor1_test"]:
            if split not in normalized:
                normalized[split] = {}

            normalized[split][error_type] = normalize(errors[split][error_type])

    eval_results_folder = get_evaluation_results_main_folder(config)
    np.savez(f"{eval_results_folder}/norm_errors.npz", normalized)

def compute_errors(config):
    # Compute errors for all loaders
    compute_raw_errors_all_splits(config)

    # normalize computed raw errors
    normalize_raw_errors_all_splits(config)