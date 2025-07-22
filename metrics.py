import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from typing import Optional



# Add this import at the top with the other imports
import os
import csv
import datetime

# Add this function after the threshold_metrics function
def save_results_to_csv(args, metrics_dict: dict, standard_metrics: tuple) -> str:
    """
    Save the experiment results to a CSV file.
    
    Args:
        args: Command line arguments containing experiment configuration
        metrics_dict: Dictionary of specialized metrics with format {metric_name: (mean, std)}
        standard_metrics: Tuple of (mse_mean, mse_std, l1_mean, l1_std, gmean_mean, gmean_std)
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build a descriptive filename
    filename = f"{args.dataset}"
    
    # Add method information
    if args.bmse:
        filename += f"_bmse-{args.imp}"
    if args.lds:
        filename += "_lds"
    if args.fds:
        filename += "_fds"
    if args.reweight != 'none':
        filename += f"_{args.reweight}"
    
    # Add loss function and other key parameters
    filename += f"_{args.loss}"
    filename += f"_e{args.epoch}"
    
    # Add timestamp and extension
    filename += f"_{timestamp}.csv"
    
    # Full path in the experiment directory
    filepath = os.path.join(args.store_root, f"{args.dataset}_{args.model}", filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write to CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with experiment details
        writer.writerow(["Experiment Summary"])
        writer.writerow(["Dataset", args.dataset])
        writer.writerow(["Model", args.model])
        writer.writerow(["BMSE", str(args.bmse)])
        if args.bmse:
            writer.writerow(["BMSE Implementation", args.imp])
        writer.writerow(["LDS", str(args.lds)])
        writer.writerow(["FDS", str(args.fds)])
        writer.writerow(["Reweighting", args.reweight])
        writer.writerow(["Loss Function", args.loss])
        writer.writerow(["Epochs", args.epoch])
        writer.writerow(["Learning Rate", args.lr])
        writer.writerow(["Batch Size", args.batch_size])
        writer.writerow(["Weight Decay", args.weight_decay])
        writer.writerow(["Number of Trials", len(args.seeds)])
        writer.writerow(["Seeds", " ".join(map(str, args.seeds))])
        writer.writerow([])
        
        # Write standard metrics
        mse_mean, mse_std, l1_mean, l1_std, gmean_mean, gmean_std = standard_metrics
        writer.writerow(["Standard Metrics"])
        writer.writerow(["Metric", "Mean", "Std Dev"])
        writer.writerow(["MSE", f"{mse_mean:.4f}", f"{mse_std:.4f}"])
        writer.writerow(["L1", f"{l1_mean:.4f}", f"{l1_std:.4f}"])
        writer.writerow(["G-Mean", f"{gmean_mean:.4f}", f"{gmean_std:.4f}"])
        writer.writerow([])
        
        # Write specialized metrics
        writer.writerow(["Specialized Metrics"])
        writer.writerow(["Metric", "Mean", "Std Dev"])
        for metric_name, (mean, std) in metrics_dict.items():
            writer.writerow([metric_name, f"{mean:.4f}", f"{std:.4f}"])
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def process_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Processes model predictions to ensure compatibility with models 
    that output either single or multiple channels. In case of multiple 
    channels, it extracts the predictions corresponding to the first channel.

    Parameters:
    - predictions (np.ndarray): The predictions made by the model. 
                    Can be either 1D or 2D (for models with multiple outputs).
    Returns:
    - processed_predictions (np.ndarray): The processed predictions.
    """
    # Handle predictions based on their shape
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # More than one channel: select the first channel
        processed_predictions = predictions[:, 0].flatten()
    else:
        # Single channel: just flatten it
        processed_predictions = predictions.flatten()

    return processed_predictions


def evaluate_mae(
        model: torch.nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        below_threshold: Optional[float] = None,
        above_threshold: Optional[float] = None,
        between_thresholds: bool = False,
        device: Optional[torch.device] = None) -> float:
    """
    Evaluates a given PyTorch model using Mean Absolute Error (MAE) on the provided test data,
    with options to conditionally calculate MAE based on specified thresholds.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - below_threshold (float, optional): The lower bound threshold for y_test.
    - above_threshold (float, optional): The upper bound threshold for y_test.
    - between_thresholds (bool, optional): If True, selects values BETWEEN thresholds
                                          rather than outside them. Default is False.
    - device (torch.device, optional): Device to run the model on. If None, will use model's current device.

    Returns:
    - float: The MAE loss of the model on the filtered test data.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert inputs to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get device
    if device is None:
        device = next(model.parameters()).device
    X_test_tensor = X_test_tensor.to(device)
    
    # Make predictions
    with torch.no_grad():
        _, predictions = model(X_test_tensor)
        predictions = predictions.cpu().numpy()

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Filter y_test and predictions based on thresholds
    if below_threshold is not None or above_threshold is not None:
        if below_threshold is not None and above_threshold is not None:
            if between_thresholds:
                # Select values BETWEEN thresholds
                mask = (y_test > below_threshold) & (y_test < above_threshold)
            else:
                # Select values OUTSIDE thresholds (original behavior)
                mask = (y_test >= above_threshold) | (y_test <= below_threshold)
        elif below_threshold is not None:
            mask = y_test <= below_threshold
        else:  # above_threshold is not None
            mask = y_test >= above_threshold

        filtered_predictions = predictions[mask]
        filtered_y_test = y_test[mask]
    else:
        filtered_predictions = predictions
        filtered_y_test = y_test

    # Calculate MAE
    mae_loss = mean_absolute_error(filtered_y_test, filtered_predictions)
    return mae_loss


def evaluate_pcc(
        model: torch.nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        below_threshold: Optional[float] = None,
        above_threshold: Optional[float] = None,
        between_thresholds: bool = False,
        device: Optional[torch.device] = None) -> float:
    """
    Evaluates a given PyTorch model using Pearson Correlation Coefficient (PCC) on the provided test data,
    with an option to conditionally calculate PCC based on specified thresholds.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - below_threshold (float, optional): The lower bound threshold for y_test to be included in PCC calculation.
    - above_threshold (float, optional): The upper bound threshold for y_test to be included in PCC calculation.
    - between_thresholds (bool, optional): If True, selects values BETWEEN thresholds
                                          rather than outside them. Default is False.
    - device (torch.device, optional): Device to run the model on. If None, will use model's current device.

    Returns:
    - float: The Pearson Correlation Coefficient between the model predictions and the filtered test data.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert inputs to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get device
    if device is None:
        device = next(model.parameters()).device
    X_test_tensor = X_test_tensor.to(device)
    
    # Make predictions
    with torch.no_grad():
        _, predictions = model(X_test_tensor)
        predictions = predictions.cpu().numpy()

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Filter y_test and predictions based on thresholds
    if below_threshold is not None or above_threshold is not None:
        if below_threshold is not None and above_threshold is not None:
            if between_thresholds:
                # Select values BETWEEN thresholds
                mask = (y_test > below_threshold) & (y_test < above_threshold)
            else:
                # Select values OUTSIDE thresholds (original behavior)
                mask = (y_test >= above_threshold) | (y_test <= below_threshold)
        elif below_threshold is not None:
            mask = y_test <= below_threshold
        else:  # above_threshold is not None
            mask = y_test >= above_threshold

        filtered_predictions = predictions[mask]
        filtered_y_test = y_test[mask]
    else:
        filtered_predictions = predictions
        filtered_y_test = y_test

    # Calculate PCC
    pcc, _ = pearsonr(filtered_y_test.flatten(), filtered_predictions.flatten())

    return pcc


def evaluate_sep_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sep_threshold: float = np.log(10),
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the SEP dataset using multiple metrics on both training and test sets.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - sep_threshold (float): Threshold value for conditional metrics (values above this are considered "rare")
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating SEP metrics for threshold: {sep_threshold}")
    
    # Standard metrics on test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Standard metrics on training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # Conditional metrics for rare samples (test set)
    error_mae_cond = evaluate_mae(model, X_test, y_test, above_threshold=sep_threshold, device=device)
    results["mae+"] = error_mae_cond
    print(f'MAE error for values >= {sep_threshold} (test): {error_mae_cond:.4f}')
    
    error_pcc_cond = evaluate_pcc(model, X_test, y_test, above_threshold=sep_threshold, device=device)
    results["pcc+"] = error_pcc_cond
    print(f'PCC for values >= {sep_threshold} (test): {error_pcc_cond:.4f}')
    
    # Conditional metrics for rare samples (training set)
    error_mae_cond_train = evaluate_mae(model, X_train, y_train, above_threshold=sep_threshold, device=device)
    results["train_mae+"] = error_mae_cond_train
    print(f'MAE error for values >= {sep_threshold} (train): {error_mae_cond_train:.4f}')
    
    error_pcc_cond_train = evaluate_pcc(model, X_train, y_train, above_threshold=sep_threshold, device=device)
    results["train_pcc+"] = error_pcc_cond_train
    print(f'PCC for values >= {sep_threshold} (train): {error_pcc_cond_train:.4f}')

    # Average of standard and conditional metrics
    avg_mae_maep = (error_mae + error_mae_cond) / 2
    results["avg_mae_maep"] = avg_mae_maep
    print(f'Average of MAE and MAE+ (test): {avg_mae_maep:.4f}')
    
    avg_pcc_pccp = (error_pcc + error_pcc_cond) / 2
    results["avg_pcc_pccp"] = avg_pcc_pccp
    print(f'Average of PCC and PCC+ (test): {avg_pcc_pccp:.4f}')
    
    return results


def evaluate_ed_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        mae_plus_threshold: float = 0.5,
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the SEP dataset with comprehensive metrics.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - mae_plus_threshold (float): Threshold value for conditional metrics (values above this are considered "rare")
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating ED metrics for threshold: {mae_plus_threshold}")
    
    # Standard metrics on test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Standard metrics on training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # Conditional metrics for rare samples (test set)
    error_mae_cond = evaluate_mae(model, X_test, y_test, above_threshold=mae_plus_threshold, device=device)
    results["mae+"] = error_mae_cond
    print(f'MAE error for values >= {mae_plus_threshold} (test): {error_mae_cond:.4f}')
    
    error_pcc_cond = evaluate_pcc(model, X_test, y_test, above_threshold=mae_plus_threshold, device=device)
    results["pcc+"] = error_pcc_cond
    print(f'PCC for values >= {mae_plus_threshold} (test): {error_pcc_cond:.4f}')
    
    # Conditional metrics for rare samples (training set)
    error_mae_cond_train = evaluate_mae(model, X_train, y_train, above_threshold=mae_plus_threshold, device=device)
    results["train_mae+"] = error_mae_cond_train
    print(f'MAE error for values >= {mae_plus_threshold} (train): {error_mae_cond_train:.4f}')
    
    error_pcc_cond_train = evaluate_pcc(model, X_train, y_train, above_threshold=mae_plus_threshold, device=device)
    results["train_pcc+"] = error_pcc_cond_train
    print(f'PCC for values >= {mae_plus_threshold} (train): {error_pcc_cond_train:.4f}')
    
    # Average of standard and conditional metrics
    avg_mae_maep = (error_mae + error_mae_cond) / 2
    results["avg_mae_maep"] = avg_mae_maep
    print(f'Average of MAE and MAE+ (test): {avg_mae_maep:.4f}')
    
    avg_pcc_pccp = (error_pcc + error_pcc_cond) / 2
    results["avg_pcc_pccp"] = avg_pcc_pccp
    print(f'Average of PCC and PCC+ (test): {avg_pcc_pccp:.4f}')
    
    return results


def evaluate_onp_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rare_low_threshold: float = np.log10(350),  # Default lower threshold
        rare_high_threshold: float = np.log10(35000),  # Default upper threshold
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the Online News Popularity (ONP) dataset using multiple metrics 
    on both training and test sets, with data segmented into rare low, frequent, and rare high regions.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - rare_low_threshold (float): Threshold below which values are considered "rare low"
    - rare_high_threshold (float): Threshold above which values are considered "rare high"
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating ONP metrics for thresholds: {rare_low_threshold} and {rare_high_threshold}")
    
    # REGULAR METRICS (all data)
    # Test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    # Training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    # Test set
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Training set
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # FREQUENT METRICS (between rare_low_threshold and rare_high_threshold)
    # Test set
    error_mae_freq = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["mae_freq"] = error_mae_freq
    print(f'MAE freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (test): {error_mae_freq:.4f}')
    
    # Training set
    error_mae_freq_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["train_mae_freq"] = error_mae_freq_train
    print(f'MAE freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (train): {error_mae_freq_train:.4f}')
    
    # Test set
    error_pcc_freq = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["pcc_freq"] = error_pcc_freq
    print(f'PCC freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (test): {error_pcc_freq:.4f}')
    
    # Training set
    error_pcc_freq_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["train_pcc_freq"] = error_pcc_freq_train
    print(f'PCC freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (train): {error_pcc_freq_train:.4f}')
    
    # RARE LOW METRICS (y < rare_low_threshold)
    # Test set
    error_mae_rare_low = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold,
        device=device)
    results["mae_rare_low"] = error_mae_rare_low
    print(f'MAE rare low (y < {rare_low_threshold:.4f}) (test): {error_mae_rare_low:.4f}')
    
    # Training set
    error_mae_rare_low_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold,
        device=device)
    results["train_mae_rare_low"] = error_mae_rare_low_train
    print(f'MAE rare low (y < {rare_low_threshold:.4f}) (train): {error_mae_rare_low_train:.4f}')
    
    # Test set
    error_pcc_rare_low = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold,
        device=device)
    results["pcc_rare_low"] = error_pcc_rare_low
    print(f'PCC rare low (y < {rare_low_threshold:.4f}) (test): {error_pcc_rare_low:.4f}')
    
    # Training set
    error_pcc_rare_low_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold,
        device=device)
    results["train_pcc_rare_low"] = error_pcc_rare_low_train
    print(f'PCC rare low (y < {rare_low_threshold:.4f}) (train): {error_pcc_rare_low_train:.4f}')
    
    # RARE HIGH METRICS (y > rare_high_threshold)
    # Test set
    error_mae_rare_high = evaluate_mae(
        model, X_test, y_test, 
        above_threshold=rare_high_threshold,
        device=device)
    results["mae_rare_high"] = error_mae_rare_high
    print(f'MAE rare high (y > {rare_high_threshold:.4f}) (test): {error_mae_rare_high:.4f}')
    
    # Training set
    error_mae_rare_high_train = evaluate_mae(
        model, X_train, y_train, 
        above_threshold=rare_high_threshold,
        device=device)
    results["train_mae_rare_high"] = error_mae_rare_high_train
    print(f'MAE rare high (y > {rare_high_threshold:.4f}) (train): {error_mae_rare_high_train:.4f}')
    
    # Test set
    error_pcc_rare_high = evaluate_pcc(
        model, X_test, y_test, 
        above_threshold=rare_high_threshold,
        device=device)
    results["pcc_rare_high"] = error_pcc_rare_high
    print(f'PCC rare high (y > {rare_high_threshold:.4f}) (test): {error_pcc_rare_high:.4f}')
    
    # Training set
    error_pcc_rare_high_train = evaluate_pcc(
        model, X_train, y_train, 
        above_threshold=rare_high_threshold,
        device=device)
    results["train_pcc_rare_high"] = error_pcc_rare_high_train
    print(f'PCC rare high (y > {rare_high_threshold:.4f}) (train): {error_pcc_rare_high_train:.4f}')
    
    return results


def evaluate_sarcos_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lower_threshold: float = -0.5,
        upper_threshold: float = 0.5,
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the SARCOS dataset using multiple metrics 
    on both training and test sets, with data segmented into rare low, frequent, and rare high regions.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - lower_threshold (float): Threshold below which values are considered "rare low"
    - upper_threshold (float): Threshold above which values are considered "rare high"
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating SARCOS metrics for thresholds: {lower_threshold} and {upper_threshold}")
    
    # REGULAR METRICS (all data)
    # Test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    # Training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    # Test set
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Training set
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # FREQUENT METRICS (between lower_threshold and upper_threshold)
    # Test set
    error_mae_freq = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=lower_threshold, 
        above_threshold=upper_threshold, 
        between_thresholds=True,
        device=device)
    results["mae_freq"] = error_mae_freq
    print(f'MAE freq ({lower_threshold:.4f} < y < {upper_threshold:.4f}) (test): {error_mae_freq:.4f}')
    
    # Training set
    error_mae_freq_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=lower_threshold, 
        above_threshold=upper_threshold, 
        between_thresholds=True,
        device=device)
    results["train_mae_freq"] = error_mae_freq_train
    print(f'MAE freq ({lower_threshold:.4f} < y < {upper_threshold:.4f}) (train): {error_mae_freq_train:.4f}')
    
    # Test set
    error_pcc_freq = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=lower_threshold, 
        above_threshold=upper_threshold, 
        between_thresholds=True,
        device=device)
    results["pcc_freq"] = error_pcc_freq
    print(f'PCC freq ({lower_threshold:.4f} < y < {upper_threshold:.4f}) (test): {error_pcc_freq:.4f}')
    
    # Training set
    error_pcc_freq_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=lower_threshold, 
        above_threshold=upper_threshold, 
        between_thresholds=True,
        device=device)
    results["train_pcc_freq"] = error_pcc_freq_train
    print(f'PCC freq ({lower_threshold:.4f} < y < {upper_threshold:.4f}) (train): {error_pcc_freq_train:.4f}')
    
    # RARE LOW METRICS (y < lower_threshold)
    # Test set
    error_mae_rare_low = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=lower_threshold,
        device=device)
    results["mae_rare_low"] = error_mae_rare_low
    print(f'MAE rare low (y < {lower_threshold:.4f}) (test): {error_mae_rare_low:.4f}')
    
    # Training set
    error_mae_rare_low_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=lower_threshold,
        device=device)
    results["train_mae_rare_low"] = error_mae_rare_low_train
    print(f'MAE rare low (y < {lower_threshold:.4f}) (train): {error_mae_rare_low_train:.4f}')
    
    # Test set
    error_pcc_rare_low = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=lower_threshold,
        device=device)
    results["pcc_rare_low"] = error_pcc_rare_low
    print(f'PCC rare low (y < {lower_threshold:.4f}) (test): {error_pcc_rare_low:.4f}')
    
    # Training set
    error_pcc_rare_low_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=lower_threshold,
        device=device)
    results["train_pcc_rare_low"] = error_pcc_rare_low_train
    print(f'PCC rare low (y < {lower_threshold:.4f}) (train): {error_pcc_rare_low_train:.4f}')
    
    # RARE HIGH METRICS (y > upper_threshold)
    # Test set
    error_mae_rare_high = evaluate_mae(
        model, X_test, y_test, 
        above_threshold=upper_threshold,
        device=device)
    results["mae_rare_high"] = error_mae_rare_high
    print(f'MAE rare high (y > {upper_threshold:.4f}) (test): {error_mae_rare_high:.4f}')
    
    # Training set
    error_mae_rare_high_train = evaluate_mae(
        model, X_train, y_train, 
        above_threshold=upper_threshold,
        device=device)
    results["train_mae_rare_high"] = error_mae_rare_high_train
    print(f'MAE rare high (y > {upper_threshold:.4f}) (train): {error_mae_rare_high_train:.4f}')
    
    # Test set
    error_pcc_rare_high = evaluate_pcc(
        model, X_test, y_test, 
        above_threshold=upper_threshold,
        device=device)
    results["pcc_rare_high"] = error_pcc_rare_high
    print(f'PCC rare high (y > {upper_threshold:.4f}) (test): {error_pcc_rare_high:.4f}')
    
    # Training set
    error_pcc_rare_high_train = evaluate_pcc(
        model, X_train, y_train, 
        above_threshold=upper_threshold,
        device=device)
    results["train_pcc_rare_high"] = error_pcc_rare_high_train
    print(f'PCC rare high (y > {upper_threshold:.4f}) (train): {error_pcc_rare_high_train:.4f}')
    
    return results


def evaluate_bf_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        freq_threshold: float = np.log10(4),  # Default for blog feedback
        rare_threshold: float = np.log10(40),  # Default for blog feedback
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the Blog Feedback dataset using multiple metrics 
    on both training and test sets, with data segmented into frequent (y < freq_threshold), 
    medium (freq_threshold < y < rare_threshold), and rare (y > rare_threshold) regions.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - freq_threshold (float): Threshold below which values are considered "frequent"
    - rare_threshold (float): Threshold above which values are considered "rare"
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating Blog Feedback metrics for thresholds: {freq_threshold} and {rare_threshold}")
    
    # REGULAR METRICS (all data)
    # Test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    # Training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    # Test set
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Training set
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # FREQUENT METRICS (y < freq_threshold)
    # Test set
    error_mae_freq = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=freq_threshold,
        device=device)
    results["mae_freq"] = error_mae_freq
    print(f'MAE freq (y < {freq_threshold:.4f}) (test): {error_mae_freq:.4f}')
    
    # Training set
    error_mae_freq_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=freq_threshold,
        device=device)
    results["train_mae_freq"] = error_mae_freq_train
    print(f'MAE freq (y < {freq_threshold:.4f}) (train): {error_mae_freq_train:.4f}')
    
    # Test set
    error_pcc_freq = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=freq_threshold,
        device=device)
    results["pcc_freq"] = error_pcc_freq
    print(f'PCC freq (y < {freq_threshold:.4f}) (test): {error_pcc_freq:.4f}')
    
    # Training set
    error_pcc_freq_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=freq_threshold,
        device=device)
    results["train_pcc_freq"] = error_pcc_freq_train
    print(f'PCC freq (y < {freq_threshold:.4f}) (train): {error_pcc_freq_train:.4f}')
    
    # MEDIUM METRICS (freq_threshold < y < rare_threshold)
    # Test set
    error_mae_med = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=freq_threshold,
        above_threshold=rare_threshold,
        between_thresholds=True,
        device=device)
    results["mae_med"] = error_mae_med
    print(f'MAE med ({freq_threshold:.4f} < y < {rare_threshold:.4f}) (test): {error_mae_med:.4f}')
    
    # Training set
    error_mae_med_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=freq_threshold,
        above_threshold=rare_threshold,
        between_thresholds=True,
        device=device)
    results["train_mae_med"] = error_mae_med_train
    print(f'MAE med ({freq_threshold:.4f} < y < {rare_threshold:.4f}) (train): {error_mae_med_train:.4f}')
    
    # Test set
    error_pcc_med = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=freq_threshold,
        above_threshold=rare_threshold,
        between_thresholds=True,
        device=device)
    results["pcc_med"] = error_pcc_med
    print(f'PCC med ({freq_threshold:.4f} < y < {rare_threshold:.4f}) (test): {error_pcc_med:.4f}')
    
    # Training set
    error_pcc_med_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=freq_threshold,
        above_threshold=rare_threshold,
        between_thresholds=True,
        device=device)
    results["train_pcc_med"] = error_pcc_med_train
    print(f'PCC med ({freq_threshold:.4f} < y < {rare_threshold:.4f}) (train): {error_pcc_med_train:.4f}')
    
    # RARE METRICS (y > rare_threshold)
    # Test set
    error_mae_rare = evaluate_mae(
        model, X_test, y_test, 
        above_threshold=rare_threshold,
        device=device)
    results["mae_rare"] = error_mae_rare
    print(f'MAE rare (y > {rare_threshold:.4f}) (test): {error_mae_rare:.4f}')
    
    # Training set
    error_mae_rare_train = evaluate_mae(
        model, X_train, y_train, 
        above_threshold=rare_threshold,
        device=device)
    results["train_mae_rare"] = error_mae_rare_train
    print(f'MAE rare (y > {rare_threshold:.4f}) (train): {error_mae_rare_train:.4f}')
    
    # Test set
    error_pcc_rare = evaluate_pcc(
        model, X_test, y_test, 
        above_threshold=rare_threshold,
        device=device)
    results["pcc_rare"] = error_pcc_rare
    print(f'PCC rare (y > {rare_threshold:.4f}) (test): {error_pcc_rare:.4f}')
    
    # Training set
    error_pcc_rare_train = evaluate_pcc(
        model, X_train, y_train, 
        above_threshold=rare_threshold,
        device=device)
    results["train_pcc_rare"] = error_pcc_rare_train
    print(f'PCC rare (y > {rare_threshold:.4f}) (train): {error_pcc_rare_train:.4f}')
    
    return results


def evaluate_asc_metrics(
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rare_low_threshold: float = np.log10(200),  
        rare_high_threshold: float = np.log10(20000),  
        device: Optional[torch.device] = None) -> dict:
    """
    Evaluates a model on the Appliance Energy Prediction (ASC) dataset using multiple metrics 
    on both training and test sets, with data segmented into rare low, frequent, and rare high regions.
    
    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate
    - X_train (np.ndarray): Training features
    - y_train (np.ndarray): Training target values
    - X_test (np.ndarray): Test features
    - y_test (np.ndarray): Test target values
    - rare_low_threshold (float): Threshold below which values are considered "rare low"
    - rare_high_threshold (float): Threshold above which values are considered "rare high"
    - device (torch.device, optional): Device to run the model on
    
    Returns:
    - dict: Dictionary containing all computed metrics
    """
    results = {}
    print(f"Evaluating ASC metrics for thresholds: {rare_low_threshold} and {rare_high_threshold}")
    
    # REGULAR METRICS (all data)
    # Test set
    error_mae = evaluate_mae(model, X_test, y_test, device=device)
    results["mae"] = error_mae
    print(f'MAE error (test): {error_mae:.4f}')
    
    # Training set
    error_mae_train = evaluate_mae(model, X_train, y_train, device=device)
    results["train_mae"] = error_mae_train
    print(f'MAE error (train): {error_mae_train:.4f}')
    
    # Test set
    error_pcc = evaluate_pcc(model, X_test, y_test, device=device)
    results["pcc"] = error_pcc
    print(f'PCC (test): {error_pcc:.4f}')
    
    # Training set
    error_pcc_train = evaluate_pcc(model, X_train, y_train, device=device)
    results["train_pcc"] = error_pcc_train
    print(f'PCC (train): {error_pcc_train:.4f}')
    
    # FREQUENT METRICS (between rare_low_threshold and rare_high_threshold)
    # Test set
    error_mae_freq = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["mae_freq"] = error_mae_freq
    print(f'MAE freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (test): {error_mae_freq:.4f}')
    
    # Training set
    error_mae_freq_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["train_mae_freq"] = error_mae_freq_train
    print(f'MAE freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (train): {error_mae_freq_train:.4f}')
    
    # Test set
    error_pcc_freq = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["pcc_freq"] = error_pcc_freq
    print(f'PCC freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (test): {error_pcc_freq:.4f}')
    
    # Training set
    error_pcc_freq_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold, 
        above_threshold=rare_high_threshold, 
        between_thresholds=True,
        device=device)
    results["train_pcc_freq"] = error_pcc_freq_train
    print(f'PCC freq ({rare_low_threshold:.4f} < y < {rare_high_threshold:.4f}) (train): {error_pcc_freq_train:.4f}')
    
    # RARE LOW METRICS (y < rare_low_threshold)
    # Test set
    error_mae_rare_low = evaluate_mae(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold,
        device=device)
    results["mae_rare_low"] = error_mae_rare_low
    print(f'MAE rare low (y < {rare_low_threshold:.4f}) (test): {error_mae_rare_low:.4f}')
    
    # Training set
    error_mae_rare_low_train = evaluate_mae(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold,
        device=device)
    results["train_mae_rare_low"] = error_mae_rare_low_train
    print(f'MAE rare low (y < {rare_low_threshold:.4f}) (train): {error_mae_rare_low_train:.4f}')
    
    # Test set
    error_pcc_rare_low = evaluate_pcc(
        model, X_test, y_test, 
        below_threshold=rare_low_threshold,
        device=device)
    results["pcc_rare_low"] = error_pcc_rare_low
    print(f'PCC rare low (y < {rare_low_threshold:.4f}) (test): {error_pcc_rare_low:.4f}')
    
    # Training set
    error_pcc_rare_low_train = evaluate_pcc(
        model, X_train, y_train, 
        below_threshold=rare_low_threshold,
        device=device)
    results["train_pcc_rare_low"] = error_pcc_rare_low_train
    print(f'PCC rare low (y < {rare_low_threshold:.4f}) (train): {error_pcc_rare_low_train:.4f}')
    
    # RARE HIGH METRICS (y > rare_high_threshold)
    # Test set
    error_mae_rare_high = evaluate_mae(
        model, X_test, y_test, 
        above_threshold=rare_high_threshold,
        device=device)
    results["mae_rare_high"] = error_mae_rare_high
    print(f'MAE rare high (y > {rare_high_threshold:.4f}) (test): {error_mae_rare_high:.4f}')
    
    # Training set
    error_mae_rare_high_train = evaluate_mae(
        model, X_train, y_train, 
        above_threshold=rare_high_threshold,
        device=device)
    results["train_mae_rare_high"] = error_mae_rare_high_train
    print(f'MAE rare high (y > {rare_high_threshold:.4f}) (train): {error_mae_rare_high_train:.4f}')
    
    # Test set
    error_pcc_rare_high = evaluate_pcc(
        model, X_test, y_test, 
        above_threshold=rare_high_threshold,
        device=device)
    results["pcc_rare_high"] = error_pcc_rare_high
    print(f'PCC rare high (y > {rare_high_threshold:.4f}) (test): {error_pcc_rare_high:.4f}')
    
    # Training set
    error_pcc_rare_high_train = evaluate_pcc(
        model, X_train, y_train, 
        above_threshold=rare_high_threshold,
        device=device)
    results["train_pcc_rare_high"] = error_pcc_rare_high_train
    print(f'PCC rare high (y > {rare_high_threshold:.4f}) (train): {error_pcc_rare_high_train:.4f}')
    
    return results


