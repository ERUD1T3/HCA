import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch

# Updated imports for tabular data and MLP
from tab_ds import TabDS, load_tabular_splits, set_seed
from mlp import create_mlp
from utils import *
from mh_utils import level_split, get_mh_weight
from scipy.stats import *

# Import dataset-specific metrics
from metrics import (
    evaluate_sep_metrics, evaluate_ed_metrics, evaluate_onp_metrics,
    evaluate_sarcos_metrics, evaluate_bf_metrics, evaluate_asc_metrics
)

import os
os.environ["KMP_WARNINGS"] = "FALSE"

# Command line argument parser with comprehensive options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset and data parameters
parser.add_argument('--dataset', type=str, required=True,
                    choices=['sep', 'sarcos', 'onp', 'bf', 'asc', 'ed'],
                    help='Name of the tabular dataset to use.')
parser.add_argument('--data_dir', type=str, default='./data', help='Root directory containing dataset subfolders')
parser.add_argument('--train_split_name', type=str, default='training', help='Name for the training data file/folder')
parser.add_argument('--val_split_name', type=str, default='validation', help='Name for the validation data file/folder')
parser.add_argument('--test_split_name', type=str, default='testing', help='Name for the test data file/folder')

# MLP model parameters
parser.add_argument('--model', type=str, default='mlp', choices=['mlp'], help='model name')
parser.add_argument('--mlp_hiddens', type=int, nargs='+', default=[100, 100, 100], help='MLP hidden layer sizes')
parser.add_argument('--mlp_embed_dim', type=int, default=128, help='MLP embedding dimension')
parser.add_argument('--mlp_skip_layers', type=int, default=1, help='MLP skip connection frequency')
parser.add_argument('--mlp_skip_repr', action='store_true', default=True, help='MLP merge skip into final representation')
parser.add_argument('--mlp_dropout', type=float, default=0.1, help='MLP dropout rate')

# LDS (Label Distribution Smoothing) parameters
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

# FDS (Feature Distribution Smoothing) parameters
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')

# Bucket configuration for hierarchical classification
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS and hierarchical classification')
parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# Re-weighting scheme
parser.add_argument('--reweight', type=str, default='sqrt_inv', choices=['none', 'sqrt_inv', 'inverse'], 
                    help='cost-sensitive reweighting scheme')

# Training parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers used in data loading')
parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='list of random seeds for multiple trials')

# Model and checkpoint parameters  
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--log_dir', type=str, default='analy_log')
parser.add_argument('--log_name', type=str, default='hca_test.log')

# Multi-head hierarchical classification parameters
parser.add_argument('--head_num', type=int, default=6, help='the number of hierarchical classification heads')
parser.add_argument('--fc_lnum', type=int, default=1, help='number of FC layers in hierarchical heads (1 or 2)')
parser.add_argument('--s2fc_lnum', type=int, default=2, help='number of FC layers in adjustment head (1, 2, or 3)')
parser.add_argument('--head_detach', action='store_true', default=True, help='whether to detach gradients for hierarchical heads')

# GPU settings
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')

args, unknown = parser.parse_known_args()
args.start_epoch, args.best_loss = 0, 1e5


def run_trial(trial_seed: int) -> Dict[str, Any]:
    """Run a single evaluation trial with the given seed.
    
    Args:
        trial_seed: Random seed for this trial
        
    Returns:
        Dictionary containing evaluation results for this trial
    """
    # Set seed for reproducibility
    set_seed(trial_seed)
    print(f"\n{'='*60}")
    print(f"Running trial with seed: {trial_seed}")
    print(f"{'='*60}")
    
    # Load tabular data
    print('=====> Preparing data...')
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_splits(
            args.dataset, args.data_dir, args.train_split_name,
            args.val_split_name, args.test_split_name, trial_seed
        )
        print(f"Data loaded. Train: {X_train.shape}/{y_train.shape}, Val: {X_val.shape}/{y_val.shape}, Test: {X_test.shape}/{y_test.shape}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return {}

    # Get input dimension and target range for hierarchical setup
    input_dim = X_train.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # Create hierarchical class boundaries based on target value range
    min_target = float(np.min(y_train))
    max_target = float(np.max(y_train))
    target_range = max_target - min_target
    print(f"Target range: [{min_target:.3f}, {max_target:.3f}]")
    
    # Create global class boundaries (for main classifier)
    # Use bucket_num bins across the target range
    global_class_indice = np.linspace(min_target + target_range/args.bucket_num, 
                                    max_target - target_range/args.bucket_num, 
                                    args.bucket_num - 1)
    global_class2count = np.linspace(min_target, max_target, args.bucket_num)
    print(f"Created {len(global_class_indice)} class boundaries and {len(global_class2count)} class representatives")

    # Create test dataset and dataloader
    test_dataset = TabDS(X=X_test, y=y_test, reweight='none', lds=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Test data size: {len(test_dataset)}")
    
    # Compute class distribution for hierarchical splitting
    # Discretize continuous targets into bins for counting
    bins = np.linspace(min_target, max_target, args.bucket_num + 1)
    bin_indices = np.digitize(y_train, bins) - 1
    bin_indices = np.clip(bin_indices, 0, args.bucket_num - 1)
    
    # Count samples per bin
    global_count_dict = {}
    for i in range(args.bucket_num):
        global_count_dict[i] = np.sum(bin_indices == i)
        
    # Use square root for more balanced splitting
    global_count_dict_sqrt = {}
    for i in range(args.bucket_num):
        global_count_dict_sqrt[i] = np.sqrt(global_count_dict[i] + 1)  # +1 to avoid zero
        
    print(f"Bin counts: min={min(global_count_dict.values())}, max={max(global_count_dict.values())}, "
          f"total={sum(global_count_dict.values())}")
        
    # Create hierarchical class boundaries and weights
    # Adapt the level_split to work with continuous targets
    bin_representatives = (bins[:-1] + bins[1:]) / 2  # Midpoint of each bin
    
    # Create hierarchical levels using the binned representation
    sinterval, sindice, sclass2count = level_split(
        [min_target, max_target + 0.01], global_count_dict_sqrt, args.head_num, 
        if_age=False, ob_vmax=max_target
    )    
    sweight = get_mh_weight(global_count_dict, sindice)
    
    print(f"Created {len(sindice)} hierarchical levels:")
    for i, (indice, class2count) in enumerate(zip(sindice, sclass2count)):
        print(f"  Level {i}: {len(indice)} boundaries, {len(class2count)} classes")
    
    # Build model
    print('=====> Building model...')
    model = create_mlp(
        input_dim=input_dim,
        output_dim=1,
        hiddens=args.mlp_hiddens, 
        embed_dim=args.mlp_embed_dim,
        skipped_layers=args.mlp_skip_layers, 
        skip_repr=args.mlp_skip_repr,
        dropout=args.mlp_dropout,
        fds=args.fds, 
        bucket_num=args.bucket_num, 
        bucket_start=args.bucket_start,
        start_update=args.start_update, 
        start_smooth=args.start_smooth,
        kernel=args.fds_kernel, 
        ks=args.fds_ks, 
        sigma=args.fds_sigma, 
        momentum=args.fds_mmt,
        class_indice=global_class_indice, 
        class2count=global_class2count,
        head_class_indice=sindice, 
        head_class2count=sclass2count,
        head_weight=sweight, 
        cmax=max_target, 
        fc_lnum=args.fc_lnum,
        s2fc_lnum=args.s2fc_lnum, 
        head_detach=args.head_detach
    )
    
    if args.gpu is not None:
        model = torch.nn.DataParallel(model).cuda(args.gpu)
        print(f"Model moved to GPU {args.gpu}")
    else:
        model = model.cpu()
        print("Model running on CPU")
    
    # Load pre-trained weights if specified
    if args.resume:
        if not os.path.isfile(args.resume):
            if not ('.pth.tar' in args.resume):
                args.resume = os.path.join(args.resume, 'ckpt.best.pth.tar')
        
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            if args.gpu is not None:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
                
            # Handle potential key mismatches between saved state and current model
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            
            if len(pretrained_dict) != len(checkpoint['state_dict']):
                print(f"Warning: Loaded {len(pretrained_dict)}/{len(checkpoint['state_dict'])} parameters")
                
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"Warning: No checkpoint found at '{args.resume}', running with random weights")
    else:
        print("No checkpoint specified, running with random weights")
    
    # Evaluate model
    print("Now testing on the test set")
    results = validate(test_loader, model, train_labels=y_train, X_train=X_train, prefix=f'Test-Seed{trial_seed}')
    results['seed'] = trial_seed
    return results


def main() -> None:
    """Main function that coordinates multiple evaluation trials.
    
    This function:
    1. Sets up logging 
    2. Runs evaluation trials for each specified seed
    3. Aggregates and reports results across all trials
    """
    # Setup logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, args.log_name)),
            logging.StreamHandler()
        ])
    print = logging.info
    print(f"Args: {args}")
    print(f"Running HCA evaluation on {args.dataset} dataset with {len(args.seeds)} seeds: {args.seeds}")
    
    # Run trials for each seed
    all_results = []
    
    for seed in args.seeds:
        try:
            results = run_trial(seed)
            if results:  # Only add if results are not empty
                all_results.append(results)
                print(f"\nSeed {seed} completed successfully")
            else:
                print(f"\nSeed {seed} failed")
        except Exception as e:
            print(f"\nSeed {seed} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results across all seeds
    if all_results:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY ACROSS {len(all_results)} SUCCESSFUL TRIALS")
        print(f"{'='*80}")
        
        # Get all metric names (excluding 'seed')
        metric_names = [key for key in all_results[0].keys() if key != 'seed']
        
        # Compute statistics across seeds for each metric
        for metric_name in metric_names:
            values = [result[metric_name] for result in all_results if metric_name in result]
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                print(f"{metric_name}:")
                print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
                print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        
        print(f"\nSuccessful seeds: {[result['seed'] for result in all_results]}")
        if len(all_results) < len(args.seeds):
            failed_seeds = [seed for seed in args.seeds if seed not in [result['seed'] for result in all_results]]
            print(f"Failed seeds: {failed_seeds}")
    else:
        print("\nNo successful trials completed!")
    
    return


def validate(val_loader: DataLoader, model: nn.Module, train_labels: Optional[np.ndarray] = None, 
            X_train: Optional[np.ndarray] = None, prefix: str = 'Val') -> Dict[str, Any]:
    """Validate the model on the given dataset using dataset-specific metrics.
    
    This function evaluates the model using both hierarchical classification strategies
    and dataset-specific metrics for comprehensive evaluation.
    
    Args:
        val_loader: DataLoader for validation/test data
        model: The trained model to evaluate
        train_labels: Training labels for computing shot-based metrics
        X_train: Training features for dataset-specific metrics
        prefix: Prefix for logging messages (default: 'Val')
        
    Returns:
        Dictionary containing evaluation results including dataset-specific metrics
    """
    # Initialize meters for tracking metrics
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    # Define loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    # Set model to evaluation mode
    model.eval()
    
    # Collect all data for dataset-specific metrics
    all_inputs = []
    all_targets = []
    all_predictions = []

    # Get head number from model (handle DataParallel wrapper)
    if hasattr(model, 'module'):
        head_num = model.module.head_num
        model_ref = model.module
    else:
        head_num = model.head_num
        model_ref = model

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            if args.gpu is not None:
                inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
            else:
                inputs, targets = inputs.cpu(), targets.cpu()
            
            # Forward pass through model - MLP returns dictionary in hierarchical mode
            allpre = model(inputs)
            
            # Handle different return types based on hierarchical classification mode
            if isinstance(allpre, dict):
                # Hierarchical classification mode - use main classifier
                if 'x' in allpre:
                    outputs = allpre['x']
                    outputs = model_ref.Class2Count(outputs)  # Convert to continuous values
                else:
                    raise ValueError("No main classifier output 'x' found in model predictions")
            else:
                # Regular regression mode (outputs are continuous values directly)
                outputs = allpre[0] if isinstance(allpre, tuple) else allpre
                outputs = outputs.squeeze(-1) if outputs.ndim > 1 and outputs.shape[-1] == 1 else outputs
        
            # Flatten outputs and targets for consistent shapes
            outputs_flat = outputs.view(-1) if outputs.ndim > 1 else outputs
            targets_flat = targets.view(-1) if targets.ndim > 1 else targets
            
            # Store for dataset-specific metrics
            all_inputs.append(inputs.detach().cpu().numpy())
            all_targets.append(targets_flat.detach().cpu().numpy())
            all_predictions.append(outputs_flat.detach().cpu().numpy())

            # Compute losses for progress tracking
            loss_mse = criterion_mse(outputs_flat, targets_flat)
            loss_l1 = criterion_l1(outputs_flat, targets_flat)
            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            # Update timing
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        
        # Concatenate all data for metrics computation
        X_test = np.vstack(all_inputs)
        y_test = np.concatenate(all_targets)
        y_pred = np.concatenate(all_predictions)
        
        # Compute overall statistics
        overall_mse = np.mean((y_pred - y_test) ** 2)
        overall_mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"\n{prefix} Results:")
        print(f"Overall MSE: {overall_mse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Target range: [{np.min(y_test):.3f}, {np.max(y_test):.3f}]")
        print(f"Prediction range: [{np.min(y_pred):.3f}, {np.max(y_pred):.3f}]")
        
        # Evaluate dataset-specific metrics
        result_dict = {'overall_mse': overall_mse, 'overall_mae': overall_mae}
        
        if train_labels is not None:
            print("\nEvaluating dataset-specific metrics...")
            
            # Get device for metrics computation
            if args.gpu is not None:
                device = torch.device(f'cuda:{args.gpu}')
            else:
                device = torch.device('cpu')
            
            # Use appropriate metrics function based on dataset
            try:
                if args.dataset == 'sep':
                    metrics = evaluate_sep_metrics(
                        model, X_train, train_labels, X_test, y_test, 
                        sep_threshold=np.log(10), device=device
                    )
                elif args.dataset == 'ed':
                    metrics = evaluate_ed_metrics(
                        model, X_train, train_labels, X_test, y_test,
                        mae_plus_threshold=0.5, device=device
                    )
                elif args.dataset == 'onp':
                    metrics = evaluate_onp_metrics(
                        model, X_train, train_labels, X_test, y_test, 
                        rare_low_threshold=np.log10(350),
                        rare_high_threshold=np.log10(35000),
                        device=device
                    )
                elif args.dataset == 'sarcos':
                    metrics = evaluate_sarcos_metrics(
                        model, X_train, train_labels, X_test, y_test,
                        lower_threshold=-0.5, upper_threshold=0.5,
                        device=device
                    )
                elif args.dataset == 'bf':
                    metrics = evaluate_bf_metrics(
                        model, X_train, train_labels, X_test, y_test,
                        freq_threshold=np.log10(4), rare_threshold=np.log10(40),
                        device=device
                    )
                elif args.dataset == 'asc':
                    metrics = evaluate_asc_metrics(
                        model, X_train, train_labels, X_test, y_test,
                        rare_low_threshold=np.log10(200),
                        rare_high_threshold=np.log10(20000),
                        device=device
                    )
                else:
                    print(f"No specific metrics defined for dataset: {args.dataset}")
                    metrics = {}
                
                # Merge dataset-specific metrics with overall results
                result_dict.update(metrics)
                
                # Print metrics in a formatted way
                print("\n===== Dataset-Specific Metrics =====")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                    
            except Exception as e:
                print(f"Warning: Dataset-specific metrics evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No training labels provided, skipping dataset-specific metrics")
        
    return result_dict


if __name__ == '__main__':
    """
    Example usage:
    
    # Basic hierarchical classification on sep dataset
    python main.py --dataset sep --data_dir ./data --head_num 4 --gpu 0
    
    # With pre-trained model
    python main.py --dataset sep --data_dir ./data --head_num 4 --resume ./checkpoint/model.pth.tar --gpu 0
    
    # With FDS and LDS enabled
    python main.py --dataset sep --data_dir ./data --head_num 4 --fds --lds --lds_ks 5 --lds_sigma 1.0 --gpu 0
    
    # Different MLP architecture
    python main.py --dataset sarcos --data_dir ./data --mlp_hiddens 256 128 64 --mlp_embed_dim 256 --head_num 6
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
