import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from typing import Optional, List, Dict, Any, Union


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is useful for tracking metrics during training and validation,
    providing both the current value and running average.
    
    Attributes:
        name (str): Name of the metric being tracked
        fmt (str): Format string for displaying values
        val (float): Current value
        avg (float): Running average
        sum (float): Sum of all values
        count (int): Number of updates
    """
    
    def __init__(self, name: str, fmt: str = ':f') -> None:
        """Initialize the AverageMeter.
        
        Args:
            name: Name of the metric being tracked
            fmt: Format string for displaying values (default: ':f')
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all tracked values to their initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """Return formatted string representation of current and average values."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display progress of training/validation with multiple metrics.
    
    This class helps format and display training progress with multiple
    AverageMeter instances in a consistent format.
    
    Attributes:
        batch_fmtstr (str): Format string for batch numbers
        meters (List[AverageMeter]): List of meters to display
        prefix (str): Prefix string for display
    """
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "") -> None:
        """Initialize the ProgressMeter.
        
        Args:
            num_batches: Total number of batches for proper formatting
            meters: List of AverageMeter instances to display
            prefix: Prefix string to add before batch info (default: "")
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Display progress for the current batch.
        
        Args:
            batch: Current batch number
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        """Generate format string for batch numbers.
        
        Args:
            num_batches: Total number of batches
            
        Returns:
            Format string for displaying batch progress like [123/1000]
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question: str) -> bool:
    """Ask a yes/no question via input() and return their answer.
    
    Args:
        question: The question string to display to user
        
    Returns:
        True if user answered yes, False if no
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']  # Default to yes for empty input
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args: Any) -> None:
    """Prepare output folders for training, with option to overwrite existing.
    
    Creates necessary directories for storing model outputs. If directories
    already exist, prompts user for confirmation before overwriting.
    
    Args:
        args: Arguments object containing store_root, store_name, resume, 
              pretrained, and evaluate flags
              
    Raises:
        RuntimeError: If output folder exists and user chooses not to overwrite
    """
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    
    # Check if we need to handle existing folder
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    
    # Create folders if they don't exist        
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args: Any) -> None:
    """Adjust learning rate according to predefined schedule.
    
    Implements step decay learning rate schedule where learning rate
    is multiplied by 0.1 at specified milestone epochs.
    
    Args:
        optimizer: PyTorch optimizer to adjust
        epoch: Current epoch number
        args: Arguments object containing lr (base learning rate) and 
              schedule (list of milestone epochs)
    """
    lr = args.lr
    # Apply step decay at milestone epochs
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
        
    # Update learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args: Any, state: Dict[str, Any], is_best: bool, prefix: str = '') -> None:
    """Save model checkpoint to disk.
    
    Saves the current model state and optionally creates a copy as the best model
    if this checkpoint achieves the best performance so far.
    
    Args:
        args: Arguments object containing store_root and store_name for output path
        state: Dictionary containing model state, optimizer state, epoch, etc.
        is_best: Whether this checkpoint is the best performing so far
        prefix: Optional prefix for the best checkpoint filename (default: '')
    """
    # Save current checkpoint
    filename = f"{args.store_root}/{args.store_name}/ckpt.pth.tar"
    torch.save(state, filename)
    
    # Save best checkpoint if this is the best performance
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        best_filename = filename.replace('ckpt.pth.tar', f'{prefix}-ckpt.best.pth.tar')
        shutil.copyfile(filename, best_filename)


def calibrate_mean_var(matrix: torch.Tensor, m1: torch.Tensor, v1: torch.Tensor, 
                      m2: torch.Tensor, v2: torch.Tensor, 
                      clip_min: float = 0.1, clip_max: float = 10) -> torch.Tensor:
    """Calibrate features by adjusting their mean and variance.
    
    This function is used in Feature Distribution Smoothing (FDS) to adjust
    the distribution of features from one set of statistics (m1, v1) to 
    another (m2, v2). This helps smooth feature distributions across classes.
    
    Args:
        matrix: Input feature matrix to calibrate [batch_size, feature_dim]
        m1: Source mean values [feature_dim]
        v1: Source variance values [feature_dim] 
        v2: Target variance values [feature_dim]
        m2: Target mean values [feature_dim]
        clip_min: Minimum clipping value for variance scaling (default: 0.1)
        clip_max: Maximum clipping value for variance scaling (default: 10)
        
    Returns:
        Calibrated feature matrix with adjusted mean and variance
    """
    # Skip calibration if source variance is too small
    if torch.sum(v1) < 1e-10:
        return matrix
        
    # Handle case where some variances are zero
    if (v1 == 0.).any():
        valid = (v1 != 0.)  # Mask for non-zero variances
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        # Only calibrate features with non-zero source variance
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    # Standard calibration: (x - m1) * sqrt(v2/v1) + m2
    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel: str, ks: int, sigma: float) -> np.ndarray:
    """Generate kernel window for Label Distribution Smoothing (LDS).
    
    Creates a smoothing kernel that can be used to smooth label distributions
    across neighboring classes in imbalanced datasets. This helps with the
    long-tail distribution problem.
    
    Args:
        kernel: Type of kernel ('gaussian', 'triang', or 'laplace')
        ks: Kernel size (should be odd number)
        sigma: Kernel parameter (standard deviation for gaussian/laplace)
        
    Returns:
        Normalized kernel window as numpy array
        
    Raises:
        AssertionError: If kernel type is not supported
    """
    assert kernel in ['gaussian', 'triang', 'laplace'], f"Unsupported kernel: {kernel}"
    
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        # Create impulse response and apply Gaussian filter
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window = kernel_window / max(kernel_window)  # Normalize
        
    elif kernel == 'triang':
        # Triangular window
        kernel_window = triang(ks)
        
    else:  # laplace
        # Laplace (double exponential) kernel
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_values = [laplace(x) for x in np.arange(-half_ks, half_ks + 1)]
        kernel_window = np.array(kernel_values) / max(kernel_values)  # Normalize

    return kernel_window
