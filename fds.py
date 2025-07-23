import logging
from typing import Literal
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calibrate_mean_var

print = logging.info


class FDS(nn.Module):
    """
    Feature Distribution Smoothing (FDS) module.
    
    This module smooths feature distributions across different label buckets to address
    the long-tailed distribution problem in regression tasks. It maintains running statistics
    of features for each label bucket and applies smoothing to normalize feature distributions.
    """

    def __init__(self, 
                 feature_dim: int, 
                 bucket_num: int = 100, 
                 bucket_start: int = 0, 
                 start_update: int = 0, 
                 start_smooth: int = 1,
                 kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian', 
                 ks: int = 5, 
                 sigma: float = 2, 
                 momentum: float = 0.9):
        """
        Initialize the FDS module.
        
        Args:
            feature_dim: Dimension of input features
            bucket_num: Number of buckets for label discretization
            bucket_start: Starting bucket index (e.g., 0 for IMDB-WIKI, 3 for AgeDB)
            start_update: Epoch to start updating statistics
            start_smooth: Epoch to start smoothing features
            kernel: Kernel type for smoothing ('gaussian', 'triang', or 'laplace')
            ks: Kernel size (should be odd)
            sigma: Sigma parameter for gaussian/laplace kernel
            momentum: Momentum for running statistics updates
        """
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        # Register buffers for tracking statistics
        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel: Literal['gaussian', 'triang', 'laplace'], 
                           ks: int, 
                           sigma: float) -> torch.Tensor:
        """
        Create a smoothing kernel window.
        
        Args:
            kernel: Type of kernel ('gaussian', 'triang', or 'laplace')
            ks: Kernel size (should be odd)
            sigma: Sigma parameter for gaussian/laplace kernel
            
        Returns:
            Tensor containing the normalized kernel weights
        """
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        
        if kernel == 'gaussian':
            # Create a base kernel with a single peak in the middle
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            # Apply gaussian filter and normalize
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            # Create a triangular window and normalize
            kernel_window = triang(ks) / sum(triang(ks))
        else:  # laplace
            # Create a laplace distribution window and normalize
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self) -> None:
        """
        Update the statistics from the last epoch and compute smoothed statistics.
        
        This method copies current running statistics to last epoch buffers and
        applies kernel smoothing to create smoothed versions of mean and variance.
        """
        # Copy current running statistics to last epoch buffers
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        # Apply kernel smoothing to mean values across buckets
        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        
        # Apply kernel smoothing to variance values across buckets
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self) -> None:
        """Reset all running statistics to their initial values."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch: int) -> None:
        """
        Update the last epoch statistics if we've moved to a new epoch.
        
        Args:
            epoch: Current epoch number
        """
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features: torch.Tensor, labels: torch.Tensor, epoch: int) -> None:
        """
        Update running statistics for each label bucket based on current features.
        
        Args:
            features: Feature tensor of shape [batch_size, feature_dim]
            labels: Label tensor of shape [batch_size]
            epoch: Current epoch number
        """
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        # Process each unique label in the batch
        for label in torch.unique(labels):
            # Skip labels outside our bucket range
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
                
            # Handle edge cases (first and last buckets)
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]  # All features with labels <= bucket_start
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]  # All features with labels >= bucket_num-1
            else:
                curr_feats = features[labels == label]  # Features with exact label match
                
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            # Update sample count for this bucket
            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            
            # Calculate update factor (momentum or sample-based)
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            # No update on first epoch
            factor = 0 if epoch == self.start_update else factor
            
            # Update running statistics with new batch data
            bucket_idx = int(label - self.bucket_start)
            self.running_mean[bucket_idx] = \
                (1 - factor) * curr_mean + factor * self.running_mean[bucket_idx]
            self.running_var[bucket_idx] = \
                (1 - factor) * curr_var + factor * self.running_var[bucket_idx]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Apply feature distribution smoothing to input features.
        
        Args:
            features: Feature tensor of shape [batch_size, feature_dim]
            labels: Label tensor of shape [batch_size, 1]
            epoch: Current epoch number
            
        Returns:
            Smoothed feature tensor of the same shape as input features
        """
        # Skip smoothing if before start_smooth epoch
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)  # Remove singleton dimension
        
        # Process each unique label in the batch
        for label in torch.unique(labels):
            # Skip labels outside our bucket range
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
                
            bucket_idx = int(label - self.bucket_start)
            
            # Handle edge cases (first and last buckets)
            if label == self.bucket_start:
                # Calibrate features for all samples with labels <= bucket_start
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_epoch[bucket_idx],
                    self.running_var_last_epoch[bucket_idx],
                    self.smoothed_mean_last_epoch[bucket_idx],
                    self.smoothed_var_last_epoch[bucket_idx])
            elif label == self.bucket_num - 1:
                # Calibrate features for all samples with labels >= bucket_num-1
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_epoch[bucket_idx],
                    self.running_var_last_epoch[bucket_idx],
                    self.smoothed_mean_last_epoch[bucket_idx],
                    self.smoothed_var_last_epoch[bucket_idx])
            else:
                # Calibrate features for samples with exact label match
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_epoch[bucket_idx],
                    self.running_var_last_epoch[bucket_idx],
                    self.smoothed_mean_last_epoch[bucket_idx],
                    self.smoothed_var_last_epoch[bucket_idx])
        return features
