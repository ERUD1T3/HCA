# -*- coding: utf-8 -*-
# author: xhp

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple

# Content:
# -- Count2Class: Convert continuous count values to discrete class labels
# -- Class2Count: Convert discrete class labels to continuous count values  
# -- UEP_order: Uniform Error Partitioning for optimal class boundaries
# -- interval_divide: Divide value range into intervals for classification

# Note: class_indice should start from V_min and not include V_max
# interval_divide returns the interval number; if directly used, then class count is num+1


def Count2Class(count_map: Union[np.ndarray, torch.Tensor], 
                label_indice: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convert continuous count values to discrete class labels.
    
    This function maps continuous count/regression values to discrete class labels
    based on provided class boundaries (label_indice). This is essential for 
    hierarchical classification where we need to convert between continuous 
    age values and discrete age class buckets.
    
    Args:
        count_map: Input tensor/array of continuous count values to classify
        label_indice: Class boundary values that define the thresholds between classes.
                     Should NOT include the maximum boundary value (V_max)
                     
    Returns:
        Class labels as tensor/array, same type as input count_map
        Values range from 0 to len(label_indice) (inclusive)
        
    Example:
        >>> count_map = torch.tensor([1.5, 5.2, 12.8, 25.0])
        >>> label_indice = torch.tensor([3.0, 10.0, 20.0])  # Boundaries at 3, 10, 20
        >>> Count2Class(count_map, label_indice)
        tensor([0, 1, 2, 3])  # Classes: [0-3), [3-10), [10-20), [20+)
    """
    # Track input format for consistent output
    IF_ret_np = False
    if isinstance(count_map, np.ndarray):
        count_map = torch.from_numpy(count_map)
        IF_ret_np = True
        
    # Determine compute device (GPU if available and input is on GPU)
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (count_map.device.type == 'cuda') 

    cls_num = len(label_indice) + 1  # Number of classes = boundaries + 1
    cls_map = torch.zeros(count_map.size()).type(torch.LongTensor) 
    
    # Move to GPU if needed
    if IF_gpu:
        count_map, cls_map = count_map.cuda(), cls_map.cuda()
    
    # For each boundary, count how many values are >= that boundary
    # This creates a cumulative count that gives us the class index
    for i in range(len(label_indice)):
        if IF_gpu:
            cls_map = cls_map + (count_map >= label_indice[i].item()).long()
        else:
            cls_map = cls_map + (count_map >= label_indice[i].item()).long()
    
    # Ensure output device matches expected format
    if not IF_ret_gpu:
        cls_map = cls_map.cpu() 
    if IF_ret_np:
        cls_map = cls_map.cpu().numpy()
        
    return cls_map


def Class2Count(pre_cls: torch.Tensor, label2count: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert discrete class labels to continuous count values.
    
    This function maps discrete class predictions back to continuous count/regression
    values using the representative value for each class. This is the inverse operation
    of Count2Class and is used to convert classification outputs back to age estimates.
    
    Args:
        pre_cls: Predicted class labels [batch_size, num_classes] (logits) or 
                [batch_size, 1] (class indices). If logits, will take argmax.
        label2count: Representative continuous value for each class [num_classes].
                    Should contain exactly num_classes elements.
                    
    Returns:
        Continuous count values [batch_size, 1] with same device as input pre_cls
        
    Example:
        >>> pre_cls = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])  # Logits
        >>> label2count = torch.tensor([1.5, 6.5, 15.0])  # Class representatives  
        >>> Class2Count(pre_cls, label2count)
        tensor([[6.5], [1.5]])  # Converts to continuous values
    """
    # Ensure label2count is a float tensor
    if isinstance(label2count, np.ndarray):
        label2count = torch.from_numpy(label2count).float()
    label2count = label2count.squeeze()
    
    # Track device information for consistent output
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (pre_cls.device.type == 'cuda')  

    # Convert logits to class indices if needed
    if pre_cls.size()[1] > 1:  # Input is logits [batch_size, num_classes]
        pre_cls = pre_cls.max(dim=1, keepdim=True)[1].cpu().long()  # Take argmax
    
    ORI_SIZE = pre_cls.size()  # Store original shape

    # Use index_select to map class indices to their representative values
    pre_counts = torch.index_select(label2count.cuda(), 0, pre_cls.cuda().view(-1))
    pre_counts = pre_counts.view(ORI_SIZE)

    # Ensure output device matches input device
    if not IF_ret_gpu:
        pre_counts = pre_counts.cpu()

    return pre_counts


def UEP_border(patch_count: np.ndarray, vmin: float = 5e-3, vmax: float = 48.5, 
               num: int = 50, epslon: float = 50) -> Tuple[np.ndarray, float]:
    """Generate class boundaries using Uniform Error Partitioning (UEP).
    
    UEP creates class boundaries that try to equalize the prediction error
    across all classes by using binary search to find optimal partition points.
    This method is particularly useful for imbalanced datasets where simple
    uniform partitioning would create poor class boundaries.
    
    Args:
        patch_count: Array of count values from the dataset [N,]
        vmin: Minimum value to consider (values below this are excluded) 
        vmax: Maximum value to consider (values above this are excluded)
        num: Target number of intervals to create (actual may be fewer)
        epslon: Convergence threshold for binary search
        
    Returns:
        Tuple containing:
        - cls_border: Array of boundary values [num_boundaries,]  
        - pl: Final partition parameter from binary search
        
    Note:
        The algorithm may produce fewer than 'num' intervals if the data
        doesn't support that many meaningful partitions.
    """
    cls_border = np.zeros(num)
    
    # Remove values outside the range [vmin, vmax)
    del_idx = np.where(patch_count < vmin)
    patch_count = np.delete(patch_count, del_idx)
    del_idx = np.where(patch_count >= vmax)
    patch_count = np.delete(patch_count, del_idx)
    
    # Get unique values and their frequencies
    count_set, elem_num = np.unique(patch_count, return_counts=True)
    ind = np.argsort(count_set)  # Sort from low to high
    count_set, elem_num = count_set[ind], elem_num[ind]
    
    # Initialize binary search bounds
    L = 0  # Lower bound
    step = (vmax - vmin) / (num - 1)  # Initial uniform step size
    N1 = (patch_count < step).sum()
    H = step * N1  # Upper bound
    
    # Binary search for optimal partition parameter
    iter_i = 0
    while (H - L) > epslon or (len(P) != num - 1):
        pp = vmin  # Current partition point
        n = 0      # Cumulative count
        P = [vmin] # Partition boundaries
        pl = (L + H) / 2  # Current partition parameter
        
        # Create partitions based on current parameter
        for idx, value in enumerate(count_set):
            n = n + elem_num[idx].item()
            if (value - pp) * n > pl:  # Create new partition
                n = 0
                p = value
                P.append(value)
        
        # Adjust binary search bounds based on number of partitions found
        if len(P) >= num:
            L = pl  # Too many partitions, increase parameter
        else:
            if len(P) == num - 1:
                # Check if last segment needs partitioning
                if (vmax - p) * n > pl:
                    L = pl
                elif (vmax - p) * n < pl:
                    H = pl
                else:
                    H = L  # Converged
            else:
                H = pl  # Too few partitions, decrease parameter
                
        print(f'{iter_i}-th iteration with {len(P)}/{num-1} border points')
        iter_i = iter_i + 1
        
        cls_border = P
        # Add final boundary
        cls_border = np.concatenate((cls_border, [vmax]), axis=0)
        
    return cls_border, pl


def interval_divide(patch_count: np.ndarray, vmin: float, vmax: float, num: int = 50, 
                   cls_parse: str = 'mean', partition: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """Divide value range into intervals and compute class representatives.
    
    This function creates class boundaries and computes representative values
    for each class. It supports multiple partitioning strategies and methods
    for computing class representatives.
    
    Args:
        patch_count: Array of count values from dataset [N,]
        vmin: Minimum value for partitioning
        vmax: Maximum value for partitioning  
        num: Number of intervals to create
        cls_parse: Method for computing class representatives ('mean' or 'median')
                  - 'mean': Use actual mean of values in each class
                  - 'median': Use midpoint of class boundaries
        partition: Partitioning strategy ('linear', 'log', or 'uep')
                  - 'linear': Uniform spacing in linear scale
                  - 'log': Uniform spacing in log scale
                  - 'uep': Uniform Error Partitioning (data-adaptive)
                  
    Returns:
        Tuple containing:
        - cls_border: Class boundary values [num,] 
        - cls2value: Representative value for each class [num,]
                    Index 0 corresponds to values < vmin, indices 1+ to actual classes
                    
    Example:
        >>> counts = np.array([1, 5, 10, 15, 20, 25])
        >>> borders, values = interval_divide(counts, 0, 30, 3, 'mean', 'linear')
        >>> # Creates 3 intervals: [0,10), [10,20), [20,30]
        >>> # values[0] = 0 (for < vmin), values[1:] = mean of each interval
    """
    # Create class boundaries based on partitioning strategy
    if partition == 'linear':
        # Uniform linear spacing
        step = (vmax - vmin) / (num - 1)
        cls_border = np.arange(vmin, vmax, step)
        cls_border = np.concatenate((cls_border, [vmax]), axis=0)
        
    elif partition == 'log':
        # Uniform logarithmic spacing  
        step = (np.log(vmax) - np.log(vmin)) / (num - 1)
        cls_border = np.arange(np.log(vmin), np.log(vmax), step)
        cls_border = np.exp(cls_border)
        cls_border = np.concatenate((cls_border, [vmax]), axis=0)
        
    elif partition == 'uep':
        # Uniform Error Partitioning (data-adaptive)
        cls_border, pl = UEP_border(patch_count, vmin=5e-3, vmax=vmax, num=num, epslon=50)
    
    # Compute representative values for each class
    cls2value = np.zeros(num)
    
    if cls_parse == 'median':
        # Use midpoint of class boundaries
        cls2value[1:] = (cls_border[:-1] + cls_border[1:]) / 2
        
    elif cls_parse == 'mean':
        # Use actual mean of values in each class
        # First classify all patch_count values
        patch_class = Count2Class(patch_count, cls_border[:-1])  # Exclude vmax
        
        for ci in range(1, num):  # Skip class 0 (< vmin)
            tmp_mask = (patch_class == ci)
            tmp_num = tmp_mask.sum().item()
            
            if tmp_num < 1:
                # No samples in this class, use midpoint
                cls2value[ci] = (cls_border[ci-1] + cls_border[ci]) / 2
            else:
                # Use actual mean of samples in this class
                tmp_c2v = (patch_count * tmp_mask).sum() / tmp_num
                cls2value[ci] = tmp_c2v.item()
    else:
        print(f'No class to count method as {cls_parse}')

    return cls_border, cls2value