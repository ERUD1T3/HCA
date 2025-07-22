import torch
import numpy as np
from typing import List, Tuple, Dict, Union
from cls_funcs import Count2Class


def div2func(interval: List[float], count_dict: Dict[int, float]) -> List[List[float]]:
    """Divide an interval into two sub-intervals based on sample distribution.
    
    This function attempts to split a given interval [vmin, vmax) into two 
    approximately equal-sample sub-intervals. It finds the optimal split point
    that balances the number of samples on each side.
    
    Args:
        interval: Two-element list [vmin, vmax] defining the interval to split
        count_dict: Dictionary mapping values to their sample counts
        
    Returns:
        List of intervals: either [original_interval] if no good split found,
        or [[vmin, vmed], [vmed, vmax]] where vmed is the split point
        
    Example:
        >>> interval = [0, 10]
        >>> counts = {1: 100, 3: 200, 7: 150, 9: 50}
        >>> div2func(interval, counts)
        [[0, 5], [5, 10]]  # Split near median sample position
    """
    vmin, vmax = interval[0], interval[1]
    
    # Find all values and counts within this interval [vmin, vmax)
    in_key = []
    in_value = []
    keylist = list(count_dict.keys())
    keylist.sort(reverse=False)  # Sort in ascending order
    
    for tkey in keylist:
        if vmin <= tkey < vmax:  # Within interval bounds
            if count_dict[tkey] > 0:  # Has samples
                in_key.append(tkey)
                in_value.append(count_dict[tkey])
    
    # Need at least 2 distinct values to split
    if len(in_key) < 2:
        return [interval]
    
    # Find split point that approximately halves the sample count
    in_value = np.array(in_value)
    in_sum = np.cumsum(in_value)  # Cumulative sample counts
    half_sum = in_sum[-1] / 2.0   # Target: half of total samples
    
    # Find the split index closest to half the samples
    err = np.abs(in_sum - half_sum)
    minidx = np.argmin(err)
    minidx = minidx + 1  # Split after this index
    
    # Ensure split is not at the boundaries (keep at least 1 key per side)
    minidx = max(1, min(minidx, len(in_key) - 1))
    
    vmed = in_key[minidx]  # Split point
    
    split_list = [[vmin, vmed], [vmed, vmax]]
    return split_list


def level_split(interval: List[float], count_dict: Dict[int, float], slevel: int, 
               if_age: bool = False, ob_vmax: Union[float, None] = None) -> Tuple[List[List[List[float]]], List[List[float]], List[List[float]]]:
    """Recursively split interval into hierarchical levels for multi-head classification.
    
    This function creates a hierarchical partitioning of the value space for 
    multi-head classification. At each level, intervals are split approximately
    in half based on sample distribution, creating a binary tree-like structure.
    
    Args:
        interval: Initial interval [vmin, vmax] to partition
        count_dict: Dictionary mapping values to sample counts  
        slevel: Number of hierarchical levels to create
        if_age: Whether this is age data (affects class index calculation)
        ob_vmax: Observed maximum value for proper class representative calculation
        
    Returns:
        Tuple containing:
        - sinterval: List of intervals for each level [level][interval_idx][vmin, vmax]
        - sindice: List of class boundary indices for each level [level][boundary_idx]  
        - sclass2count: List of class representatives for each level [level][class_idx]
        
    Example:
        >>> interval = [0, 120]
        >>> counts = {i: np.random.randint(1, 100) for i in range(120)}
        >>> sinterval, sindice, sclass2count = level_split(interval, counts, 3, if_age=True)
        >>> # Creates 3 levels of hierarchical partitioning
    """
    sinterval = []
    tmp_level_split = [interval]
    
    # Create hierarchical splits level by level
    for si in range(slevel):
        sinterval.append([])
        
        # Split each interval from the previous level
        for tmp_interval in tmp_level_split:
            tmp_split = div2func(tmp_interval, count_dict)
            sinterval[-1] = sinterval[-1] + tmp_split
            
        tmp_level_split = sinterval[-1]
        
        # Check if we can't split further (same number of intervals as previous level)
        if len(sinterval) >= 2 and len(sinterval[-1]) == len(sinterval[-2]):
            print(f"Can only split {si} levels.")
            break
    
    # Convert interval splits to class indices and class representatives
    sindice = []      # Class boundary indices for each level
    sclass2count = [] # Representative values for each class at each level
    
    for si in range(len(sinterval)):
        sindice.append([])
        sclass2count.append([])
        tmp_intervallist = sinterval[si]
        
        for idx in range(len(tmp_intervallist)):
            # Compute class representative (midpoint of interval)
            tmp_class2count = (tmp_intervallist[idx][0] + tmp_intervallist[idx][1]) / 2.0
            
            # Use observed max for the last interval if provided
            if ob_vmax is not None:
                if idx == len(tmp_intervallist) - 1:
                    tmp_class2count = (tmp_intervallist[idx][0] + ob_vmax) / 2.0
            
            # Adjust for age data (shift by 0.5)
            if if_age:
                tmp_class2count -= 0.5
                
            sclass2count[-1].append(tmp_class2count)
            
            # Create class boundary indices (skip first interval)
            if idx == 0:
                continue
                
            if if_age:
                sindice[-1].append(tmp_intervallist[idx][0] - 0.5)
            else:
                sindice[-1].append(tmp_intervallist[idx][0])
    
    return sinterval, sindice, sclass2count


def get_mh_weight(count_dict: Dict[int, float], sindice: List[List[float]]) -> List[List[float]]:
    """Compute balanced class weights for multi-head hierarchical classification.
    
    This function computes inverse frequency weights for each class at each
    hierarchical level. The weights help balance the loss contributions across
    classes with different sample frequencies.
    
    Args:
        count_dict: Dictionary mapping values to their sample counts
        sindice: List of class boundary indices for each hierarchical level
                from level_split function
                
    Returns:
        List of normalized weights for each level [level][class_idx]
        Weights are normalized so they sum to 1.0 at each level
        
    Example:
        >>> counts = {0: 100, 5: 50, 10: 200, 15: 25}
        >>> indices = [[7.5], [3.5, 12.5]]  # 2 levels with boundaries
        >>> weights = get_mh_weight(counts, indices)
        >>> # Returns normalized inverse frequency weights for each level
    """
    # Convert count dictionary to tensors for efficient computation
    allkey = list(count_dict.keys())
    allkey = torch.Tensor(allkey).float()
    allvalue = list(count_dict.values())
    allvalue = torch.Tensor(allvalue).float()
    
    sweight = []
    
    # Compute weights for each hierarchical level
    for hi in range(len(sindice)):
        sweight.append([])
        
        # Classify all values according to current level's boundaries
        allclass = Count2Class(allkey, torch.Tensor(sindice[hi]))
        tmp_cnum = len(sindice[hi]) + 1  # Number of classes = boundaries + 1
        
        # Compute total sample count for each class
        for ci in range(tmp_cnum):
            # Sum counts for all values belonging to this class
            tmp_count = ((allclass == ci).float() * allvalue).sum().item()
            
            # Compute inverse frequency weight (avoid division by zero)
            tmp_weight = 1.0 / tmp_count if tmp_count > 1e-6 else 0.0
            sweight[-1].append(tmp_weight)
            
        # Normalize weights to sum to 1.0
        tmp_norm = sum(sweight[-1])
        if tmp_norm > 0:
            for ci in range(tmp_cnum):
                sweight[-1][ci] /= tmp_norm
    
    return sweight


if __name__ == '__main__':
    """Example usage and testing of the hierarchical splitting functions."""
    # Example setup
    interval = [0, 9]
    count_dict = {0: 100, 1: 200, 2: 400, 3: 200, 4: 200, 5: 100, 6: 100, 7: 50, 8: 50}
    
    # Test basic interval splitting
    split_list = div2func(interval, count_dict)
    print(f"Basic split: {split_list}")
    
    # Test hierarchical splitting
    SLEVEL = 5
    sinterval = []
    tmp_split = [interval]
    
    for si in range(SLEVEL):
        sinterval.append([])
        for tmp_interval in tmp_split:
            tmp_split_result = div2func(tmp_interval, count_dict)
            sinterval[-1] = sinterval[-1] + tmp_split_result
        tmp_split = sinterval[-1]
        
    # Print results for each level
    for level in range(min(SLEVEL, len(sinterval))):
        print(f"Level {level}: {sinterval[level]}")
    
    # Test full level_split function
    slevel = 5
    sinterval, sindice, sclass2count = level_split(interval, count_dict, slevel, if_age=True, ob_vmax=8)
    
    print("\nFull level_split results:")
    for si in range(len(sinterval)):
        print(f'Level {si} intervals: {sinterval[si]}')
        
    for si in range(len(sindice)):
        print(f'Level {si} indices: {sindice[si]}')
        
    for si in range(len(sclass2count)):
        print(f'Level {si} class2count: {sclass2count[si]}')
        
    # Test weight computation
    weights = get_mh_weight(count_dict, sindice)
    for si in range(len(weights)):
        print(f'Level {si} weights: {weights[si]}')