from calendar import c
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
from fds import FDS

# Classification related functions
from cls_funcs import Count2Class, Class2Count

print = logging.info


def myJS_div(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two logit distributions.
    
    JS divergence is a symmetric measure of similarity between probability distributions.
    It's computed as the average of two KL divergences from each distribution to their average.
    
    Args:
        logit1: First set of logits [batch_size, num_classes]
        logit2: Second set of logits [batch_size, num_classes]
        
    Returns:
        JS divergence values [batch_size, 1]
    """
    net_1_probs = F.softmax(logit1, dim=1)
    net_2_probs = F.softmax(logit2, dim=1)

    m = 0.5 * (net_1_probs + net_2_probs)  # Average distribution
    loss = 0.0
    # JS = 0.5 * (KL(P||M) + KL(Q||M))
    loss += F.kl_div(F.log_softmax(logit1, dim=1), m, reduction="none", log_target=False).sum(dim=1, keepdim=True) 
    loss += F.kl_div(F.log_softmax(logit2, dim=1), m, reduction="none", log_target=False).sum(dim=1, keepdim=True) 
    return (0.5 * loss)


def myKL_div(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """Compute Kullback-Leibler divergence from logit1 to logit2.
    
    KL divergence measures how one probability distribution diverges from another.
    KL(P||Q) where P is derived from logit1 and Q from logit2.
    
    Args:
        logit1: Input logits [batch_size, num_classes] 
        logit2: Target logits [batch_size, num_classes]
        
    Returns:
        KL divergence values [batch_size, 1]
    """
    loss = F.kl_div(
        F.log_softmax(logit1, dim=1), 
        F.softmax(logit2, dim=1), 
        reduction="none", 
        log_target=False
    ).sum(dim=1, keepdim=True)
    return loss


def myKL_div_prob(prob1: torch.Tensor, prob2: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two probability distributions.
    
    Computes KL(prob1 || prob2) with numerical stability through epsilon clamping.
    
    Args:
        prob1: Input probability distribution [batch_size, num_classes]
        prob2: Target probability distribution [batch_size, num_classes]
        
    Returns:
        KL divergence values [batch_size, 1]
    """
    EPS = 1e-6
    # Clamp probabilities to avoid log(0)
    prob1 = torch.where(prob1 >= EPS, prob1, prob1 + EPS)
    prob2 = torch.where(prob2 >= EPS, prob2, prob2 + EPS)
    
    loss = F.kl_div(
        torch.log(prob1), 
        torch.log(prob2), 
        reduction="none", 
        log_target=True
    ).sum(dim=1, keepdim=True)
    return loss


def myJS_div_prob(net_1_probs: torch.Tensor, net_2_probs: torch.Tensor) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two probability distributions.
    
    More numerically stable version that works directly with probability distributions.
    
    Args:
        net_1_probs: First probability distribution [batch_size, num_classes]
        net_2_probs: Second probability distribution [batch_size, num_classes]
        
    Returns:
        JS divergence values [batch_size, 1]
    """
    EPS = 1e-6
    # Clamp probabilities for numerical stability
    net_1_probs = torch.where(net_1_probs >= EPS, net_1_probs, net_1_probs + EPS)
    net_2_probs = torch.where(net_2_probs >= EPS, net_2_probs, net_2_probs + EPS)
    
    m = 0.5 * (net_1_probs + net_2_probs)  # Average distribution
    loss = 0.0
    loss += F.kl_div(torch.log(net_1_probs), m, reduction="none", log_target=False).sum(dim=1, keepdim=True) 
    loss += F.kl_div(torch.log(net_2_probs), m, reduction="none", log_target=False).sum(dim=1, keepdim=True) 

    return (0.5 * loss)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding for ResNet blocks."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34.
    
    Implements the basic residual block with two 3x3 convolutions.
    
    Attributes:
        expansion (int): Output channel expansion factor (always 1 for BasicBlock)
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None) -> None:
        """Initialize BasicBlock.
        
        Args:
            inplanes: Number of input channels
            planes: Number of output channels (before expansion)
            stride: Stride for the first convolution (default: 1)
            downsample: Downsample module for skip connection (default: None)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the basic block."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual  # Skip connection
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block for ResNet-50, ResNet-101, and ResNet-152.
    
    Implements the bottleneck residual block with 1x1-3x3-1x1 convolutions
    that reduces computational cost while maintaining representational power.
    
    Attributes:
        expansion (int): Output channel expansion factor (always 4 for Bottleneck)
    """
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None) -> None:
        """Initialize Bottleneck block.
        
        Args:
            inplanes: Number of input channels
            planes: Number of intermediate channels
            stride: Stride for the 3x3 convolution (default: 1)
            downsample: Downsample module for skip connection (default: None)
        """
        super(Bottleneck, self).__init__()
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv (main computation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck block."""
        residual = x
        
        # 1x1 conv down
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv up
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply downsampling to residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual  # Skip connection
        out = self.relu(out)
        
        return out


class ResNet_cls_mh(nn.Module):
    """ResNet with Multi-Head Hierarchical Classification for Age Estimation.
    
    This class implements a ResNet backbone with multiple hierarchical classification heads
    for age estimation. It supports Feature Distribution Smoothing (FDS) to handle 
    imbalanced datasets and includes multiple prediction heads at different hierarchical levels.
    
    Key Features:
    - ResNet backbone (18/34/50/101/152) for feature extraction
    - Multiple hierarchical classification heads for coarse-to-fine prediction
    - Feature Distribution Smoothing (FDS) support for imbalanced learning
    - Flexible head architectures (1-3 fully connected layers)
    - Class-to-count and count-to-class conversion utilities
    - Hierarchical prediction combination strategies
    
    Attributes:
        inplanes (int): Current number of input planes for residual blocks
        class_indice (torch.Tensor): Class boundary indices for main classifier
        class2count (torch.Tensor): Representative values for each class
        cnum (int): Total number of classes for main classifier
        linear (nn.Linear): Main classification head
        FDS (FDS): Feature Distribution Smoothing module (if enabled)
        fds (bool): Whether FDS is enabled
        start_smooth (int): Epoch to start FDS smoothing
        use_dropout (bool): Whether dropout is enabled
        dropout (nn.Dropout): Dropout layer (if enabled)
        head_num (int): Number of hierarchical heads
        head_class_indice (List): Class indices for each hierarchical head
        head_class2count (List): Class representatives for each hierarchical head
        head_weight (List): Class weights for each hierarchical head
        fc_lnum (int): Number of layers in hierarchical heads
        s2fc_lnum (int): Number of layers in adjustment head
        cmax (int): Maximum age value
        head_detach (bool): Whether to detach gradients for hierarchical heads
    """

    def __init__(
        self, 
        block, 
        layers: List[int], 
        fds: bool, 
        bucket_num: int, 
        bucket_start: int, 
        start_update: int, 
        start_smooth: int, 
        kernel: str, 
        ks: int, 
        sigma: float, 
        momentum: float, 
        dropout: Optional[float] = None, 
        class_indice: Optional[np.ndarray] = None,
        class2count: Optional[np.ndarray] = None, 
        head_class_indice: List = [],
        head_class2count: List = [], 
        head_weight: Optional[List] = None, 
        cmax: Optional[int] = None, 
        fc_lnum: int = 1, 
        class_weight: Optional[torch.Tensor] = None,
        head_detach: bool = True, 
        s2fc_lnum: int = 2) -> None:
        """Initialize ResNet with multi-head hierarchical classification.
        
        Args:
            block: ResNet block type (BasicBlock or Bottleneck)
            layers: List of number of blocks in each ResNet stage [4 elements]
            fds: Whether to enable Feature Distribution Smoothing
            bucket_num: Number of age buckets for FDS (default: 100)
            bucket_start: Starting bucket index for FDS (default: 0)
            start_update: Epoch to start FDS statistics update (default: 0)
            start_smooth: Epoch to start FDS feature smoothing (default: 1)
            kernel: FDS kernel type ('gaussian', 'triang', 'laplace')
            ks: FDS kernel size (should be odd)
            sigma: FDS kernel parameter
            momentum: FDS momentum for running statistics
            dropout: Dropout probability (None to disable)
            class_indice: Class boundary indices for main classifier
            class2count: Representative values for main classifier classes
            head_class_indice: List of class indices for each hierarchical head
            head_class2count: List of class representatives for each hierarchical head  
            head_weight: Class weights for hierarchical heads
            cmax: Maximum age value
            fc_lnum: Number of FC layers in hierarchical heads (1 or 2)
            class_weight: Class weights for main classifier
            head_detach: Whether to detach gradients for hierarchical heads
            s2fc_lnum: Number of FC layers in adjustment head (1, 2, or 3)
        """
        self.inplanes = 64
        super(ResNet_cls_mh, self).__init__()
        
        # ResNet backbone layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Main classification head setup
        self.class_indice = torch.Tensor(class_indice).float()
        self.class2count = torch.Tensor(class2count).float()
        assert len(self.class_indice) + 1 == len(self.class2count), \
            "Mismatch between class indices and class counts"
        self.cnum = len(class2count)
        self.linear = nn.Linear(512 * block.expansion, self.cnum)
        self.class_weight = class_weight

        # Feature Distribution Smoothing setup
        if fds:
            self.FDS = FDS(
                feature_dim=512 * block.expansion, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, 
                sigma=sigma, momentum=momentum
            )
        self.fds = fds
        self.start_smooth = start_smooth

        # Dropout setup
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        # Initialize network weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Multi-head hierarchical classification setup
        self.cmax = cmax
        self.head_num = len(head_class2count)
        self.head_class_indice = head_class_indice
        self.head_class2count = head_class2count
        self.head_weight = head_weight
        self.fc_lnum = fc_lnum
        self.head_detach = head_detach

        # Create hierarchical classification heads
        if self.head_num > 0:
            for hi in range(self.head_num):
                tmp_cnum = len(self.head_class2count[hi])
                head_name = f'cls_head_{hi}'
                
                if fc_lnum == 1:
                    # Single layer head
                    setattr(self, head_name, nn.Sequential(
                        nn.Linear(512 * block.expansion, tmp_cnum)
                    ))
                elif fc_lnum == 2:
                    # Two layer head with ReLU
                    setattr(self, head_name, nn.Sequential(
                        nn.Linear(512 * block.expansion, 512),
                        nn.ReLU(),  
                        nn.Linear(512, tmp_cnum)
                    ))

        # Adjustment head for feature calibration
        self.s2fc_lnum = s2fc_lnum
        if self.s2fc_lnum == 1:
            self.adjust_head = nn.Sequential(
                nn.Linear(512 * block.expansion, self.cnum)
            )
        elif self.s2fc_lnum == 2:
            self.adjust_head = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Softplus(),
                nn.Linear(512, self.cnum)
            )
        elif self.s2fc_lnum == 3:
            self.adjust_head = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Softplus(),  
                nn.Linear(512, 256),
                nn.Softplus(),
                nn.Linear(256, self.cnum)
            )           

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks.
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            planes: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for the first block (default: 1)
            
        Returns:
            Sequential module containing all blocks in the layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Create downsampling layer for skip connection
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block (may have stride > 1 and downsampling)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks (stride = 1, no downsampling)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using ResNet backbone.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Feature vectors [batch_size, feature_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)  # Flatten

        return encoding

    def forward_feat(self, encoding: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                    epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass from features to all predictions.
        
        Args:
            encoding: Feature vectors [batch_size, feature_dim]
            targets: Target age values for FDS (optional)
            epoch: Current epoch for FDS (optional)
            
        Returns:
            Dictionary containing:
            - 'x': Main classifier predictions
            - 'adjust_x': Adjustment head predictions  
            - 'encoding': Input feature vectors
            - 'mhpre_dict': Hierarchical head predictions
        """
        encoding_s = encoding

        # Apply Feature Distribution Smoothing if enabled
        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        # Apply dropout if enabled
        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        
        # Main classification head
        x = self.linear(encoding_s) 

        # Multi-head hierarchical predictions
        mhpre_dict = {}
        if self.head_num > 0:
            for hi in range(self.head_num):
                head_name = f'cls_{hi}'
                head_module = getattr(self, f'cls_head_{hi}')
                
                if self.head_detach:
                    # Detach gradients for hierarchical heads
                    mhpre_dict[head_name] = head_module(encoding_s.detach())
                else:
                    mhpre_dict[head_name] = head_module(encoding_s)

        # Adjustment head for feature calibration
        adjust_x = self.adjust_head(encoding_s.detach())
        
        return {
            'x': x,
            'adjust_x': adjust_x,
            'encoding': encoding,
            'mhpre_dict': mhpre_dict
        }

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, 
               epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Complete forward pass from images to predictions.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            targets: Target age values for FDS (optional)
            epoch: Current epoch for FDS (optional)
            
        Returns:
            Dictionary containing all predictions and features
        """
        # Extract features using ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)

        # Forward through classification heads
        return self.forward_feat(encoding, targets, epoch)

    def Count2Class(self, gtcounts: torch.Tensor) -> torch.Tensor:
        """Convert continuous age values to discrete class labels using main classifier boundaries.
        
        Args:
            gtcounts: Continuous age values [batch_size, ...]
            
        Returns:
            Discrete class labels [batch_size, ...]
        """
        gtclass = Count2Class(gtcounts, self.class_indice)
        return gtclass

    def Class2Count(self, preclass: torch.Tensor) -> torch.Tensor:
        """Convert discrete class predictions to continuous age values using main classifier.
        
        Args:
            preclass: Class predictions [batch_size, num_classes] or [batch_size, 1]
            
        Returns:
            Continuous age predictions [batch_size, 1]
        """
        precounts = Class2Count(preclass, self.class2count)
        return precounts

    def Count2Class_mh(self, gtcounts: torch.Tensor, class_indice: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to class labels using custom boundaries.
        
        Args:
            gtcounts: Continuous values to classify
            class_indice: Custom class boundary indices
            
        Returns:
            Discrete class labels
        """
        gtclass = Count2Class(gtcounts, class_indice)
        return gtclass
    
    def Class2Count_mh(self, preclass: torch.Tensor, class2count: torch.Tensor) -> torch.Tensor:
        """Convert class predictions to continuous values using custom representatives.
        
        Args:
            preclass: Class predictions
            class2count: Custom class representative values
            
        Returns:
            Continuous value predictions
        """
        precounts = Class2Count(preclass, class2count)
        return precounts

    def Count2Class_idx(self, gtcounts: torch.Tensor, level: int) -> torch.Tensor:
        """Convert continuous values to class labels for a specific hierarchical level.
        
        Args:
            gtcounts: Continuous values to classify
            level: Hierarchical level (0 to head_num for hierarchical heads, head_num for main)
            
        Returns:
            Class labels for the specified level
        """
        if level < self.head_num:
            class_indice = self.head_class_indice[level]
            class_indice = torch.Tensor(class_indice).float().cuda()
        else:
            class_indice = self.class_indice    
        gtclass = Count2Class(gtcounts, class_indice)
        return gtclass
    
    def Class2Count_idx(self, preclass: torch.Tensor, level: int) -> torch.Tensor:
        """Convert class predictions to continuous values for a specific hierarchical level.
        
        Args:
            preclass: Class predictions  
            level: Hierarchical level (0 to head_num for hierarchical heads, head_num for main)
            
        Returns:
            Continuous value predictions for the specified level
        """
        if level < self.head_num:
            class2count = self.head_class2count[level]
            class2count = torch.Tensor(class2count).float().cuda()
        else:
            class2count = self.class2count          
        precounts = Class2Count(preclass, class2count)
        return precounts


    def hier_combine_test(self,allpre,if_entropy=False):  
        tmp_combine_pre = F.softmax(allpre['x'],dim=1)
        tmp_combine_premul = F.softmax(allpre['x'],dim=1)
        if if_entropy:
            entropy_w =  (torch.special.entr(tmp_combine_pre)).sum(dim=1,keepdim=True)
            tmpc = tmp_combine_pre.size()[1]
            maxe = tmpc*torch.special.entr(torch.Tensor([1.0/tmpc]).float().cuda())
            entropy_w = 1- entropy_w/maxe 
        else:
            entropy_w = 1                  
        tmp_combine_pre = tmp_combine_pre* entropy_w         
        tmp_combine_premul = torch.pow(tmp_combine_premul,entropy_w)  
        
        head_num = self.head_num
        for hi in range(head_num):
            tmp_key = 'cls_'+str(hi)
            tmp_clspre = allpre['mhpre_dict'][tmp_key]
            tmp_clspre = F.softmax(tmp_clspre,dim=1)
            if if_entropy:
                entropy_w =  (torch.special.entr(tmp_clspre)).sum(dim=1,keepdim=True)
                tmpc = tmp_clspre.size()[1]
                maxe = tmpc*torch.special.entr(torch.Tensor([1.0/tmpc]).float().cuda())
                entropy_w = 1- entropy_w/maxe
            else:
                entropy_w = 1

            tmp_clspre_mul = torch.pow(tmp_clspre,entropy_w)
            tmp_clspre = tmp_clspre*entropy_w

            tmp_cindice = self.head_class_indice[hi]
            tmp_cindice =  torch.Tensor(tmp_cindice).float().cuda()
            
            last_class2count = self.class2count.cuda()
            tmp_cnum = len(tmp_cindice)+1
            tmp_mask = self.Count2Class_mh(last_class2count,tmp_cindice)                
            _med = torch.arange(tmp_cnum).long().view(1,-1).cuda()
            tmp_mask = (tmp_mask.view(-1,1)==_med).float()
            tmp_combine_pre = tmp_combine_pre.unsqueeze(2)+tmp_clspre.unsqueeze(1)
            tmp_combine_pre = (tmp_combine_pre*tmp_mask).sum(dim=2)
            tmp_combine_premul = tmp_combine_premul.unsqueeze(2)*tmp_clspre_mul.unsqueeze(1)
            tmp_combine_premul = (tmp_combine_premul*tmp_mask).sum(dim=2)
        combine_dict = dict()
        combine_dict['add'] = tmp_combine_pre
        combine_dict['mul'] = tmp_combine_premul
        combine_dict['addv'] = self.Class2Count_idx(combine_dict['add'],self.head_num)
        combine_dict['mulv'] = self.Class2Count_idx(combine_dict['mul'],self.head_num)

        return combine_dict

    def hier_pool1(self,tmp_logit,pooltype='max',coarse_idx=-1,tmp_idx=-1,coarse_cindice=None,tmp_class2count=None):
        if coarse_idx<0:
            coarse_cindice =  torch.Tensor(coarse_cindice).float().cuda()
        else:
            if coarse_idx<self.head_num:
                coarse_cindice = self.head_class_indice[coarse_idx]
                coarse_cindice =  torch.Tensor(coarse_cindice).float().cuda()
            else:
                coarse_cindice = self.class_indice
            
        if tmp_idx<0:
            tmp_class2count = torch.Tensor(tmp_class2count).float().cuda()
        else:
            if tmp_idx<self.head_num:
                tmp_class2count = self.head_class2count[tmp_idx]
                tmp_class2count = torch.Tensor(tmp_class2count).float().cuda()
            else:
                tmp_class2count = self.class2count    
        
        coarse_cnum = len(coarse_cindice)+1
        tmp_mask = self.Count2Class_mh(tmp_class2count,coarse_cindice)                
        tmp_mask = F.one_hot(tmp_mask.view(-1),num_classes=coarse_cnum).cuda().detach()
        if len(tmp_logit.size())==2:
            tmp_mask = tmp_mask.unsqueeze(0)
        elif len(tmp_logit.size())==4:
            tmp_mask = tmp_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        tmp_maxpool = tmp_logit.unsqueeze(2)
        
        if pooltype == 'max':
            maxabs = tmp_maxpool.abs().max().item()
            tmp_maxpool = (tmp_maxpool*tmp_mask - 2*maxabs*(1-tmp_mask)).max(dim=1)[0] 
        elif pooltype == 'sum':
            tmp_maxpool = (tmp_maxpool*tmp_mask).sum(dim=1)
        elif pooltype == 'mean':
            tmp_maxpool = (tmp_maxpool*tmp_mask).mean(dim=1)            

        return tmp_maxpool

    def get_mask_gt(self,allpre,gt):
        if len(allpre.size())>len(gt.size()):
            gt = gt.unsqueeze(1)
        
        
        head_num_adj = self.head_num+1
        for hi in range(head_num_adj):
            if hi<self.head_num:
                tmp_cls = 'cls_'+str(hi)
                tmp_cls = allpre['mhpre_dict'][tmp_cls]
            else:
                tmp_cls = allpre['x']
            
                
            tmp_value = self.Class2Count_idx(tmp_cls.detach(),hi)
            tmp_str = 'v_'+str(hi)
            allpre[tmp_str]=tmp_value

        mask_gt = []
        for hi1 in range(head_num_adj-1):
            tmp_str = 'v_'+str(hi1)      
            tmp_class = self.Count2Class_idx(allpre[tmp_str],hi1)
            gt_class = self.Count2Class_idx(gt,hi1)
            tmp_class_abs = (tmp_class-gt_class).float().abs()
            other_cls_abs = []
            for hi2 in range(hi1,head_num_adj):
                tmp_str2 = 'v_'+str(hi2)
                other_cls = self.Count2Class_idx(allpre[tmp_str2],hi1)
                other_cls_abs.append((other_cls-gt_class).float().abs())

            other_cls_abs = torch.cat(other_cls_abs,dim=1)
            tmp_mask = (tmp_class_abs<=other_cls_abs.min(dim=1,keepdim=True)[0]).float()
            mask_gt.append(tmp_mask)

            if hi1==(head_num_adj-2):
                tmp_mask = (other_cls_abs[:,-1]<=other_cls_abs.min(dim=1)[0]).float()
                mask_gt.append(tmp_mask.view(-1,1))
        mask_gt = torch.cat(mask_gt,dim=1)
        return mask_gt

    def get_mask_gt2(self,allpre,gt):
        if len(allpre['x'].size())>len(gt.size()):
            gt = gt.unsqueeze(1)
        
        
        head_num_adj = self.head_num+1
        for hi in range(head_num_adj):
            if hi<self.head_num:
                tmp_cls = 'cls_'+str(hi)
                tmp_cls = allpre['mhpre_dict'][tmp_cls]
            else:
                tmp_cls = allpre['x']

            tmp_value = self.Class2Count_idx(tmp_cls.detach(),hi)
            tmp_str = 'v_'+str(hi)
            allpre[tmp_str]=tmp_value

        mask_gt = []
        for hi1 in range(head_num_adj-1):
            gt_class = self.Count2Class_idx(gt,hi1)
            other_cls_abs = []
            for hi2 in range(hi1,head_num_adj):
                tmp_str2 = 'v_'+str(hi2)
                other_cls = self.Count2Class_idx(allpre[tmp_str2],hi1)
                other_cls_abs.append((other_cls-gt_class).float().abs())

            other_cls_abs = torch.cat(other_cls_abs,dim=1)
            tmp_mask = (other_cls_abs<=other_cls_abs.min(dim=1,keepdim=True)[0]).float()
            if hi1==0:
                mask_gt=tmp_mask
            else:
                mask_gt[:,hi1:] = mask_gt[:,hi1:]*tmp_mask
        return mask_gt

    def get_correctmask(self,allpre,gt):
        if len(allpre['x'].size())>len(gt.size()):
            gt = gt.unsqueeze(1) 
        
        mask_dict = dict()
        head_num_adj = self.head_num+1
        for hi in range(head_num_adj):
            if hi<self.head_num:
                tmp_cls = 'cls_'+str(hi)
                tmp_cls = allpre['mhpre_dict'][tmp_cls].detach()
            else:
                tmp_cls = allpre['x'].detach()

            tmp_cnum = tmp_cls.size()[1]
            gt_class = self.Count2Class_idx(gt,hi)
            if len(gt_class.size())==4:
                gt_class = F.one_hot(gt_class,tmp_cnum).squeeze(1).permute(0,3,1,2)
            else:
                gt_class = F.one_hot(gt_class,tmp_cnum).squeeze(1)
            thre = (tmp_cls*gt_class).sum(dim=1,keepdim=True)
            tmp_mask = (tmp_cls>thre).float()
            tmp_mask = 1-tmp_mask
            tmp_str = 'mask_'+str(hi)
            mask_dict[tmp_str] = tmp_mask
        return mask_dict

    def hier_bestKL(self,allpre,gt,ptype='max',mask_type='all',addgt=False,maskerr=False,losstype='KL'):
        if maskerr:
            mask_dict = self.get_correctmask(allpre,gt)
        head_num = self.head_num
        adjust_logit = allpre['adjust_x'] 

        ret_KL = []
        for cidx in range(0,head_num+1): 
            if cidx == head_num:
                tmp_fine_logits = allpre['x'].detach()
            else:
                tmp_str = 'cls_'+str(cidx)
                tmp_fine_logits = allpre['mhpre_dict'][tmp_str].detach()
            
            tmp_fine_logits_pool = tmp_fine_logits
            if ptype=='sum':
                adjust_logit = F.softmax(adjust_logit,dim=1)
                tmp_fine_logits_pool = F.softmax(tmp_fine_logits_pool,dim=1)
            
            if cidx<head_num:
                adjust_logit_pool = self.hier_pool1(adjust_logit,pooltype=ptype,coarse_idx=cidx,tmp_idx=self.head_num)
            else:
                adjust_logit_pool = adjust_logit 

            if not maskerr:
                if ptype=='max':
                    if losstype=='KL':
                        tmp_KL = myKL_div(adjust_logit_pool,tmp_fine_logits_pool.detach())
                    elif losstype=='JS':
                        tmp_KL = myJS_div(adjust_logit_pool,tmp_fine_logits_pool.detach())                
                elif ptype=='sum':
                    if losstype=='KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())
                    elif losstype=='JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())                
            else:
                tmp_maskstr = 'mask_'+str(cidx)
                tmp_mask_right = mask_dict[tmp_maskstr]

                def mask_softmax(pre,mask,dim=1,ifexp=True):
                    if ifexp:
                        prenorm = torch.exp(pre)
                    prenorm = prenorm*mask
                    pre_sum = prenorm.sum(dim=dim,keepdim=True)
                    EPS = 1e-6
                    prenorm = prenorm/torch.where(pre_sum<EPS,pre_sum+EPS,pre_sum)
                    return prenorm

                if ptype=='max':
                    adjust_logit_pool = mask_softmax(adjust_logit_pool,tmp_mask_right,dim=1,ifexp=True)
                    tmp_fine_logits_pool  = mask_softmax(tmp_fine_logits_pool,tmp_mask_right,dim=1,ifexp=True)
                    if losstype=='KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())
                    elif losstype=='JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())                
                elif ptype=='sum':
                    adjust_logit_pool = mask_softmax(adjust_logit_pool,tmp_mask_right,dim=1,ifexp=False)
                    tmp_fine_logits_pool  = mask_softmax(tmp_fine_logits_pool,tmp_mask_right,dim=1,ifexp=False)
                    if losstype=='KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())
                    elif losstype=='JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool,tmp_fine_logits_pool.detach())  


            ret_KL.append(tmp_KL)

        ret_KL = torch.cat(ret_KL,dim=1)

        if mask_type == 'all':
            if len(ret_KL.size())==2:
                ret_KL = (ret_KL).mean(dim=0)
            else:
                ret_KL = (ret_KL).mean(dim=3).mean(dim=2).mean(dim=0)
        elif mask_type == 'best':
            best_mask = self.get_mask_gt2(allpre,gt)
            if len(ret_KL.size())==2:
                ret_KL = (ret_KL*best_mask).mean(dim=0)
            else:
                ret_KL = (ret_KL*best_mask).mean(dim=3).mean(dim=2).mean(dim=0)


        return ret_KL



    def get_range_mask(self,allpre,gt):
        head_num = self.head_num
        range_mask = 0
        for hi in range(head_num+1):
            tmp_gt_class = self.Count2Class_idx(gt,hi).float()
            tmp_cls = 'cls_'+str(hi)
            if hi<head_num:
                tmp_cls_class = allpre['mhpre_dict'][tmp_cls].detach().max(dim=1,keepdim=True)[1].float()
            else:
                tmp_cls_class = allpre['x'].detach().max(dim=1,keepdim=True)[1].float()
            tmp_reg_class = self.Count2Class_idx(allpre['reg'].detach(),hi).float()

            cls_mae = (tmp_cls_class-tmp_gt_class).abs()
            reg_mae = (tmp_reg_class-tmp_gt_class).abs()

            tmp_mask = (reg_mae>cls_mae).float()
            range_mask = range_mask+tmp_mask
        range_mask = (range_mask>1e-5).float()
        return range_mask

    def get_range_mask2(self,allpre,gt):
        head_num = self.head_num
        range_mask = 0
        for hi in range(head_num+1):
            tmp_gt_class = self.Count2Class_idx(gt,hi).float()
            tmp_cls = 'cls_'+str(hi)
            if hi<head_num:
                tmp_cls_class = allpre['mhpre_dict'][tmp_cls].detach().max(dim=1,keepdim=True)[1].float()
            else:
                tmp_cls_class = allpre['x'].detach().max(dim=1,keepdim=True)[1].float()
            tmp_reg_class = self.Count2Class_idx(allpre['reg'].detach(),hi).float()
            bound_cls = torch.cat([tmp_gt_class,tmp_cls_class],dim=1)
            tmp_range_mask = (tmp_reg_class>=bound_cls.min(dim=1,keepdim=True)[0]).float()
            tmp_range_mask = tmp_range_mask* (tmp_reg_class<=bound_cls.max(dim=1,keepdim=True)[0]).float()
            tmp_range_mask = 1-tmp_range_mask
            range_mask = range_mask+tmp_range_mask
        range_mask = (range_mask>1e-5).float()
        return range_mask

# ===================================================================
def resnet50_cls_mh(**kwargs) -> 'ResNet_cls_mh':
    """Create ResNet-50 with multi-head hierarchical classification.
    
    Factory function to create a ResNet-50 model with hierarchical classification heads
    and optional Feature Distribution Smoothing.
    
    Args:
        **kwargs: Keyword arguments passed to ResNet_cls_mh constructor
        
    Returns:
        ResNet-50 model with multi-head classification
    """
    return ResNet_cls_mh(Bottleneck, [3, 4, 6, 3], **kwargs)