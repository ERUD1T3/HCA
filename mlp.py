from fds import FDS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, List, Optional, Tuple, Any


# Classification related functions
from cls_funcs import Count2Class, Class2Count


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


class MLP(nn.Module):
    """
    A PyTorch MLP that:
      - Supports skip connections every 'skipped_layers' blocks
      - Applies batch normalization
      - Optionally applies dropout
      - Always produces a linear output for regression
      - Optionally applies FDS smoothing to the final representation
      - Supports multi-head hierarchical classification for age estimation
    """

    def __init__(
        self,
        input_dim: int = 100,
        output_dim: int = 1,
        hiddens: Union[list[int], None] = None,
        skipped_layers: int = 1,
        embed_dim: int = 128,
        skip_repr: bool = True,
        activation: Union[nn.Module, None] = None,
        dropout: float = 0.2,
        name: str = 'mlp',
        fds: bool = False,
        bucket_num: int = 50,
        bucket_start: int = 0,
        start_update: int = 0,
        start_smooth: int = 1,
        kernel: str = 'gaussian',
        ks: int = 5,
        sigma: float = 2.0,
        momentum: float = 0.9,
        # Multi-head hierarchical classification parameters
        class_indice: Optional[np.ndarray] = None,
        class2count: Optional[np.ndarray] = None,
        head_class_indice: List = [],
        head_class2count: List = [],
        head_weight: Optional[List] = None,
        cmax: Optional[int] = None,
        fc_lnum: int = 1,
        class_weight: Optional[torch.Tensor] = None,
        head_detach: bool = True,
        s2fc_lnum: int = 2
    ) -> None:
        """
        Creates an MLP with optional skip connections, batch normalization, dropout, FDS smoothing,
        and multi-head hierarchical classification.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features (always linear regression).
            hiddens (list[int]): Sizes of hidden layers. Defaults to [50, 50] if None.
            skipped_layers (int): Frequency of residual skip connections.
            embed_dim (int): Size of the final embedding layer.
            skip_repr (bool): If True, merges a skip into the final embedding block.
            activation (nn.Module): Activation function to use, defaults to LeakyReLU if None.
            dropout (float): Dropout probability. No dropout if 0.
            name (str): Name of the model (not used internally, just for reference).
            fds (bool): If True, enables FDS smoothing on the final representation.
            bucket_num, bucket_start, start_update, start_smooth, kernel, ks, sigma, momentum:
                Parameters for FDS if enabled.
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
        super().__init__()

        if hiddens is None:
            hiddens = [50, 50]
        if skipped_layers >= len(hiddens):
            raise ValueError(
                f"skipped_layers ({skipped_layers}) must be < number of hidden layers ({len(hiddens)})"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens
        self.skipped_layers = skipped_layers
        self.embed_dim = embed_dim
        self.skip_repr = skip_repr
        self.dropout_rate = dropout
        self.name = name
        self.activation_fn = activation if (activation is not None) else nn.LeakyReLU()

        # Set up FDS if requested
        self.fds_enabled = fds
        self.start_smooth = start_smooth
        if self.fds_enabled:
            self.fds_module = FDS(
                feature_dim=embed_dim,
                bucket_num=bucket_num,
                bucket_start=bucket_start,
                start_update=start_update,
                start_smooth=start_smooth,
                kernel=kernel,
                ks=ks,
                sigma=sigma,
                momentum=momentum
            )
        else:
            self.fds_module = None

        # Multi-head hierarchical classification setup
        if class_indice is not None and class2count is not None:
            self.class_indice = torch.Tensor(class_indice).float()
            self.class2count = torch.Tensor(class2count).float()
            assert len(self.class_indice) + 1 == len(self.class2count), \
                "Mismatch between class indices and class counts"
            self.cnum = len(class2count)
            self.class_weight = class_weight
        else:
            self.class_indice = None
            self.class2count = None
            self.cnum = output_dim
            self.class_weight = None

        self.cmax = cmax
        self.head_num = len(head_class2count)
        self.head_class_indice = head_class_indice
        self.head_class2count = head_class2count
        self.head_weight = head_weight
        self.fc_lnum = fc_lnum
        self.head_detach = head_detach
        self.s2fc_lnum = s2fc_lnum

        # Define hidden blocks as nn.Sequential modules stored in a ModuleList
        self.layers = nn.ModuleList()

        # 1) First block
        block0 = []
        block0.append(nn.Linear(input_dim, hiddens[0], bias=True))
        block0.append(nn.BatchNorm1d(hiddens[0]))
        self.layers.append(nn.Sequential(*block0))

        # 2) Additional hidden blocks
        for idx, units in enumerate(hiddens[1:], start=1):
            block = []
            block.append(nn.Linear(hiddens[idx - 1], units, bias=True))
            block.append(nn.BatchNorm1d(units))
            self.layers.append(nn.Sequential(*block))

        # 3) Final embedding block
        final_block = []
        final_block.append(nn.Linear(hiddens[-1], embed_dim, bias=True))
        final_block.append(nn.BatchNorm1d(embed_dim))
        self.final_embed = nn.Sequential(*final_block)

        # 4) Main classification head (if hierarchical classification is enabled)
        if self.class_indice is not None:
            self.linear = nn.Linear(embed_dim, self.cnum)
        else:
            # 4) Output layer (always linear for regression)
            self.output_layer = nn.Linear(embed_dim, output_dim, bias=True)

        # Create hierarchical classification heads
        if self.head_num > 0:
            for hi in range(self.head_num):
                tmp_cnum = len(self.head_class2count[hi])
                head_name = f'cls_head_{hi}'
                
                if fc_lnum == 1:
                    # Single layer head
                    setattr(self, head_name, nn.Sequential(
                        nn.Linear(embed_dim, tmp_cnum)
                    ))
                elif fc_lnum == 2:
                    # Two layer head with ReLU
                    setattr(self, head_name, nn.Sequential(
                        nn.Linear(embed_dim, 512),
                        nn.ReLU(),  
                        nn.Linear(512, tmp_cnum)
                    ))

        # Adjustment head for feature calibration (if hierarchical classification is enabled)
        if self.class_indice is not None:
            if self.s2fc_lnum == 1:
                self.adjust_head = nn.Sequential(
                    nn.Linear(embed_dim, self.cnum)
                )
            elif self.s2fc_lnum == 2:
                self.adjust_head = nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.Softplus(),
                    nn.Linear(512, self.cnum)
                )
            elif self.s2fc_lnum == 3:
                self.adjust_head = nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.Softplus(),  
                    nn.Linear(512, 256),
                    nn.Softplus(),
                    nn.Linear(256, self.cnum)
                )
        else:
            self.adjust_head = None

        # Single dropout module (applied manually in forward)
        self.dropout_module = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None

    def Count2Class(self, gtcounts: torch.Tensor) -> torch.Tensor:
        """Convert continuous age values to discrete class labels using main classifier boundaries.
        
        Args:
            gtcounts: Continuous age values [batch_size, ...]
            
        Returns:
            Discrete class labels [batch_size, ...]
        """
        if self.class_indice is None:
            raise ValueError("class_indice not set - hierarchical classification not enabled")
        gtclass = Count2Class(gtcounts, self.class_indice)
        return gtclass

    def Class2Count(self, preclass: torch.Tensor) -> torch.Tensor:
        """Convert discrete class predictions to continuous age values using main classifier.
        
        Args:
            preclass: Class predictions [batch_size, num_classes] or [batch_size, 1]
            
        Returns:
            Continuous age predictions [batch_size, 1]
        """
        if self.class2count is None:
            raise ValueError("class2count not set - hierarchical classification not enabled")
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
            if self.class_indice is None:
                raise ValueError("class_indice not set - hierarchical classification not enabled")
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
            if self.class2count is None:
                raise ValueError("class2count not set - hierarchical classification not enabled")
            class2count = self.class2count          
        precounts = Class2Count(preclass, class2count)
        return precounts

    def forward_feat(self, encoding: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                    epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass from features to all predictions.
        
        Args:
            encoding: Feature vectors [batch_size, feature_dim]
            targets: Target age values for FDS (optional)
            epoch: Current epoch for FDS (optional)
            
        Returns:
            Dictionary containing:
            - 'x': Main classifier predictions (if hierarchical classification enabled)
            - 'adjust_x': Adjustment head predictions (if hierarchical classification enabled)
            - 'encoding': Input feature vectors
            - 'mhpre_dict': Hierarchical head predictions (if hierarchical classification enabled)
            - 'preds': Regression predictions (if hierarchical classification disabled)
        """
        encoding_s = encoding

        # Apply Feature Distribution Smoothing if enabled
        if self.training and self.fds_enabled:
            if epoch is not None and epoch >= self.start_smooth:
                if targets is None:
                    raise ValueError("Labels must be provided for FDS smoothing.")
                encoding_s = self.fds_module.smooth(encoding_s, targets, epoch)

        # Apply dropout if enabled
        if self.dropout_module is not None:
            encoding_s = self.dropout_module(encoding_s)

        result = {'encoding': encoding}

        if self.class_indice is not None:
            # Hierarchical classification mode
            # Main classification head
            x = self.linear(encoding_s) 
            result['x'] = x

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

            result['mhpre_dict'] = mhpre_dict

            # Adjustment head for feature calibration
            if self.adjust_head is not None:
                adjust_x = self.adjust_head(encoding_s.detach())
                result['adjust_x'] = adjust_x
        else:
            # Original regression mode
            preds = self.output_layer(encoding_s)
            result['preds'] = preds

        return result

    def forward(
        self,
        x: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
        epoch: Union[int, None] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            labels (torch.Tensor | None): Target labels for FDS. May be None if FDS is off.
            epoch (int | None): Current epoch for enabling FDS smoothing. May be None if FDS is off.

        Returns:
            If hierarchical classification is enabled:
                Dictionary containing all predictions and features
            If hierarchical classification is disabled:
                If output_layer is defined:
                    (prediction, final_repr)
                Otherwise:
                    final_repr
        """
        # First hidden block
        out = self.layers[0](x)       # linear + BN
        out = self.activation_fn(out) # activation

        # Possibly skip from input
        if self.skipped_layers > 0:
            if out.shape[1] != x.shape[1]:
                # For dimension mismatch, define a small projection or do partial slice
                # Here, we do a direct linear for clarity:
                projection = nn.Linear(x.shape[1], out.shape[1], bias=False).to(x.device)
                skip_out = projection(x)
            else:
                skip_out = x

            out = out + skip_out
            # Apply dropout *after* first skip connection + activation
            if self.dropout_module is not None:
                out = self.dropout_module(out)
        else:
            # If no skip, just dropout after activation
            if self.dropout_module is not None:
                out = self.dropout_module(out)

        residual = out

        # Middle hidden blocks
        for idx in range(1, len(self.layers)):
            out = self.layers[idx](out)
            out = self.activation_fn(out)

            if self.skipped_layers > 0 and idx % self.skipped_layers == 0:
                if out.shape[1] != residual.shape[1]:
                    # Keep the original dynamic projection method
                    projection = nn.Linear(residual.shape[1], out.shape[1], bias=False).to(x.device)
                    skip_out = projection(residual)
                else:
                    skip_out = residual

                out = out + skip_out
                # Apply dropout *after* skip connection + activation
                if self.dropout_module is not None:
                    out = self.dropout_module(out)
                residual = out # Update residual state *after* dropout
            else:
                # If no skip, just dropout after activation
                if self.dropout_module is not None:
                    out = self.dropout_module(out)
                # Note: residual does not update here if no skip

        # Final embedding block
        out = self.final_embed(out) # linear + BN

        # --- Activation and Dropout Placement Logic Changed ---
        if self.skip_repr:
            # Apply activation *before* potential skip connection addition
            activated_out = self.activation_fn(out)

            if self.skipped_layers > 0:
                # Prepare skip connection from the last residual state
                if out.shape[1] != residual.shape[1]: # Compare shape *before* activation
                    projection = nn.Linear(residual.shape[1], out.shape[1], bias=False).to(x.device)
                    skip_out = projection(residual)
                else:
                    skip_out = residual

                # Apply dropout to the activated output *before* adding the skip
                if self.dropout_module is not None:
                    activated_out = self.dropout_module(activated_out)

                # Add skip connection to the *activated* output
                final_repr = activated_out + skip_out
                # NO activation after the add
            else:
                # Case: skip_repr is True, but no skip connection (skipped_layers=0)
                # Apply dropout if needed to the activated output
                if self.dropout_module is not None:
                    activated_out = self.dropout_module(activated_out)
                # Final repr is just the activated output (potentially dropout-applied)
                final_repr = activated_out
        else:
            # Case: skip_repr is False. Representation is pre-activation.
            # Apply dropout if needed to the output of final_embed
            if self.dropout_module is not None:
                out = self.dropout_module(out)
            final_repr = out # No activation applied here when skip_repr is False

        # Forward through classification/regression heads
        return self.forward_feat(final_repr, labels, epoch)

    def hier_combine_test(self, allpre, if_entropy=False):
        """Hierarchical prediction combination for test time.
        
        Args:
            allpre: Dictionary containing all predictions
            if_entropy: Whether to use entropy weighting
            
        Returns:
            Dictionary with combined predictions
        """
        tmp_combine_pre = F.softmax(allpre['x'], dim=1)
        tmp_combine_premul = F.softmax(allpre['x'], dim=1)
        
        if if_entropy:
            entropy_w = (torch.special.entr(tmp_combine_pre)).sum(dim=1, keepdim=True)
            tmpc = tmp_combine_pre.size()[1]
            maxe = tmpc * torch.special.entr(torch.Tensor([1.0/tmpc]).float().cuda())
            entropy_w = 1 - entropy_w/maxe 
        else:
            entropy_w = 1                  
        
        tmp_combine_pre = tmp_combine_pre * entropy_w         
        tmp_combine_premul = torch.pow(tmp_combine_premul, entropy_w)  
        
        head_num = self.head_num
        for hi in range(head_num):
            tmp_key = 'cls_' + str(hi)
            tmp_clspre = allpre['mhpre_dict'][tmp_key]
            tmp_clspre = F.softmax(tmp_clspre, dim=1)
            
            if if_entropy:
                entropy_w = (torch.special.entr(tmp_clspre)).sum(dim=1, keepdim=True)
                tmpc = tmp_clspre.size()[1]
                maxe = tmpc * torch.special.entr(torch.Tensor([1.0/tmpc]).float().cuda())
                entropy_w = 1 - entropy_w/maxe
            else:
                entropy_w = 1

            tmp_clspre_mul = torch.pow(tmp_clspre, entropy_w)
            tmp_clspre = tmp_clspre * entropy_w

            tmp_cindice = self.head_class_indice[hi]
            tmp_cindice = torch.Tensor(tmp_cindice).float().cuda()
            
            if self.class2count is None:
                raise ValueError("class2count not set - hierarchical classification not enabled")
            last_class2count = self.class2count.cuda()
            tmp_cnum = len(tmp_cindice) + 1
            tmp_mask = self.Count2Class_mh(last_class2count, tmp_cindice)                
            _med = torch.arange(tmp_cnum).long().view(1, -1).cuda()
            tmp_mask = (tmp_mask.view(-1, 1) == _med).float()
            
            tmp_combine_pre = tmp_combine_pre.unsqueeze(2) + tmp_clspre.unsqueeze(1)
            tmp_combine_pre = (tmp_combine_pre * tmp_mask).sum(dim=2)
            tmp_combine_premul = tmp_combine_premul.unsqueeze(2) * tmp_clspre_mul.unsqueeze(1)
            tmp_combine_premul = (tmp_combine_premul * tmp_mask).sum(dim=2)
            
        combine_dict = dict()
        combine_dict['add'] = tmp_combine_pre
        combine_dict['mul'] = tmp_combine_premul
        combine_dict['addv'] = self.Class2Count_idx(combine_dict['add'], self.head_num)
        combine_dict['mulv'] = self.Class2Count_idx(combine_dict['mul'], self.head_num)

        return combine_dict

    def hier_pool1(self, tmp_logit, pooltype='max', coarse_idx=-1, tmp_idx=-1, 
                   coarse_cindice=None, tmp_class2count=None):
        """Hierarchical pooling operation.
        
        Args:
            tmp_logit: Input logits to pool
            pooltype: Type of pooling ('max', 'sum', 'mean')
            coarse_idx: Index of coarse level (-1 to use custom)
            tmp_idx: Index of target level (-1 to use custom)
            coarse_cindice: Custom coarse class indices
            tmp_class2count: Custom class representatives
            
        Returns:
            Pooled logits
        """
        if coarse_idx < 0:
            coarse_cindice = torch.Tensor(coarse_cindice).float().cuda()
        else:
            if coarse_idx < self.head_num:
                coarse_cindice = self.head_class_indice[coarse_idx]
                coarse_cindice = torch.Tensor(coarse_cindice).float().cuda()
            else:
                if self.class_indice is None:
                    raise ValueError("class_indice not set - hierarchical classification not enabled")
                coarse_cindice = self.class_indice
            
        if tmp_idx < 0:
            tmp_class2count = torch.Tensor(tmp_class2count).float().cuda()
        else:
            if tmp_idx < self.head_num:
                tmp_class2count = self.head_class2count[tmp_idx]
                tmp_class2count = torch.Tensor(tmp_class2count).float().cuda()
            else:
                if self.class2count is None:
                    raise ValueError("class2count not set - hierarchical classification not enabled")
                tmp_class2count = self.class2count    
        
        coarse_cnum = len(coarse_cindice) + 1
        tmp_mask = self.Count2Class_mh(tmp_class2count, coarse_cindice)                
        tmp_mask = F.one_hot(tmp_mask.view(-1), num_classes=coarse_cnum).cuda().detach()
        
        if len(tmp_logit.size()) == 2:
            tmp_mask = tmp_mask.unsqueeze(0)
        elif len(tmp_logit.size()) == 4:
            tmp_mask = tmp_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        tmp_maxpool = tmp_logit.unsqueeze(2)
        
        if pooltype == 'max':
            maxabs = tmp_maxpool.abs().max().item()
            tmp_maxpool = (tmp_maxpool * tmp_mask - 2 * maxabs * (1 - tmp_mask)).max(dim=1)[0] 
        elif pooltype == 'sum':
            tmp_maxpool = (tmp_maxpool * tmp_mask).sum(dim=1)
        elif pooltype == 'mean':
            tmp_maxpool = (tmp_maxpool * tmp_mask).mean(dim=1)            

        return tmp_maxpool

    def get_mask_gt2(self, allpre, gt):
        """Get ground truth based mask for hierarchical predictions.
        
        Args:
            allpre: Dictionary containing all predictions
            gt: Ground truth values
            
        Returns:
            Mask tensor for best predictions
        """
        if len(allpre['x'].size()) > len(gt.size()):
            gt = gt.unsqueeze(1)
        
        head_num_adj = self.head_num + 1
        for hi in range(head_num_adj):
            if hi < self.head_num:
                tmp_cls = 'cls_' + str(hi)
                tmp_cls = allpre['mhpre_dict'][tmp_cls]
            else:
                tmp_cls = allpre['x']

            tmp_value = self.Class2Count_idx(tmp_cls.detach(), hi)
            tmp_str = 'v_' + str(hi)
            allpre[tmp_str] = tmp_value

        mask_gt = []
        for hi1 in range(head_num_adj - 1):
            gt_class = self.Count2Class_idx(gt, hi1)
            other_cls_abs = []
            for hi2 in range(hi1, head_num_adj):
                tmp_str2 = 'v_' + str(hi2)
                other_cls = self.Count2Class_idx(allpre[tmp_str2], hi1)
                other_cls_abs.append((other_cls - gt_class).float().abs())

            other_cls_abs = torch.cat(other_cls_abs, dim=1)
            tmp_mask = (other_cls_abs <= other_cls_abs.min(dim=1, keepdim=True)[0]).float()
            if hi1 == 0:
                mask_gt = tmp_mask
            else:
                mask_gt[:, hi1:] = mask_gt[:, hi1:] * tmp_mask
        return mask_gt

    def get_correctmask(self, allpre, gt):
        """Get correctness mask for predictions.
        
        Args:
            allpre: Dictionary containing all predictions
            gt: Ground truth values
            
        Returns:
            Dictionary of correctness masks
        """
        if len(allpre['x'].size()) > len(gt.size()):
            gt = gt.unsqueeze(1) 
        
        mask_dict = dict()
        head_num_adj = self.head_num + 1
        for hi in range(head_num_adj):
            if hi < self.head_num:
                tmp_cls = 'cls_' + str(hi)
                tmp_cls = allpre['mhpre_dict'][tmp_cls].detach()
            else:
                tmp_cls = allpre['x'].detach()

            tmp_cnum = tmp_cls.size()[1]
            gt_class = self.Count2Class_idx(gt, hi)
            
            if len(gt_class.size()) == 4:
                gt_class = F.one_hot(gt_class, tmp_cnum).squeeze(1).permute(0, 3, 1, 2)
            else:
                gt_class = F.one_hot(gt_class, tmp_cnum).squeeze(1)
                
            thre = (tmp_cls * gt_class).sum(dim=1, keepdim=True)
            tmp_mask = (tmp_cls > thre).float()
            tmp_mask = 1 - tmp_mask
            tmp_str = 'mask_' + str(hi)
            mask_dict[tmp_str] = tmp_mask
        return mask_dict

    def hier_bestKL(self, allpre, gt, ptype='max', mask_type='all', addgt=False, 
                    maskerr=False, losstype='KL'):
        """Compute hierarchical best KL divergence.
        
        Args:
            allpre: Dictionary containing all predictions
            gt: Ground truth values
            ptype: Pooling type ('max' or 'sum')
            mask_type: Mask type ('all' or 'best')
            addgt: Whether to add ground truth
            maskerr: Whether to use error masking
            losstype: Loss type ('KL' or 'JS')
            
        Returns:
            KL divergence tensor
        """
        if maskerr:
            mask_dict = self.get_correctmask(allpre, gt)
            
        head_num = self.head_num
        if 'adjust_x' not in allpre:
            raise ValueError("adjust_x not found in predictions - adjustment head not available")
        adjust_logit = allpre['adjust_x'] 

        ret_KL = []
        for cidx in range(0, head_num + 1): 
            if cidx == head_num:
                tmp_fine_logits = allpre['x'].detach()
            else:
                tmp_str = 'cls_' + str(cidx)
                tmp_fine_logits = allpre['mhpre_dict'][tmp_str].detach()
            
            tmp_fine_logits_pool = tmp_fine_logits
            if ptype == 'sum':
                adjust_logit = F.softmax(adjust_logit, dim=1)
                tmp_fine_logits_pool = F.softmax(tmp_fine_logits_pool, dim=1)
            
            if cidx < head_num:
                adjust_logit_pool = self.hier_pool1(adjust_logit, pooltype=ptype, 
                                                  coarse_idx=cidx, tmp_idx=self.head_num)
            else:
                adjust_logit_pool = adjust_logit 

            if not maskerr:
                if ptype == 'max':
                    if losstype == 'KL':
                        tmp_KL = myKL_div(adjust_logit_pool, tmp_fine_logits_pool.detach())
                    elif losstype == 'JS':
                        tmp_KL = myJS_div(adjust_logit_pool, tmp_fine_logits_pool.detach())                
                elif ptype == 'sum':
                    if losstype == 'KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())
                    elif losstype == 'JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())                
            else:
                tmp_maskstr = 'mask_' + str(cidx)
                tmp_mask_right = mask_dict[tmp_maskstr]

                def mask_softmax(pre, mask, dim=1, ifexp=True):
                    if ifexp:
                        prenorm = torch.exp(pre)
                    prenorm = prenorm * mask
                    pre_sum = prenorm.sum(dim=dim, keepdim=True)
                    EPS = 1e-6
                    prenorm = prenorm / torch.where(pre_sum < EPS, pre_sum + EPS, pre_sum)
                    return prenorm

                if ptype == 'max':
                    adjust_logit_pool = mask_softmax(adjust_logit_pool, tmp_mask_right, 
                                                   dim=1, ifexp=True)
                    tmp_fine_logits_pool = mask_softmax(tmp_fine_logits_pool, tmp_mask_right, 
                                                      dim=1, ifexp=True)
                    if losstype == 'KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())
                    elif losstype == 'JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())                
                elif ptype == 'sum':
                    adjust_logit_pool = mask_softmax(adjust_logit_pool, tmp_mask_right, 
                                                   dim=1, ifexp=False)
                    tmp_fine_logits_pool = mask_softmax(tmp_fine_logits_pool, tmp_mask_right, 
                                                      dim=1, ifexp=False)
                    if losstype == 'KL':
                        tmp_KL = myKL_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())
                    elif losstype == 'JS':
                        tmp_KL = myJS_div_prob(adjust_logit_pool, tmp_fine_logits_pool.detach())  

            ret_KL.append(tmp_KL)

        ret_KL = torch.cat(ret_KL, dim=1)

        if mask_type == 'all':
            if len(ret_KL.size()) == 2:
                ret_KL = (ret_KL).mean(dim=0)
            else:
                ret_KL = (ret_KL).mean(dim=3).mean(dim=2).mean(dim=0)
        elif mask_type == 'best':
            best_mask = self.get_mask_gt2(allpre, gt)
            if len(ret_KL.size()) == 2:
                ret_KL = (ret_KL * best_mask).mean(dim=0)
            else:
                ret_KL = (ret_KL * best_mask).mean(dim=3).mean(dim=2).mean(dim=0)

        return ret_KL


def create_mlp(
    input_dim: int = 100,
    output_dim: int = 1,
    hiddens: Union[list[int], None] = None,
    skipped_layers: int = 1,
    embed_dim: int = 128,
    skip_repr: bool = True,
    activation: Union[nn.Module, None] = None,
    dropout: float = 0.2,
    name: str = 'mlp',
    fds: bool = False,
    bucket_num: int = 50,
    bucket_start: int = 0,
    start_update: int = 0,
    start_smooth: int = 1,
    kernel: str = 'gaussian',
    ks: int = 5,
    sigma: float = 2.0,
    momentum: float = 0.9,
    # Multi-head hierarchical classification parameters
    class_indice: Optional[np.ndarray] = None,
    class2count: Optional[np.ndarray] = None,
    head_class_indice: List = [],
    head_class2count: List = [],
    head_weight: Optional[List] = None,
    cmax: Optional[int] = None,
    fc_lnum: int = 1,
    class_weight: Optional[torch.Tensor] = None,
    head_detach: bool = True,
    s2fc_lnum: int = 2
) -> MLP:
    """
    Creates an MLP instance for regression or hierarchical classification, supporting optional 
    skip connections, batch normalization, dropout, and FDS smoothing.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features (for regression) or ignored if hierarchical classification enabled.
        hiddens (list[int]): Hidden layer sizes (defaults to [50, 50] if None).
        skipped_layers (int): Skip connection frequency.
        embed_dim (int): Size of the final embedding.
        skip_repr (bool): If True, merges skip into the final embedding block.
        activation (nn.Module): Activation function, defaults to LeakyReLU if None.
        dropout (float): Dropout probability.
        name (str): Model name, unused in PyTorch but kept for reference.
        fds (bool): If True, enable FDS smoothing.
        bucket_num, bucket_start, start_update, start_smooth, kernel, ks, sigma, momentum:
            Hyperparameters for FDS.
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

    Returns:
        MLP: A PyTorch MLP module with the specified configuration.
    """
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        activation=activation,
        dropout=dropout,
        name=name,
        fds=fds,
        bucket_num=bucket_num,
        bucket_start=bucket_start,
        start_update=start_update,
        start_smooth=start_smooth,
        kernel=kernel,
        ks=ks,
        sigma=sigma,
        momentum=momentum,
        class_indice=class_indice,
        class2count=class2count,
        head_class_indice=head_class_indice,
        head_class2count=head_class2count,
        head_weight=head_weight,
        cmax=cmax,
        fc_lnum=fc_lnum,
        class_weight=class_weight,
        head_detach=head_detach,
        s2fc_lnum=s2fc_lnum
    )
    return model
