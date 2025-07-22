#train.py
import time
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import gmean
from typing import Dict, Tuple, Optional, Union
from balanaced_mse import GAILoss, BMCLoss, BNILoss

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss import *
from tab_ds import TabDS, load_tabular_splits, set_seed
from mlp import create_mlp
from utils import *
from balanaced_mse import *
from metrics import (
    evaluate_sep_metrics, evaluate_ed_metrics, evaluate_onp_metrics,
    evaluate_sarcos_metrics, evaluate_bf_metrics, evaluate_asc_metrics,
    save_results_to_csv
)
# RankSim and ConR imports
from ranksim import batchwise_ranking_regularizer
# from debug_kde import debug_kde_visualization

# make only gpu:1 visible
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Disable KMP warnings
os.environ["KMP_WARNINGS"] = "FALSE"

# Set up command line argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ----- Imbalanced learning related arguments -----
# LDS (Label Distribution Smoothing)
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS (Feature Distribution Smoothing)
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket number for FDS and BNI Loss (TabDS uses DEFAULT_NUM_BINS=100)')
parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# Label value thresholds for data frequency categorization
parser.add_argument('--lower_threshold', type=float, default=None, help='lower threshold value for label range (rare/low-shot data)')
parser.add_argument('--upper_threshold', type=float, default=None, help='upper threshold value for label range (frequent/many-shot data)')

# BMSE (Balanced MSE)
parser.add_argument('--bmse', action='store_true', default=False, help='use Balanced MSE')
parser.add_argument('--imp', type=str, default='gai', choices=['gai', 'bmc', 'bni'], help='implementation options')
parser.add_argument('--gmm_file', type=str, default=None, help='Path to preprocessed GMM file (e.g., sep_gmm_K8.pkl). If None, constructed from dataset/K.')
parser.add_argument('--init_noise_sigma', type=float, default=1., help='initial scale of the noise')
parser.add_argument('--sigma_lr', type=float, default=1e-2, help='learning rate of the noise scale')
parser.add_argument('--fix_noise_sigma', action='store_true', default=False, help='disable joint optimization')

# Re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'],
                    help='cost-sensitive reweighting scheme')
# Two-stage training: RRT (Regressor Re-Training)
parser.add_argument('--retrain_regressor', action='store_true', default=False,
                    help='whether to retrain last regression layer (regressor) of MLP')

# PCC Regularization
parser.add_argument('--pcc_lambda', type=float, default=0.0, help='Weight for the Pearson Correlation Coefficient (1-PCC) regularizer.')

# RankSim regularization
parser.add_argument('--regularization_weight', type=float, default=0, help='weight of the ranking regularization term')
parser.add_argument('--interpolation_lambda', type=float, default=1.0, help='interpolation strength for ranking regularization')

# ConR (Contrastive Regularizer)
parser.add_argument('--conr', action='store_true', default=False, help='whether to enable ConR')
parser.add_argument('-w', type=float, default=1, help='similarity window for ConR loss')
parser.add_argument('--beta', type=float, default=4, help='ConR loss coefficient')
parser.add_argument('-t', type=float, default=.07, help='temperature parameter for ConR')
parser.add_argument('-e', type=float, default=0.01, help="coefficient for eta in ConR")

# ----- Training/optimization related arguments -----
parser.add_argument('--dataset', type=str, required=True,
                    choices=['sep', 'sarcos', 'onp', 'bf', 'asc', 'ed'],
                    help='Name of the tabular dataset to use.')
parser.add_argument('--data_dir', type=str, default='/mnt/c/Users/the_3/Documents/github/BalancedMSE/neurips2025/data', help='Root directory containing dataset subfolders.')
parser.add_argument('--train_split_name', type=str, default='training', help='Name for the training data file/folder.')
parser.add_argument('--val_split_name', type=str, default='validation', help='Name for the validation data file/folder.')
parser.add_argument('--test_split_name', type=str, default='testing', help='Name for the test data file/folder.')

# Added MLP specific args
parser.add_argument('--model', type=str, default='mlp', choices=['mlp'], help='model name')
parser.add_argument('--mlp_hiddens', type=int, nargs='+', default=[100, 100, 100], help='MLP hidden layer sizes')
parser.add_argument('--mlp_embed_dim', type=int, default=128, help='MLP embedding dimension (output of backbone)')
parser.add_argument('--mlp_skip_layers', type=int, default=1, help='MLP skip connection frequency')
parser.add_argument('--mlp_skip_repr', action='store_true', default=True, help='MLP merge skip into final representation')
parser.add_argument('--mlp_dropout', type=float, default=0.1, help='MLP dropout rate')

parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'],
                    help='training loss type')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int, default=100, help='logging frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers used in data loading')
# Checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')
parser.add_argument('--seeds', type=int, nargs='+', default=[456789], help='list of random seeds for multiple trials')
parser.add_argument('--pretrained', type=str, default='', help='pretrained model file path')

args, unknown = parser.parse_known_args()

# Initialize training state variables
args.start_epoch, args.best_loss = 0, 1e5

# Build experiment name based on configuration
if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_regressor:
    args.store_name += f'_retrainReg'
if args.bmse:
    args.store_name += f'_{args.imp}_{args.init_noise_sigma}_{args.sigma_lr}'
    if args.imp == 'gai':
        gmm_suffix = args.gmm_file.split('/')[-1].replace('.pkl','')
        args.store_name += f'_{gmm_suffix}'
    if args.fix_noise_sigma:
        args.store_name += '_fixNoise'
if args.pcc_lambda > 0:
    args.store_name += f'_pccL{args.pcc_lambda}'
if args.conr:
    args.store_name += f'_ConR_{args.beta}_w={args.w}'
if args.regularization_weight > 0:
    args.store_name += f'_ranksim_{args.regularization_weight}'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_epoch{args.epoch}"

# Create folders for storing results
prepare_folders(args)

# Set up logging
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

# Initialize TensorBoard logger
# tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)

# Wrapper for combining base loss with PCC regularizer
class CombinedLoss(nn.Module):
    def __init__(self, base_criterion, pcc_lambda):
        super().__init__()
        self.base_criterion = base_criterion
        self.pcc_lambda = pcc_lambda
        self.is_bmse = isinstance(base_criterion, (GAILoss, BMCLoss, BNILoss))

        # Expose noise_sigma if the base criterion has it
        if hasattr(self.base_criterion, 'noise_sigma'):
            self.noise_sigma = self.base_criterion.noise_sigma

    def forward(self, inputs, targets, weights=None):
        # Calculate base loss
        if self.is_bmse:
            base_loss = self.base_criterion(inputs, targets)
        else:
            # Standard losses expect weights
            base_loss = self.base_criterion(inputs, targets, weights)

        # Calculate PCC loss (always needs weights, default to ones if None)
        coreg_loss = weighted_coreg_loss(inputs, targets, weights)

        # Combine losses
        total_loss = base_loss + self.pcc_lambda * coreg_loss
        return total_loss

def run_trial(trial_seed):
    """
    Run a single trial with the specified random seed.
    
    Args:
        trial_seed: Random seed for this trial
    """
    # Set seed for reproducibility
    set_seed(trial_seed)
    print(f"=== Starting trial with seed: {trial_seed} ===")
    
    # Update the store name to include the seed
    original_store_name = args.store_name
    args.store_name = f"{original_store_name}_seed{trial_seed}"
    
    # Create folders for this trial
    prepare_folders(args)
    
    # Reset best loss for this trial
    args.best_loss = 1e5
    
    # Data preparation
    print('=====> Preparing data...')
    start_time = time.time()
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_splits(
            args.dataset, args.data_dir, args.train_split_name,
            args.val_split_name, args.test_split_name, trial_seed
        )
        print(f"Data loaded. Train: {X_train.shape}/{y_train.shape}, Val: {X_val.shape}/{y_val.shape}, Test: {X_test.shape}/{y_test.shape}")
        print(f'Data loading time: {time.time() - start_time:.2f} seconds')
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return

    # Get input dimension from training data
    input_dim = X_train.shape[1]
    print(f"Input feature dimension: {input_dim}")

    # Create TabDS datasets
    print("Creating TabDS datasets...")
    train_dataset = TabDS(
        X=X_train, y=y_train,
        reweight=args.reweight,
        lds=args.lds, 
        lds_kernel=args.lds_kernel, 
        lds_ks=args.lds_ks, 
        lds_sigma=args.lds_sigma,
        bins=args.bucket_num
    )
    
    # # ADD THIS: Debug KDE if LDS is enabled
    # if args.lds and trial_seed == args.seeds[0]:  # Only debug for first trial
    #     print("\n" + "="*60)
    #     print("DEBUGGING KDE FOR LDS")
    #     print("="*60)
    #     debug_result = debug_kde_visualization(
    #         dataset=args.dataset,
    #         data_dir=args.data_dir,
    #         seed=trial_seed,
    #         lds=args.lds,
    #         lds_kernel=args.lds_kernel,
    #         lds_ks=args.lds_ks,
    #         lds_sigma=args.lds_sigma,
    #         bins=args.bucket_num,
    #         output_dir=os.path.join(args.store_root, args.store_name, 'kde_debug'),
    #         save_plots=True
    #     )
    #     print("KDE debugging completed.")
    #     print("="*60 + "\n")
    
    val_dataset = TabDS(X=X_val, y=y_val, reweight='none', lds=False)
    test_dataset = TabDS(X=X_test, y=y_test, reweight='none', lds=False)

    # Create data loaders
    pin_memory = True if args.gpu is not None else False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model initialization
    print('=====> Building model...')
    model = create_mlp(
        input_dim=input_dim,
        output_dim=1,
        hiddens=args.mlp_hiddens,
        skipped_layers=args.mlp_skip_layers,
        embed_dim=args.mlp_embed_dim,
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
        momentum=args.fds_mmt
    )

    model = torch.nn.DataParallel(model).cuda() if args.gpu is not None else model.cpu()

    # Evaluation mode - load model and evaluate
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, 'ckpt.best.pth.tar')
        if not os.path.isfile(ckpt_path):
            print(f"Error: Checkpoint file not found at {ckpt_path}")
            return

        print(f"Loading checkpoint for evaluation: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=True)
        print(f"===> Checkpoint '{ckpt_path}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, prefix='Test')

        # Evaluate dataset-specific metrics
        print("\nEvaluating specialized metrics...")
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cpu')
        
        # Use appropriate metrics function based on dataset
        if args.dataset == 'sep':
            metrics = evaluate_sep_metrics(
                model, X_train, y_train, X_test, y_test, 
                sep_threshold=np.log(10), device=device
            )
        elif args.dataset == 'ed':
            metrics = evaluate_ed_metrics(
                model, X_train, y_train, X_test, y_test,
                mae_plus_threshold=0.5, device=device
            )
        elif args.dataset == 'onp':
            metrics = evaluate_onp_metrics(
                model, X_train, y_train, X_test, y_test, 
                rare_low_threshold=np.log10(350),
                rare_high_threshold=np.log10(35000),
                device=device
            )
        elif args.dataset == 'sarcos':
            metrics = evaluate_sarcos_metrics(
                model, X_train, y_train, X_test, y_test,
                lower_threshold=-0.5, upper_threshold=0.5,
                device=device
            )
        elif args.dataset == 'bf':
            metrics = evaluate_bf_metrics(
                model, X_train, y_train, X_test, y_test,
                freq_threshold=np.log10(4), rare_threshold=np.log10(40),
                device=device
            )
        elif args.dataset == 'asc':
            metrics = evaluate_asc_metrics(
                model, X_train, y_train, X_test, y_test,
                rare_low_threshold=np.log10(200),
                rare_high_threshold=np.log10(20000),
                device=device
            )
        
        # Print metrics in a formatted way
        print("\n===== Specialized Metrics =====")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Incorporate specialized metrics in the return values
        return test_loss_mse, test_loss_l1, test_loss_gmean, metrics

    # For retraining only the final layer (transfer learning)
    if args.retrain_regressor:
        assert args.reweight != 'none' and args.pretrained or args.bmse
        print('===> Retrain last regression layer (output_layer) only!')
        for name, param in model.named_parameters():
            if 'output_layer' not in name:
                param.requires_grad = False
            else:
                print(f"Keeping param trainable: {name}")

    # Set up optimizer
    if not args.retrain_regressor:
        # Optimize all parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # Optimize only the last linear layer parameters
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None,
                            [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load pretrained weights if specified
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # Only load non-classifier weights
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    # Enable CUDA optimization
    cudnn.benchmark = True

    # --- Set up loss function --- START ---
    base_criterion = None
    if args.bmse:
        if args.imp == 'gai':
            gmm_path = args.gmm_file
            if not os.path.exists(gmm_path):
                raise FileNotFoundError(f"Specified GMM file not found: {gmm_path}")
            print(f"Loading GMM parameters from: {gmm_path}")
            device_for_loss = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
            base_criterion = GAILoss(args.init_noise_sigma, gmm_path, device=device_for_loss)
        elif args.imp == 'bmc':
            base_criterion = BMCLoss(args.init_noise_sigma)
        elif args.imp == 'bni':
            print("Fetching bucket info for BNI loss...")
            bucket_centers, bucket_weights = train_dataset.get_bucket_info(
                bins=args.bucket_num,
                lds=args.lds, lds_kernel=args.lds_kernel,
                lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
            print(f"Obtained {len(bucket_centers)} buckets for BNI.")
            base_criterion = BNILoss(args.init_noise_sigma, bucket_centers, bucket_weights)
        else:
            raise NotImplementedError(f"BMSE implementation '{args.imp}' not supported.")
    else:
        # Use standard weighted loss functions
        base_criterion = globals()[f"weighted_{args.loss}_loss"]

    # Move base criterion to GPU if needed (standard losses are functions, BMSE are modules)
    if isinstance(base_criterion, nn.Module) and args.gpu is not None:
        base_criterion.to(f'cuda:{args.gpu}')

    # Wrap with CombinedLoss if PCC regularization is enabled
    if args.pcc_lambda > 0:
        print(f"Combining base loss with PCC regularizer (lambda={args.pcc_lambda})")
        criterion = CombinedLoss(base_criterion, args.pcc_lambda)
        # Move the wrapper to GPU if necessary (contains the base criterion module)
        if args.gpu is not None:
            criterion.to(f'cuda:{args.gpu}')
    else:
        criterion = base_criterion # Use base criterion directly
    # --- Set up loss function --- END ---

    # Add noise sigma parameter to optimizer if not fixed
    if args.bmse and not args.fix_noise_sigma:
        # Check if noise_sigma exists either directly or within the wrapper
        noise_sigma_param = None
        if hasattr(criterion, 'noise_sigma'):
            noise_sigma_param = criterion.noise_sigma
        elif hasattr(criterion, 'base_criterion') and hasattr(criterion.base_criterion, 'noise_sigma'):
             noise_sigma_param = criterion.base_criterion.noise_sigma
             
        if noise_sigma_param is not None:
             optimizer.add_param_group({'params': noise_sigma_param, 'lr': args.sigma_lr, 'name': 'noise_sigma'})
             print(f"Added noise_sigma to optimizer with lr: {args.sigma_lr}")
        else:
             print(f"Warning: BMSE criterion ({args.imp}) or wrapper does not expose 'noise_sigma'. Cannot optimize it.")

    # Training loop
    for epoch in range(args.start_epoch, args.epoch):
        # Adjust learning rate according to schedule
        adjust_learning_rate(optimizer, epoch, args)
        
        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, criterion)
        
        # Evaluate on validation set
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, prefix='Val')

        # Determine which metric to use for model selection
        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
            
        # Check if current model is the best so far
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best Validation {'MSE' if 'mse' in args.loss else 'L1'} Loss: {args.best_loss:.4f}")
        
        # Save checkpoint
        state_dict_to_save = model.state_dict()
        enhanced_save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': state_dict_to_save,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }, is_best, epoch + 1 == args.epoch)
        
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        # Log metrics to TensorBoard
        # trial_logger.log_value('train_loss', train_loss, epoch)
        # trial_logger.log_value('val/loss_mse', val_loss_mse, epoch)
        # trial_logger.log_value('val/loss_l1', val_loss_l1, epoch)
        # trial_logger.log_value('val/loss_gmean', val_loss_gmean, epoch)
        
        # for i, param_group in enumerate(optimizer.param_groups):
            # trial_logger.log_value(f'lr/group_{i}', param_group['lr'], epoch)
        # if args.bmse and not args.fix_noise_sigma:
            # Correctly access noise_sigma, potentially through the wrapper
            # noise_sigma_val = None
            # if hasattr(criterion, 'noise_sigma'):
                 # noise_sigma_val = criterion.noise_sigma.item()
            # elif hasattr(criterion, 'base_criterion') and hasattr(criterion.base_criterion, 'noise_sigma'):
                 # noise_sigma_val = criterion.base_criterion.noise_sigma.item()
                 
            # if noise_sigma_val is not None:
                # trial_logger.log_value('noise_sigma', noise_sigma_val, epoch)

    # Test with best checkpoint after training
    print("=" * 120)
    print("Testing best model on testset...")
    best_ckpt_path = os.path.join(args.store_root, args.store_name, 'ckpt.best.pth.tar')
    if not os.path.exists(best_ckpt_path):
        print(f"Error: Best checkpoint not found at {best_ckpt_path}")
        return

    checkpoint = torch.load(best_ckpt_path)
    print(f"Loaded best model from epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")

    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=True)

    # Evaluate on test set
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(test_loader, model, prefix='Test')
    
    print(f"Test Results: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")

    # Evaluate dataset-specific metrics
    print("\nEvaluating specialized metrics...")
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Use appropriate metrics function based on dataset
    if args.dataset == 'sep':
        metrics = evaluate_sep_metrics(
            model, X_train, y_train, X_test, y_test, 
            sep_threshold=np.log(10), device=device
        )
    elif args.dataset == 'ed':
        metrics = evaluate_ed_metrics(
            model, X_train, y_train, X_test, y_test,
            mae_plus_threshold=0.5, device=device
        )
    elif args.dataset == 'onp':
        metrics = evaluate_onp_metrics(
            model, X_train, y_train, X_test, y_test, 
            rare_low_threshold=np.log10(350),
            rare_high_threshold=np.log10(35000),
            device=device
        )
    elif args.dataset == 'sarcos':
        metrics = evaluate_sarcos_metrics(
            model, X_train, y_train, X_test, y_test,
            lower_threshold=-0.5, upper_threshold=0.5,
            device=device
        )
    elif args.dataset == 'bf':
        metrics = evaluate_bf_metrics(
            model, X_train, y_train, X_test, y_test,
            freq_threshold=np.log10(4), rare_threshold=np.log10(40),
            device=device
        )
    elif args.dataset == 'asc':
        metrics = evaluate_asc_metrics(
            model, X_train, y_train, X_test, y_test,
            rare_low_threshold=np.log10(200),
            rare_high_threshold=np.log10(20000),
            device=device
        )
    
    # Print metrics in a formatted way
    print("\n===== Specialized Metrics =====")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Incorporate specialized metrics in the return values
    return test_loss_mse, test_loss_l1, test_loss_gmean, metrics

def main():
    """
    Run multiple trials with different seeds.
    """
    if args.gpu is not None:
        if torch.cuda.is_available():
            print(f"Using GPU: {args.gpu} for training")
        else:
            print("CUDA not available, using CPU.")
            args.gpu = None
    
    # Store original store name before adding seed
    original_store_name = args.store_name
    
    # Track metrics across all trials
    all_test_metrics = []
    
    # Run each trial with a different seed
    for trial_idx, seed in enumerate(args.seeds):
        print(f"\n\n{'='*50}")
        print(f"TRIAL {trial_idx+1}/{len(args.seeds)} - SEED {seed}")
        print(f"{'='*50}\n")
        
        # Reset store name for each trial
        args.store_name = original_store_name
        
        # Run the trial
        trial_results = run_trial(seed)
        if trial_results:
            test_mse, test_l1, test_gmean, specialized_metrics = trial_results
            all_test_metrics.append((test_mse, test_l1, test_gmean))
            
            # Track specialized metrics across trials
            if trial_idx == 0:
                all_specialized_metrics = {k: [] for k in specialized_metrics.keys()}
            
            for k, v in specialized_metrics.items():
                all_specialized_metrics[k].append(v)
    
    # Print summary of all trials if we have results
    if all_test_metrics:
        print("\n\n" + "="*80)
        print(f"SUMMARY OF {len(args.seeds)} TRIALS")
        print("="*80)
        
        # Calculate mean and std of metrics across trials
        mse_values = [m[0] for m in all_test_metrics]
        l1_values = [m[1] for m in all_test_metrics]
        gmean_values = [m[2] for m in all_test_metrics]
        
        mse_mean, mse_std = np.mean(mse_values), np.std(mse_values)
        l1_mean, l1_std = np.mean(l1_values), np.std(l1_values)
        gmean_mean, gmean_std = np.mean(gmean_values), np.std(gmean_values)
        
        print(f"Test MSE: {mse_mean:.4f} ± {mse_std:.4f}")
        print(f"Test L1: {l1_mean:.4f} ± {l1_std:.4f}")
        print(f"Test G-Mean: {gmean_mean:.4f} ± {gmean_std:.4f}")

        # After reporting MSE, L1, and G-Mean
        print("\n----- Specialized Metrics -----")
        specialized_metrics_dict = {}
        for metric_name, values in all_specialized_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            specialized_metrics_dict[metric_name] = (mean_val, std_val)
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results to CSV
        standard_metrics = (mse_mean, mse_std, l1_mean, l1_std, gmean_mean, gmean_std)
        csv_path = save_results_to_csv(args, specialized_metrics_dict, standard_metrics)
        print(f"Complete benchmark results saved to: {csv_path}")


        
def train(train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, criterion: nn.Module) -> float:
    """
    Train the model for one epoch.
    
    Args:
        train_loader: DataLoader for training data
        model: The neural network model
        optimizer: Optimizer for updating model weights
        epoch: Current epoch number
        criterion: Loss function
        
    Returns:
        float: Average training loss for the epoch
    """
    # Initialize meters for tracking time and performance
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    
    loss_name = f'Loss ({args.imp.upper()})' if args.bmse else f'Loss ({args.loss.upper()})'
    if args.pcc_lambda > 0:
        loss_name += f' + {args.pcc_lambda}*(1-PCC)'
    losses = AverageMeter(loss_name, ':.4f')
    
    meters_to_display = [batch_time, data_time, losses]
    
    # Add ConR loss meter if enabled
    if args.conr:
        loss_conr = AverageMeter('Loss(ConR)', ':.4f')
        meters_to_display.append(loss_conr)

    # Add additional metrics for balanced MSE if enabled
    if args.bmse:
        noise_var = AverageMeter('Noise Var', ':.5f')
        l2 = AverageMeter('L2', ':.5f')
        meters_to_display.extend([noise_var, l2])

    progress = ProgressMeter(len(train_loader), meters_to_display, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    
    # Iterate through batches
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            inputs, targets, weights = \
                inputs.cuda(args.gpu, non_blocking=True), \
                targets.cuda(args.gpu, non_blocking=True), \
                weights.cuda(args.gpu, non_blocking=True)
        else:
            inputs, targets, weights = inputs.cpu(), targets.cpu(), weights.cpu()

        if args.fds:
            predictions, features = model(inputs, targets, epoch)
        else:
            predictions, features = model(inputs, targets, epoch)

        predictions = predictions.squeeze(-1) if predictions.ndim > 1 and predictions.shape[-1] == 1 else predictions
        targets = targets.squeeze(-1) if targets.ndim > 1 and targets.shape[-1] == 1 else targets
        weights = weights.squeeze(-1) if weights.ndim > 1 and weights.shape[-1] == 1 else weights

        # Calculate loss using the (potentially wrapped) criterion
        if args.bmse and args.pcc_lambda == 0:
            loss = criterion(predictions, targets)  # No weights for pure BMSE
        else:
            loss = criterion(predictions, targets, weights)  # Weights for standard losses or PCC

        # Add ConR loss if enabled
        if args.conr:
            l_conr = ConR(
                        features, 
                        targets, 
                        predictions.clone().detach(), 
                        w=args.w, 
                        weights=weights, 
                        t=args.t, 
                        e=args.e
                    )
            loss += args.beta * l_conr
            loss_conr.update(l_conr.item(), inputs.size(0))

        # Add RankSim regularization if enabled
        if args.regularization_weight > 0:
            ranking_loss = batchwise_ranking_regularizer(features, targets, args.interpolation_lambda)
            loss += args.regularization_weight * ranking_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        # Update running loss average
        losses.update(loss.item(), inputs.size(0))

        # Track additional metrics for balanced MSE
        if args.bmse:
            if hasattr(criterion, 'noise_sigma') and criterion.noise_sigma is not None:
                noise_var.update(criterion.noise_sigma.item() ** 2)
            elif hasattr(criterion, 'base_criterion') and hasattr(criterion.base_criterion, 'noise_sigma') and criterion.base_criterion.noise_sigma is not None:
                 noise_var.update(criterion.base_criterion.noise_sigma.item() ** 2)
            l2.update(F.mse_loss(predictions, targets).item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress at specified intervals
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update and hasattr(model, 'fds_module') and model.fds_module is not None:
        print(f"Updating FDS statistics for Epoch [{epoch}]...")
        encodings, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader, desc="FDS Feature Extraction"):
                if args.gpu is not None:
                    inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
                else:
                    inputs, targets = inputs.cpu(), targets.cpu()

                _, feature = model(inputs, targets, epoch)

                encodings.append(feature.detach().cpu().numpy())
                labels_list.append(targets.detach().cpu().numpy())

        encodings = np.vstack(encodings)
        labels_array = np.concatenate([lbl.squeeze() if lbl.ndim > 1 and lbl.shape[-1] == 1 else lbl for lbl in labels_list])

        encodings_tensor = torch.from_numpy(encodings)
        labels_tensor = torch.from_numpy(labels_array)
        if args.gpu is not None:
            encodings_tensor = encodings_tensor.cuda(args.gpu)
            labels_tensor = labels_tensor.cuda(args.gpu)

        print(f"Calling FDS update with {encodings_tensor.shape} features and {labels_tensor.shape} labels.")
        fds_module = model.fds_module
        fds_module.update_last_epoch_stats(epoch)
        fds_module.update_running_stats(encodings_tensor, labels_tensor, epoch)
        print("FDS statistics updated.")
        model.train()

    return losses.avg


def validate(
        val_loader: DataLoader, 
        model: nn.Module,  
        prefix: str = 'Val'
    ) -> Tuple[float, float, float]:
    """
    Evaluate the model on validation or test data.
    
    Args:
        val_loader: DataLoader for validation/test data
        model: The neural network model
        prefix: Prefix for progress display ('Val' or 'Test')
        
    Returns:
        Tuple containing:
            - MSE loss average
            - L1 loss average
            - Geometric mean of losses
    """
    # Initialize meters for tracking time and performance
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.4f')
    losses_l1 = AverageMeter('Loss (L1)', ':.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    # Define loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')  # For calculating geometric mean

    # Set model to evaluation mode
    model.eval()
    all_losses_l1_for_gmean = []
    preds_list, labels_list = [], []

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            if args.gpu is not None:
                inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
            else:
                inputs, targets = inputs.cpu(), targets.cpu()

            predictions, _ = model(inputs, labels=None, epoch=None)

            predictions = predictions.squeeze(-1) if predictions.ndim > 1 and predictions.shape[-1] == 1 else predictions
            targets = targets.squeeze(-1) if targets.ndim > 1 and targets.shape[-1] == 1 else targets

            preds_list.append(predictions.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())

            loss_mse = criterion_mse(predictions, targets)
            loss_l1 = criterion_l1(predictions, targets)
            loss_l1_all = criterion_gmean(predictions, targets)
            all_losses_l1_for_gmean.append(loss_l1_all.detach().cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Print progress at specified intervals
            if idx % args.print_freq == 0:
                progress.display(idx)

        # Apply threshold-based metrics
        metrics_dict = threshold_metrics(
            np.hstack(preds_list), 
            np.hstack(labels_list),
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold
        )
        
        # Calculate geometric mean of all losses
        loss_gmean = gmean(np.hstack(all_losses_l1_for_gmean), axis=None).astype(float)
        
        # Print detailed performance metrics
        print(f" * Overall: MSE {metrics_dict['overall']['mse']:.3f}\t"
              f"L1 {metrics_dict['overall']['l1']:.3f}\t"
              f"G-Mean {metrics_dict['overall']['gmean']:.3f}\t"
              f"Count {metrics_dict['overall']['count']}")
        print('-' * 40)
        
        # Print metrics for each threshold range if applicable
        if 'below_lower' in metrics_dict:
            print(f" * Below {args.lower_threshold}: MSE {metrics_dict['below_lower']['mse']:.3f}\t"
                  f"L1 {metrics_dict['below_lower']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['below_lower']['gmean']:.3f}\t"
                  f"Count {metrics_dict['below_lower']['count']}")
        
        if 'middle' in metrics_dict:
            print(f" * Between {args.lower_threshold} and {args.upper_threshold}: MSE {metrics_dict['middle']['mse']:.3f}\t"
                  f"L1 {metrics_dict['middle']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['middle']['gmean']:.3f}\t"
                  f"Count {metrics_dict['middle']['count']}")
        
        if 'above_upper' in metrics_dict:
            print(f" * Above {args.upper_threshold}: MSE {metrics_dict['above_upper']['mse']:.3f}\t"
                  f"L1 {metrics_dict['above_upper']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['above_upper']['gmean']:.3f}\t"
                  f"Count {metrics_dict['above_upper']['count']}")
              
    return losses_mse.avg, losses_l1.avg, loss_gmean

def threshold_metrics(
        preds: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different value ranges based on thresholds.
    
    Args:
        preds: Model predictions
        labels: Ground truth labels
        lower_threshold: Lower threshold for label values
        upper_threshold: Upper threshold for label values
        
    Returns:
        Dictionary with metrics for each range (below_lower, middle, above_upper)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    # Initialize result dictionary
    metrics_dict = defaultdict(dict)
    
    # Calculate overall metrics
    overall_mse = np.mean((preds - labels) ** 2)
    overall_l1 = np.mean(np.abs(preds - labels))
    overall_l1_all = np.abs(preds - labels)
    overall_gmean = gmean(overall_l1_all, axis=None).astype(float)
    
    metrics_dict['overall'] = {
        'mse': overall_mse,
        'l1': overall_l1,
        'gmean': overall_gmean,
        'count': len(labels)
    }
    
    # If thresholds are provided, calculate metrics for each range
    if lower_threshold is not None or upper_threshold is not None:
        # Below lower threshold
        if lower_threshold is not None:
            below_mask = labels < lower_threshold
            if np.any(below_mask):
                below_mse = np.mean((preds[below_mask] - labels[below_mask]) ** 2)
                below_l1 = np.mean(np.abs(preds[below_mask] - labels[below_mask]))
                below_l1_all = np.abs(preds[below_mask] - labels[below_mask])
                below_gmean = gmean(below_l1_all, axis=None).astype(float) if len(below_l1_all) > 0 else 0
                
                metrics_dict['below_lower'] = {
                    'mse': below_mse,
                    'l1': below_l1,
                    'gmean': below_gmean,
                    'count': np.sum(below_mask)
                }
        
        # Above upper threshold
        if upper_threshold is not None:
            above_mask = labels > upper_threshold
            if np.any(above_mask):
                above_mse = np.mean((preds[above_mask] - labels[above_mask]) ** 2)
                above_l1 = np.mean(np.abs(preds[above_mask] - labels[above_mask]))
                above_l1_all = np.abs(preds[above_mask] - labels[above_mask])
                above_gmean = gmean(above_l1_all, axis=None).astype(float) if len(above_l1_all) > 0 else 0
                
                metrics_dict['above_upper'] = {
                    'mse': above_mse,
                    'l1': above_l1,
                    'gmean': above_gmean,
                    'count': np.sum(above_mask)
                }
        
        # Between thresholds
        if lower_threshold is not None and upper_threshold is not None:
            middle_mask = (labels >= lower_threshold) & (labels <= upper_threshold)
            if np.any(middle_mask):
                middle_mse = np.mean((preds[middle_mask] - labels[middle_mask]) ** 2)
                middle_l1 = np.mean(np.abs(preds[middle_mask] - labels[middle_mask]))
                middle_l1_all = np.abs(preds[middle_mask] - labels[middle_mask])
                middle_gmean = gmean(middle_l1_all, axis=None).astype(float) if len(middle_l1_all) > 0 else 0
                
                metrics_dict['middle'] = {
                    'mse': middle_mse,
                    'l1': middle_l1,
                    'gmean': middle_gmean,
                    'count': np.sum(middle_mask)
                }
    
    return metrics_dict

def enhanced_save_checkpoint(args, state, is_best, is_final=False):
    """
    Optimized checkpoint saving - only saves best and final models
    """
    try:
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(args.store_root, args.store_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Only save best and final checkpoints
        if is_best:
            best_filename = os.path.join(checkpoint_dir, 'ckpt.best.pth.tar')
            torch.save(state, best_filename + '.tmp')
            shutil.move(best_filename + '.tmp', best_filename)
            logging.info(f"Best checkpoint saved to: {best_filename}")
        
        if is_final:
            final_filename = os.path.join(checkpoint_dir, 'ckpt.final.pth.tar')
            torch.save(state, final_filename + '.tmp')
            shutil.move(final_filename + '.tmp', final_filename)
            logging.info(f"Final checkpoint saved to: {final_filename}")
    
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")




if __name__ == '__main__':
    main()
