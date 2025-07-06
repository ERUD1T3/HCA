import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
from loss import *
from datasets import IMDBWIKI
from utils import *
import os
os.environ["KMP_WARNINGS"] = "FALSE"
from resnet_cls_mh import resnet50_cls_mh
from mh_utils import level_split, get_mh_weight
from scipy.stats import *
from eval_metrics import shot_metrics_balanced

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
parser.add_argument('--reweight', type=str, default='sqrt_inv', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./imdbwiki_data', help='data directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=0, help='number of workers used in data loading')
parser.add_argument('--resume', type=str, default='./models/cls+lds+r/ckpt.best.pth.tar', help='checkpoint file path to resume training')
parser.add_argument('--log_dir',type=str,default='analy_log')
parser.add_argument('--log_name',type=str,default='comb_test.log')
parser.add_argument('--head_num', type=int, default=6, help = 'the number of hierarchical classification heads')
parser.add_argument('--fc_lnum', type=int, default=1)
parser.add_argument('--s2fc_lnum', type=int, default=2)
parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()
args.start_epoch, args.best_loss = 0, 1e5

def main():
    max_age = 120
    global_class_indice = np.linspace(1,max_age,max_age)-0.5 
    global_class2count = np.linspace(0,max_age,max_age+1)
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
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Test data size: {len(test_dataset)}")
    
    global_count_dict = {x: 0 for x in range(max_age+1)}
    labels = df_train['age'].values
    for label in labels:
        global_count_dict[min(max_age, int(label))] += 1
    global_count_dict_sqrt = dict()
    for label in range(max_age+1):
        global_count_dict_sqrt[label] = np.sqrt(global_count_dict[label])
    sinterval, sindice, sclass2count = level_split([0,max_age+0.01],global_count_dict_sqrt,args.head_num,if_age=True,ob_vmax=max_age)    
    sweight = get_mh_weight(global_count_dict,sindice)
    
    print('=====> Building model...')
    model = resnet50_cls_mh(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt
                     ,class_indice=global_class_indice,class2count=global_class2count,\
                    head_class_indice=sindice,head_class2count=sclass2count,\
                    head_weight=sweight,cmax=max_age,fc_lnum=args.fc_lnum,
                    s2fc_lnum=args.s2fc_lnum)
    model = torch.nn.DataParallel(model).cuda()
    
    assert args.resume, 'Specify a trained model using [args.resume]'
    if not ('.pth.tar' in args.resume):
        args.resume = os.path.join(args.resume,'ckpt.best.pth.tar')
    checkpoint = torch.load(args.resume)
    torch.save(checkpoint,args.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Now testing on the test set")
    validate(test_loader, model, train_labels=train_labels, prefix='Test')
    return


class age_evalautor():
    def __init__(self):
        self.pre=[]
        self.label=[]

    def update(self,tmp_pre,tmp_label):
        self.pre.append(tmp_pre.view(-1,1))
        self.label.append(tmp_label.view(-1,1))

    def print_str(self):
        pass

    def evaluate(self, train_labels):
        tmp_pre = torch.cat(self.pre,dim=0).view(-1)
        tmp_label = torch.cat(self.label,dim=0).view(-1)
        self.result = dict()
        self.result['bal'] = shot_metrics_balanced(tmp_pre, tmp_label, train_labels)
        return self.result

    def print_result(self, bstr='bal'):
        shot_dict = self.result[bstr]
        return_str = ''  
        return_str += '%10s&\t%10s&\t%10s&\t%10s&\t%10s&\t\n' %('Metrics', 'Overall','Many','Medium','Few')
        tkey = 'l1'
        return_str += '%10s&\t%10.2f&\t%10.2f&\t%10.2f&\t%10.2f&\t\n' %(tkey,shot_dict['all'][tkey],shot_dict['many'][tkey],shot_dict['median'][tkey],shot_dict['low'][tkey] )
        return return_str

def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    
    preds, labels = [], []

    head_num = model.module.head_num
    
    eval_dict = dict()
    eval_dict['singlecls'] = age_evalautor()
    eval_dict['adjcls'] = age_evalautor()
    eval_dict['combadd'] = age_evalautor()
    eval_dict['combmul'] = age_evalautor()



    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            allpre = model(inputs)
            outputs = allpre['x']
            outputs = model.module.Class2Count(outputs)
            # update the dict
            eval_dict['singlecls'].update(outputs.cpu(),targets.cpu())
            adjoutput = allpre['adjust_x']
            adjoutput = model.module.Class2Count(adjoutput)
            eval_dict['adjcls'].update(adjoutput.cpu(),targets.cpu())
        
            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())
            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
            
            comb_dict =  model.module.hier_combine_test(allpre,if_entropy=False)
            eval_dict['combadd'].update(comb_dict['addv'].cpu().detach(),targets.cpu())
            eval_dict['combmul'].update(comb_dict['mulv'].cpu().detach(),targets.cpu())
        
        result_dict = dict()
        print_str = ''
        bstr = 'bal'
        print_str+='Now is %s metrics\n' %(bstr)
        for tkey in list(eval_dict.keys()):
            result_dict[tkey] = eval_dict[tkey].evaluate(train_labels)
            return_str = eval_dict[tkey].print_result(bstr=bstr)
            print_str += '<%s> \n' % (tkey)
            print_str += return_str
            print_str += '-*'*20 +'\n'
        print_str += '=='*20 +'\n'
        print(print_str)     
    return result_dict




if __name__ == '__main__':
    main()
    
