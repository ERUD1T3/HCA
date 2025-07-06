from calendar import c
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fds import FDS

# this is classification related functions
from cls_funcs import Count2Class,Class2Count

print = logging.info

def myJS_div(logit1,logit2):
    net_1_probs =  F.softmax(logit1, dim=1)
    net_2_probs=  F.softmax(logit2, dim=1)

    m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(logit1, dim=1), m, reduction="none",log_target=False).sum(dim=1,keepdim=True) 
    loss += F.kl_div(F.log_softmax(logit2, dim=1), m, reduction="none",log_target=False).sum(dim=1,keepdim=True) 
    return (0.5 * loss)                
    
def myKL_div(logit1,logit2): #logit1 is input,logit2 is target
    loss = 0.0
    loss += F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction="none",log_target=False).sum(dim=1,keepdim=True) 
    return loss

def myKL_div_prob(prob1,prob2): #logit1 is input,logit2 is target
    loss = 0.0
    EPS=1e-6
    prob1 = torch.where(prob1>=EPS,prob1,prob1+EPS)
    prob2 = torch.where(prob2>=EPS,prob2,prob2+EPS)
    loss += F.kl_div(torch.log(prob1), torch.log(prob2), reduction="none",log_target=True).sum(dim=1,keepdim=True) 
    return loss

def myJS_div_prob(net_1_probs,net_2_probs):
    m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    EPS=1e-6
    net_1_probs = torch.where(net_1_probs>=EPS,net_1_probs,net_1_probs+EPS)
    net_2_probs = torch.where(net_2_probs>=EPS,net_2_probs,net_2_probs+EPS)
    loss += F.kl_div(torch.log(net_1_probs), m, reduction="none",log_target=False).sum(dim=1,keepdim=True) 
    loss += F.kl_div(torch.log(net_2_probs), m, reduction="none",log_target=False).sum(dim=1,keepdim=True) 

    return (0.5 * loss)  


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_cls_mh(nn.Module):

    def __init__(self, block, layers, fds, bucket_num, bucket_start, start_update, start_smooth,
                 kernel, ks, sigma, momentum, dropout=None,class_indice=None,class2count=None,\
                head_class_indice=[],head_class2count=[],head_weight=None, cmax=None,fc_lnum=1,\
                class_weight=None,head_detach=True,s2fc_lnum=2):
        self.inplanes = 64
        super(ResNet_cls_mh, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.class_indice = torch.Tensor(class_indice).float()
        self.class2count = torch.Tensor(class2count).float()
        assert len(self.class_indice)+1==len(self.class2count)
        self.cnum = len(class2count)
        self.linear = nn.Linear(512 * block.expansion, self.cnum)
        self.class_weight = class_weight

        if fds:
            self.FDS = FDS(
                feature_dim=512 * block.expansion, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma, momentum=momentum
            )
        self.fds = fds
        self.start_smooth = start_smooth

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        # this is for initializing
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.cmax = cmax
        self.head_num = len(head_class2count)
        self.head_class_indice = head_class_indice
        self.head_class2count = head_class2count
        self.head_weight = head_weight
        self.fc_lnum = fc_lnum

        self.head_detach=head_detach
        if self.head_num>0:
            for hi in range(self.head_num):
                tmp_cnum = len(self.head_class2count[hi])
                if fc_lnum==1:
                    setattr(self,'cls_head_%d' %(hi),\
                        nn.Sequential(
                        *[nn.Linear(512 * block.expansion, tmp_cnum)])
                        )
                elif fc_lnum==2:
                    setattr(self,'cls_head_%d' %(hi),\
                        nn.Sequential(
                        *[nn.Linear(512 * block.expansion, 512),
                        nn.ReLU(),  
                        nn.Linear(512, tmp_cnum)])
                        )

     
        self.s2fc_lnum = s2fc_lnum
        if self.s2fc_lnum==1:
            setattr(self,'adjust_head',\
                nn.Sequential(
                *[nn.Linear(512 * block.expansion, self.cnum)])
                )
        elif self.s2fc_lnum==2:
            setattr(self,'adjust_head',\
                nn.Sequential(
                *[nn.Linear(512 * block.expansion, 512),
                nn.Softplus(),
                nn.Linear(512, self.cnum)])
                )
        elif self.s2fc_lnum==3:
            setattr(self,'adjust_head',\
                nn.Sequential(
                *[nn.Linear(512 * block.expansion, 512),
                nn.Softplus(),  
                nn.Linear(512, 256),
                nn.Softplus(),
                nn.Linear(256, self.cnum)])
                )           

        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def get_feat(self,x): 
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

        return encoding
    
    
    def forward_feat(self, encoding, targets=None, epoch=None):
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        
        x = self.linear(encoding_s) 

        if self.head_num>0:
            mhpre_dict = dict()
            for hi in range(self.head_num):
                tmp_save_name = 'cls_%d' %(hi)
                if self.head_detach:
                    mhpre_dict[tmp_save_name] = getattr(self,'cls_head_%d' %(hi))(encoding_s.detach())
                else:
                    mhpre_dict[tmp_save_name] = getattr(self,'cls_head_%d' %(hi))(encoding_s)


        res_dict = dict()
        res_dict['adjust_x'] = getattr(self,'adjust_head' )(encoding_s.detach())  
        res_dict['x'] = x
        res_dict['encoding'] = encoding
        res_dict['mhpre_dict'] = mhpre_dict
        return res_dict

    def forward(self, x, targets=None, epoch=None):
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

        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        
        x = self.linear(encoding_s) 

        if self.head_num>0:
            mhpre_dict = dict()
            for hi in range(self.head_num):
                tmp_save_name = 'cls_%d' %(hi)
                if self.head_detach:
                    mhpre_dict[tmp_save_name] = getattr(self,'cls_head_%d' %(hi))(encoding_s.detach())
                else:
                    mhpre_dict[tmp_save_name] = getattr(self,'cls_head_%d' %(hi))(encoding_s)

        res_dict = dict()
        res_dict['adjust_x'] = getattr(self,'adjust_head' )(encoding_s.detach())  
        res_dict['x'] = x
        res_dict['encoding'] = encoding
        res_dict['mhpre_dict'] = mhpre_dict
        return res_dict

    def Count2Class(self,gtcounts):
        gtclass = Count2Class(gtcounts,self.class_indice)
        return gtclass

    def Class2Count(self,preclass):
        precounts = Class2Count(preclass,self.class2count)
        return precounts

    def Count2Class_mh(self,gtcounts,class_indice):
        gtclass = Count2Class(gtcounts,class_indice)
        return gtclass
    
    def Class2Count_mh(self,preclass,class2count):
        precounts = Class2Count(preclass,class2count)
        return precounts

    def Count2Class_idx(self,gtcounts,level):
        if level<self.head_num:
            class_indice = self.head_class_indice[level]
            class_indice = torch.Tensor(class_indice).float().cuda()
        else:
            class_indice = self.class_indice    
        gtclass = Count2Class(gtcounts,class_indice)
        return gtclass
    
    def Class2Count_idx(self,preclass,level):
        if level<self.head_num:
            class2count = self.head_class2count[level]
            class2count = torch.Tensor(class2count).float().cuda()
        else:
            class2count = self.class2count          
        precounts = Class2Count(preclass,class2count)
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
def resnet50_cls_mh(**kwargs):
    return ResNet_cls_mh(Bottleneck, [3, 4, 6, 3], **kwargs)