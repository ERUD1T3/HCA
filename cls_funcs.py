# -*- coding: utf-8 -*-
# author: xhp


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# content
# -- Count2Class
# -- Class2Count
# -- UEP_order
# -- interval_divide

#  now should tell, class_indice should be start from V_{min}, and not include V_{max}
# interval_divide is the interval number




# change value to class with label_indice
# class indice do not contain 0, also dot not conatin C_{max}
# output is the class map, the same size as countmap
def Count2Class(count_map,label_indice):
    # if isinstance(label_indice,np.ndarray):
    #     label_indice = torch.from_numpy(label_indice) 
    IF_ret_np = False
    if isinstance(count_map,np.ndarray):
        count_map = torch.from_numpy(count_map)
        IF_ret_np = True
    # try to compute on the gpu
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (count_map.device.type == 'cuda') 

    # label_indice  = label_indice.cpu().type(torch.float32)
    cls_num = len(label_indice)+1
    cls_map = torch.zeros(count_map.size()).type(torch.LongTensor) 
    if IF_gpu:
        count_map,cls_map = count_map.cuda(),cls_map.cuda()
        # count_map,label_indice,cls_map = count_map.cuda(),label_indice.cuda(),cls_map.cuda()
    
    for i in range(len(label_indice)):
        if IF_gpu:
            cls_map = cls_map + (count_map >= label_indice[i].item()).long()#.cpu().type(torch.LongTensor).cuda()
        else:
            cls_map = cls_map + (count_map >= label_indice[i].item()).long()#.cpu().type(torch.LongTensor)
    
    if not IF_ret_gpu:
        cls_map = cls_map.cpu() 
    if IF_ret_np:
        cls_map = cls_map.cpu().numpy()
    return cls_map



#  convert class (0->c-1) to count number
# label2count should contain c element, which denotes the return value of the class
def Class2Count(pre_cls,label2count): 
    '''
    # --Input:
    # 1.pre_cls is class label range in [0,1,2,...,C-1]
    # 2.label2count is the proxy value of each class
    # --Output:
    # 1.count value, the same size as pre_cls
    '''   
    # check for the input type
    if isinstance(label2count,np.ndarray):
        label2count = torch.from_numpy(label2count).float()
    label2count = label2count.squeeze()
    # whether use gpu or not
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (pre_cls.device.type == 'cuda')  


    # if logits, change into class
    if pre_cls.size()[1]>1:
        pre_cls = pre_cls.max(dim=1,keepdim=True)[1].cpu().long() # here only need the place
    ORI_SIZE = pre_cls.size()


    # recover count from class

    pre_counts = torch.index_select(label2count.cuda(),0,pre_cls.cuda().view(-1))
    pre_counts = pre_counts.view(ORI_SIZE)

    if not IF_ret_gpu:
        pre_counts = pre_counts.cpu() # the output should be the same device as input

    return pre_counts

# Cmin-Cmax has 50 indices; Cmin>0, so denotes 51 classes 
def UEP_border(patch_count,vmin=5e-3,vmax=48.5,num=50,epslon=50):
    # first delete values that is less than vmin
    cls_border = np.zeros(num)
    # remove less than vmin
    del_idx = np.where(patch_count < vmin)
    patch_count = np.delete(patch_count,del_idx)
    # remove vmax from patch_count
    del_idx = np.where(patch_count >= vmax )
    patch_count = np.delete(patch_count,del_idx)
    
    # get the number of unique elements
    count_set,elem_num = np.unique(patch_count,return_counts=True)
    ind = np.argsort(count_set) # from low to high
    count_set, elem_num = count_set[ind],elem_num[ind]
    
    # intialize L
    L = 0
    # initialize H with uniform partition
    step = (vmax-vmin)/(num-1)
    N1 = (patch_count<step).sum()
    H = step*N1
    # Begin Binary Search
    iter_i = 0
    while (H-L)>epslon or (len(P)!=num-1):
        pp = vmin
        n = 0
        P = [vmin]
        pl = (L+H)/2
        # divide all the interval by l
        for idx,value in enumerate(count_set):
            n = n+elem_num[idx].item()
            if (value-pp)*n>pl:
                n = 0
                p = value
                P.append(value)
        
        # begin adjust [L,H] by |P| and num
        if len(P)>=num:
            L=pl
        else:
            if len(P)==num-1:
                if (vmax-p)*n>pl:
                    L = pl
                elif (vmax-p)*n<pl:
                    H = pl
                else:
                    H=L #exit
            else:
                H = pl
        print('%d-th iteration with %d/%d border points' %(iter_i,len(P),num-1) )
        iter_i = iter_i +1
        
        cls_border = P
        add = np.array([vmax])
        cls_border = np.concatenate((cls_border,add),axis=0)  
    return cls_border,pl # pl = nili



# [0,Vmin] is the first index
# return cls_border should be num-1: vmin, ...,vmax
# cls2value should be num: 0, v1,...,v_{num-1}
# # parameters
# cls_parse could be 'mean' or 'median'
# partition could be 'linear','log', or 'uep'
# the final indice is num; if directly used, then class is num+1
def interval_divide(patch_count,vmin,vmax,num=50,cls_parse='mean',partition='linear'):
    # this function use all numpy varaiables
    if partition == 'linear':
        step = (vmax-vmin)/(num-1)
        cls_border = np.arange(vmin,vmax,step)
        add = np.array([vmax])
        cls_border = np.concatenate((cls_border,add),axis=0)
    elif partition == 'log':
        step = (np.log(vmax)-np.log(vmin))/(num-1)
        cls_border = np.arange(np.log(vmin),np.log(vmax),step)
        cls_border = np.exp(cls_border)
        add = np.array([vmax])
        cls_border = np.concatenate((cls_border,add),axis=0)  
    elif partition == 'uep':
        cls_border, pl = UEP_border(patch_count,vmin=5e-3,vmax=vmax,num=num,epslon=50)

    
    # get cls2value
    if cls_parse == 'median':
        cls2value = np.zeros(num)
        cls2value[1:] = (cls_border[:-1]+cls_border[1:])/2
    elif cls_parse == 'mean':
        cls2value = np.zeros(num)
        patch_class = Count2Class(patch_count,cls_border[:-1])# should delete the vmax
        for ci in range(1,num):
            tmp_mask = (patch_class==ci)#.float()
            tmp_num = tmp_mask.sum().item()
            if tmp_num<1:
                cls2value[ci] = (cls_border[ci-1]+cls_border[ci])/2
            else:
                tmp_c2v = (patch_count*tmp_mask).sum()/tmp_num
                cls2value[ci] = tmp_c2v.item()
    else:
        print('No class to count method as %s' %(cls_parse))

    return cls_border,cls2value