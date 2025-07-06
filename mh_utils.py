import torch
import numpy as np
from cls_funcs import Count2Class

def div2func(interval,count_dict):
# ----input----
# interval is interval for (vmin,vmax], is a list
# count_dict contain the number of samples within each interval
# ----output----
# we just insert points into it for making [vmin,vmed),(vmed,vmax], as two list
    vmin,vmax = interval[0],interval[1]
    
    # find the index lie in this range
    in_key = []
    in_value = []
    keylist = list(count_dict.keys())
    keylist.sort(reverse=False)
    for tkey in keylist:
        if tkey>=vmin and tkey<vmax:
            if count_dict[tkey]>0:
                in_key.append(tkey)
                in_value.append(count_dict[tkey])
            
    if len(in_key)<2:
        return [interval]
     
    # get the accumulate sum
    in_value = np.array(in_value)
    in_sum = np.cumsum(in_value)
    half_sum = in_sum[-1]/2.0
    
    # find the approximate half split (should avoid the first and the last)
    err = np.abs(in_sum - half_sum)
    minidx = np.argmin(err)
    minidx = minidx+1
    minidx = max(1,min(minidx,len(in_key)-1))
    
    vmed = in_key[minidx]        
            
    
    
    split_list = [[vmin,vmed],[vmed,vmax]]
    # # check the number of objects within each interval
    # num1,num2 = 0.0,0.0
    # for ti,tkey in enumerate(in_key):
    #     if tkey<vmed:
    #         num1+=in_value[ti]
    #     else:
    #         num2+=in_value[ti]
            
    # if num1<1e-6 or num2<1e-6:
    #     return [interval]
    
    return split_list

# all the interval is [), so vmax would be large than the interval_idx
def level_split(interval,count_dict,slevel,if_age=False,ob_vmax=None):
    sinterval = []
    tmp_level_split = [interval]
    for si in range(slevel):
        # if si==2:
        #     print('Watch.')
        sinterval.append([])
        for tmp_interval in tmp_level_split:
            tmp_split = div2func(tmp_interval,count_dict)
            sinterval[-1] = sinterval[-1]+tmp_split
        tmp_level_split = sinterval[-1]
        
        if len(sinterval)>=2 and len(sinterval[-1])==len(sinterval[-2]):
            print("Can only split %d levels." %(si) )    
        
    # change tmp_split into sindice and sclass2count
    sindice = []
    sclass2count = []
    for si in range(slevel):
        sindice.append([])
        sclass2count.append([])
        tmp_intervallist = sinterval[si]
        for idx in range(len(tmp_intervallist)):
            # first class2count
            tmp_class2count = (tmp_intervallist[idx][0]+tmp_intervallist[idx][1])/2.0
            if not (ob_vmax is None):
                if idx == len(tmp_intervallist)-1:
                   tmp_class2count =  (tmp_intervallist[idx][0]+ob_vmax)/2.0
            if if_age:
                tmp_class2count -= 0.5

            sclass2count[-1].append(tmp_class2count)
            # then sindice
            if idx==0:
                continue
            if if_age:
                sindice[-1].append(tmp_intervallist[idx][0]-0.5)
            else:
                sindice[-1].append(tmp_intervallist[idx][0])
            
            
        
    return sinterval, sindice, sclass2count

def get_mh_weight(count_dict,sindice):
    allkey = list(count_dict.keys())
    allkey = torch.Tensor(allkey).float()
    allvalue = list(count_dict.values())
    allvalue = torch.Tensor(allvalue).float()
    sweight = []
    for hi in range(len(sindice)):
        sweight.append([])
        allclass = Count2Class(allkey,torch.Tensor(sindice[hi]) )   
        tmp_cnum = len(sindice[hi])+1
        for ci in range(tmp_cnum):
            tmp_count = ((allclass==ci).float()*allvalue).sum().item()  
            tmp_weight = 1.0/tmp_count if tmp_count>1e-6 else 0.0
            sweight[-1].append(tmp_weight)
        tmp_norm = sum(sweight[-1])
        for ci in range(tmp_cnum):
            sweight[-1][ci] /= tmp_norm
            
    return sweight
        

if __name__ == '__main__':
    interval = [0,9]
    count_dict = {0:100,1:200,2:400,3:200,4:200,5:100,6:100,7:50,8:50}
    split_list = div2func(interval,count_dict)
    
    SLEVEL=5
    sinterval = []
    tmp_split = [interval]
    for si in range(SLEVEL):
        sinterval.append([])
        for tmp_interval in tmp_split:
            tmp_split = div2func(tmp_interval,count_dict)
            sinterval[-1] = sinterval[-1]+tmp_split
        tmp_split = sinterval[-1]
    print(sinterval[0])
    print(sinterval[1])
    print(sinterval[2])
    print(sinterval[3])
    print(sinterval[4])
    
    slevel = 5
    sinterval, sindice, sclass2count = level_split(interval,count_dict,slevel,if_age=True,ob_vmax=8)
    for si in range(slevel):
        print('%d' %(si), sinterval[si])
    for si in range(slevel):
        print('%d' %(si), sindice[si])
    for si in range(slevel):
        print('%d' %(si), sclass2count[si])