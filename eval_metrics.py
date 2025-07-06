import numpy as np
import torch
from scipy.stats import *
from collections import defaultdict
# =========================================================
# this is for overall metrics
def imbalanced_metrics(preds,labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    mean_mse = np.mean((preds - labels) ** 2)
    mean_mse = np.sqrt(mean_mse)
    mean_l1 = np.mean(np.abs(preds - labels))
    return mean_mse, mean_l1    

# modified 2023/07/02, this function can do all
# this is for shot metrics
def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))
        # this is not reduced version, each element is a list

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    # here is not the bMAE setting, bmae setting will give the same weight to each class
    shot_dict = defaultdict(dict)
    # here is a bad example
    # should consider some_shot do not have samples
    shot_dict['many']['mse'] = -1
    shot_dict['many']['l1'] = -1
    if np.sum(many_shot_cnt)>0:
        shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt) 
        shot_dict['many']['mse'] = np.sqrt(shot_dict['many']['mse'])
        shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)

    shot_dict['median']['mse'] = -1
    shot_dict['median']['l1'] = -1
    if np.sum(median_shot_cnt)>0:
        shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
        shot_dict['median']['mse'] = np.sqrt(shot_dict['median']['mse'])
        shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)

    shot_dict['low']['mse'] = -1
    shot_dict['low']['l1'] = -1  
    if np.sum(low_shot_cnt)>0:
        shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
        shot_dict['low']['mse'] = np.sqrt(shot_dict['low']['mse'])
        shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)

    # compute for the all
    shot_dict['all']['mse'] = -1
    shot_dict['all']['l1'] = -1  
    all_num = np.sum(many_shot_cnt+median_shot_cnt+low_shot_cnt)
    if all_num>0: 
        shot_dict['all']['mse'] = np.sum(many_shot_mse+median_shot_mse+low_shot_mse) / all_num
        shot_dict['all']['mse'] = np.sqrt(shot_dict['all']['mse'])
        shot_dict['all']['l1'] = np.sum(many_shot_l1+median_shot_l1+low_shot_l1) / all_num

    return shot_dict

# =========================================================
# this is for overall metrics
# compute overall balanced metrics
def balanced_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    mse_per_class, l1_per_class = [], []
    for l in np.unique(labels):
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))

    mean_mse = sum(mse_per_class) / len(mse_per_class)
    mean_mse = np.sqrt(mean_mse)
    mean_l1 = sum(l1_per_class) / len(l1_per_class)
    return mean_mse, mean_l1


# compute detailed balanced metrics
def shot_metrics_balanced(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_cnt.append(test_class_count[i])


        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = -1
    shot_dict['many']['l1'] = -1
    if np.sum(many_shot_cnt)>0:
        shot_dict['many']['mse'] = np.sum(many_shot_mse) / len(many_shot_cnt) 
        shot_dict['many']['mse'] = np.sqrt(shot_dict['many']['mse'])
        shot_dict['many']['l1'] = np.sum(many_shot_l1) / len(many_shot_cnt)

    shot_dict['median']['mse'] = -1
    shot_dict['median']['l1'] = -1
    if np.sum(median_shot_cnt)>0:
        shot_dict['median']['mse'] = np.sum(median_shot_mse) / len(median_shot_cnt)
        shot_dict['median']['mse'] = np.sqrt(shot_dict['median']['mse'])
        shot_dict['median']['l1'] = np.sum(median_shot_l1) / len(median_shot_cnt)

    shot_dict['low']['mse'] = -1
    shot_dict['low']['l1'] = -1  
    if np.sum(low_shot_cnt)>0:
        shot_dict['low']['mse'] = np.sum(low_shot_mse) / len(low_shot_cnt)
        shot_dict['low']['mse'] = np.sqrt(shot_dict['low']['mse'])
        shot_dict['low']['l1'] = np.sum(low_shot_l1) / len(low_shot_cnt)

    # compute for the all
    shot_dict['all']['mse'] = -1
    shot_dict['all']['l1'] = -1   
    all_num = len(many_shot_cnt+median_shot_cnt+low_shot_cnt)
    if all_num>0: 
        shot_dict['all']['mse'] = np.sum(many_shot_mse+median_shot_mse+low_shot_mse) / all_num
        shot_dict['all']['mse'] = np.sqrt(shot_dict['all']['mse'])
        shot_dict['all']['l1'] = np.sum(many_shot_l1+median_shot_l1+low_shot_l1) / all_num
    return shot_dict

# =========================================================
# this is the shot metric extend for any input, contain both balanced and imbanlanced results
def balanced_extend_metrics(inputs, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(inputs)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.mean(inputs[labels == l]) )


    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []
    all_shot_mse = []
    all_shot_cnt = []


    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_cnt.append(test_class_count[i])

        all_shot_mse.append(mse_per_class[i])
        all_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['bal'] = -1
    shot_dict['median']['bal'] = -1
    shot_dict['low']['bal'] = -1
    shot_dict['all']['bal'] = -1
    shot_dict['many']['imbal'] = -1
    shot_dict['median']['imbal'] = -1
    shot_dict['low']['imbal'] = -1
    shot_dict['all']['imbal']  = -1
    if np.sum(many_shot_cnt)>0:
        shot_dict['many']['bal'] = np.sum(many_shot_mse) / len(many_shot_mse)
        shot_dict['many']['imbal'] = np.sum(np.array(many_shot_mse)*np.array(many_shot_cnt)) / np.sum(many_shot_cnt)
    if np.sum(median_shot_cnt)>0:
        shot_dict['median']['bal'] = np.sum(median_shot_mse) / len(median_shot_mse)
        shot_dict['median']['imbal'] = np.sum(np.array(median_shot_mse)*np.array(median_shot_cnt)) / np.sum(median_shot_cnt)
    if np.sum(low_shot_cnt)>0:
        shot_dict['low']['bal'] = np.sum(low_shot_mse) / len(low_shot_mse)
        shot_dict['low']['imbal'] = np.sum(np.array(low_shot_mse)*np.array(low_shot_cnt)) / np.sum(low_shot_cnt)

    if np.sum(many_shot_cnt+median_shot_cnt+low_shot_cnt)>0:
        shot_dict['all']['bal'] = np.sum(all_shot_mse) / len(all_shot_mse)
        shot_dict['all']['imbal'] = np.sum(np.array(all_shot_mse)*np.array(all_shot_cnt)) / np.sum(all_shot_cnt)

    return shot_dict

# =========================================================