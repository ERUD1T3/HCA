import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
import torch
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info


class IMDBWIKI(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split

        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        return img, label, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform


    # this is for doing lds in the datastet level
    # age from 0 to 120
    def _prepare_weights(self, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        if reweight=='la':
            reweight='inverse'
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"



        value_dict = {x: 0 for x in range(max_target)} # the dict is the age:number of samples
        labels = self.df['age'].values # this is to get thelabel space
        for label in labels: # get the number of each class
            value_dict[min(max_target - 1, int(label))] += 1
        value_dict_original = value_dict # restore the original value dict
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")
        # 1-add the global weight here
        self.global_weight = torch.Tensor( np.asarray([v for _, v in value_dict.items()]) ).float()
        self.global_weight_original = torch.Tensor( np.asarray([v for _, v in value_dict_original.items()]) ).float()
        
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
            
            smoothed_value_original = convolve1d(
                np.asarray([v for _, v in value_dict_original.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label_original =  [smoothed_value_original[min(max_target - 1, int(label))] for label in labels]
            # 2-add the global weight here
            self.global_weight = torch.Tensor(smoothed_value).float()
            self.global_weight_original = torch.Tensor(smoothed_value_original).float()
            self.global_number =  torch.Tensor(num_per_label_original).float()

        self.global_weightn = torch.where(self.global_weight>=1,1/self.global_weight,0)* (self.global_weight.sum())/len(self.global_weight)

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights) # this is the average weight that is computer for each batcj
        weights = [scaling * x for x in weights]
        return weights


class IMDBWIKI_feats(data.Dataset):
    def __init__(self,  data_dict):
        self.data_dict = data_dict 

    def __len__(self):
        length = self.data_dict['feats'].size()[0]
        return length

    def __getitem__(self, index):
        length = self.data_dict['feats'].size()[0]
        index = index % length
        feat = self.data_dict['feats'][index]
        label = self.data_dict['labels'][index]
        weight = self.data_dict['weights'][index]

        return feat, label, weight