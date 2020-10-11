
import os
import pandas as pd
import torch
import torch.utils.data as data
import mrcfile as mrc
import numpy as np
import scipy.ndimage as nd
from tqdm.notebook import tqdm
import pandas as pd

class Dataset_subtomo(data.Dataset):
    def __init__(self, dir_csv,task='classification',test=False):

        self.image_dirs = pd.read_csv(dir_csv).iloc[:, :].values #from DataFrame to array
        self.names=pd.read_csv(dir_csv).iloc[:, :].values
        self.task=task



    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):

        img_name = self.image_dirs[item][0]
        name=self.names[item][0]
        with mrc.open(img_name, permissive=True) as f:
                img = f.data  # (32, 32, 32)
        img=np.copy(img)
        img = nd.interpolation.zoom(img, zoom=2)
        img = np.expand_dims(img,0)
        if self.task=='classification' or self.task=='all':

            class_label=self.image_dirs[item][2]
            class_label = np.expand_dims(class_label, 0)
            sample = {'image': np.copy(img), 'class':np.copy(class_label),'name':name}

        if self.task=='segmentation' or self.task=='all':
            ##change mask name
            mask_name = './data2_SNR005/processed_densitymap_mrc_mrc2binary/packtarget'+img_name[41:]#self.image_dirs[item][1]

            with mrc.open(mask_name, permissive=True) as f:
                mask = f.data  # (32, 32, 32)
            mask = np.expand_dims(mask, 0)
            mask = nd.interpolation.zoom(mask, zoom=2)
            sample = {'image': np.copy(img), 'mask': np.copy(mask),'name':name}


        return sample,img_name


if __name__=='__main__':
    d1=CustomDataset('csv_split/train.csv','classification',features=False)
    d,name=d1.__getitem__(2)
    image,label=d['image'],d['class']
    print(label,image.shape,name)

