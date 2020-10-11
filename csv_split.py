import csv
import math
import numpy as np
import os
import sys
import shutil
import random
import re
from glob import glob
import pandas as pd
import re


def atoi(s):
    return int(s) if s.isdigit() else s

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def write_csv(file,valid_lists,get_source):
    w = csv.writer(file)

    for name in valid_lists:
        tomo_name = name
        seg_name = tomo_name.replace('subtomogram_mrc', 'processed_densitymap_mrc_mrc2binary')
        seg_name = seg_name.replace('tomotarget', 'packtarget')
        no = re.findall(r"\d+", tomo_name.split('/')[-1])
        assert len(no) == 1, 'Class Length error'
        class_label = int(no[0]) // 500
        class_data=['1bxn','1f1b','1yg6','2byn','3gl1','4d4r','6t3e','2ldb','2h12','3hhb']
        
        class_data=class_data[class_label]
        SNR_level=get_source
        # print(no, class_label)
        if not os.path.exists(seg_name):
            continue
        w.writerow((tomo_name, seg_name, class_label,SNR_level,class_data))  # attention: the first row defult to tile

def Split_data(data_dir,csv_dir):

    #Split data into train-valid-test = [6:1:3]= [0.6,0.1,0.3]
    # NewData\data1_SNR003\subtomogram_mrc\tomotarget0.mrc
    # NewData\data1_SNR003\processed_densitymap_mrc_mrc2binary\packtarget0.mrc

    sub_file = ['data1_SNR003','data2_SNR005','data3_SNRinfinity']#['data1_SNR003']#


    for sub in sub_file:

        file_list = glob(os.path.join(data_dir,sub,'subtomogram_mrc','*.mrc'))
        

        random.seed(9)
        random.shuffle(file_list)
        total = len(file_list)
        train_lists = file_list[0:int(total*0.6)]
        valid_lists = file_list[int(total*0.6):int(total*0.7)]
        test_lists = file_list[int(total*0.7):total]
        print(total,len(train_lists),len(valid_lists),len(test_lists))



        with open(os.path.join(csv_dir, 'train.csv'), 'a') as file:
            write_csv(file,train_lists,sub[9:])
        with open(os.path.join(csv_dir, 'valid.csv'), 'a') as file:
            write_csv(file,valid_lists,sub[9:])
        with open(os.path.join(csv_dir, 'test.csv'), 'a') as file:
            write_csv(file,test_lists,sub[9:])

if __name__ == '__main__':
    data_dir = '../data_/'
    csv_dir = './csv_split'

    # clear the existing file
    if os.path.isdir(csv_dir):
        shutil.rmtree(csv_dir)

    os.mkdir(csv_dir)
    with open(os.path.join(csv_dir, 'train.csv'), 'a') as f:
        csv.writer(f).writerow(('Input Data','Segmentation Ground Truth','Class','SNR Level','Actual Label'))
    with open(os.path.join(csv_dir, 'valid.csv'), 'a') as f:
        csv.writer(f).writerow(('Input Data','Segmentation Ground Truth','Class','SNR Level', 'Actual Label'))
    with open(os.path.join(csv_dir, 'test.csv'), 'a') as f:
        csv.writer(f).writerow(('Input Data','Segmentation Ground Truth','Class','SNR Level','Actual Label'))

    Split_data(data_dir,csv_dir)

    for csv in ['train.csv', 'test.csv', 'valid.csv']:
        df = pd.read_csv(os.path.join(csv_dir, csv))
        ds = df.sample(frac=1)
        ds.to_csv(os.path.join(csv_dir, csv),index=False)



