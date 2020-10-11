import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Conv3D
from loader.dataset import Dataset_subtomo
import sklearn.metrics as skm
from tqdm.auto import tqdm
import numpy as np
import sys

batch_size=64
model = Conv3D().cuda()
##change checkpoint name
SAVE_PATH='./model_save/'

checkpoint = torch.load('{}'.format(sys.argv[1]))
state_dict = checkpoint['state_dict']

model.load_state_dict(state_dict,strict=False)
train_data=Dataset_subtomo('csv_split/train.csv','classification')
test_data=Dataset_subtomo('csv_split/valid.csv','classification')


for params in model.parameters():
    params.requires_grad=False

testloader=DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers=0)
trainloader=DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)

model_test=nn.Sequential(nn.Linear(512,10).cuda())
optimizer=torch.optim.Adam(model_test.parameters(),lr=0.01)
for epoch in range(3):
    total_batches=int(len(trainloader.dataset)//batch_size)
    train_loss=0
    total=0

    model_test.train()
    count=0
    with tqdm(total=total_batches) as pbar:

        for en,(data_array,names) in enumerate(trainloader):
            inputs, targets = data_array['image'].cuda(), data_array['class'].cuda()
            outputs=model_test(model(inputs,test=True))
            optimizer.zero_grad()
            targets=targets.squeeze()
            loss=nn.CrossEntropyLoss()(outputs,targets)
            loss.backward()
            train_loss+=float(loss)
            optimizer.step()

            pbar.update(1)
            break
    print("Training Epoch {} Loss {} ".format(epoch,train_loss/total_batches))

    test_loss=0
    preds=[]
    labels_list=[]
    names_list=[]
    with torch.no_grad():
        model_test.eval()
        total_batches_test=int(len(testloader.dataset)//batch_size)
        with tqdm(total=total_batches_test) as pbar:
            for data_array,names in testloader:
                inputs, targets = data_array['image'].cuda(), data_array['class'].cuda()

                features=model_test(model(inputs,test=True))
                targets=targets.squeeze()
                loss=float(nn.CrossEntropyLoss()(features,targets))
                test_loss+=loss

                targets=targets.unsqueeze(1)
                _, predicted = torch.max(nn.Softmax(dim=1)(features.data), 1)
                predicted=(predicted.unsqueeze(1))
                preds.append(predicted.cpu().numpy())
                labels_list.append(targets.detach().cpu().numpy())

                pbar.update(1)

        labels_list=np.vstack(labels_list)
        preds=np.vstack(preds)


        print( skm.classification_report(labels_list,preds))

    print("Validation Epoch {} Loss {} ".format(epoch,test_loss/total_batches_test))


