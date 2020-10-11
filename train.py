import sklearn.metrics as skm
from torch.nn import functional as F
import torch
import os
from loader.dataset import Dataset_subtomo
from model import Conv3D
from loss.loss import *
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SAVE_PATH="./model_save/"
def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight,a=0,mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def train(epochs,model,train_data,test_data,optimizer,batch_size):
    lgml= CenterLoss(num_classes=10, feat_dim=512)
    optimizer_centloss = torch.optim.SGD(lgml.parameters(), lr=0.05,momentum=0.9)

    model.train()
    val_loss=[]
    trainloader=DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)
    testloader=DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        loss_2=0
        total_batches=int(len(trainloader.dataset)//batch_size)
        model.train()
        running_mean=0
        with tqdm(total=total_batches) as pbar:
            for batch_idx, (data_array,names) in enumerate(trainloader):

                inputs, targets = data_array['image'].cuda(), data_array['class'].cuda()
                targets=targets.squeeze()
                out_=model(inputs,labels=targets)
                out_conv=out_['conv']
                out_loss=out_['loss']
                loss=lgml(out_conv,targets)
                # print(float(loss))
                loss+=out_loss
                optimizer.zero_grad()
                optimizer_centloss.zero_grad()
                loss.backward()
                optimizer.step()

                optimizer_centloss.step()


                train_loss += float(loss)
                pbar.update(1)


        print("Training Loss {}".format((train_loss/total_batches)))


        with torch.no_grad():
            model.eval()
            loss_=0
            total_batches_val=int(len(testloader.dataset)//batch_size)
            with tqdm(total=total_batches_val) as pbar:
                for batch_idx, (data_array,names) in enumerate(testloader):
                    inputs, targets = data_array['image'].cuda(), data_array['class'].cuda()
                    targets=targets.squeeze()
                    loss_+=float(model(inputs,labels=targets)['loss'])
                    pbar.update(1)

        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        if len(val_loss)!=0:
            if loss_<=min(val_loss):
                state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
                savepath=SAVE_PATH+'checkpoint{}.pt'.format(epoch)
                torch.save(state,savepath)
        else:
            state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
            savepath=SAVE_PATH+'checkpoint{}.pt'.format(epoch)
            torch.save(state,savepath)


        val_loss.append(loss_)
        
        print("Validation Loss {} ".format(loss_/total_batches_val))

        scheduler.step(loss_/total_batches_val)

        with open(SAVE_PATH+'loss.txt', 'a') as f:
                str1="Epoch  "+str(epoch)+"  Training Loss"+str((train_loss/total_batches))+'\t'+"Val loss"+str((loss_/total_batches_val))+'\n'
                f.write(str1)





if __name__ == '__main__':

    epochs=60
    train_data=Dataset_subtomo('csv_split/train.csv','classification')
    test_data=Dataset_subtomo('csv_split/valid.csv','classification')
    batch_size=16
    model = Conv3D().cuda()

    optimizer=torch.optim.Adam(model.parameters(), lr=0.01,betas=(0.9,0.999),weight_decay=0.0005)
    train(epochs,model,train_data,test_data,optimizer,batch_size)



