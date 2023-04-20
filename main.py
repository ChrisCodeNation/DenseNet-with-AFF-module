#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import sys
sys.path.insert(0, str(Path("../").resolve()))
import islab_gpu as islab

@islab.register     #這行與train function請則一添加，若一起寫到也沒關係
def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
    from torchvision import datasets, transforms
    #from pathlib import Path
    from tqdm import tqdm
    
    import torch.optim as optim
    from torch.optim import lr_scheduler
    
    import torchvision.models as models
    
    import time
    import copy
    import os
    
    from AFF import DenseNet
    
    # training 
    def training(model, dataloader, dataset_size, criterion, optimizer, scheduler, device):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        i = 0
        
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #梯度歸零
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1) 
            loss = criterion(outputs, labels)
            
            #backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)
            running_corrects += (preds==labels).sum().item()
            
        scheduler.step()
        
        return running_loss/dataset_size,  running_corrects/dataset_size
    
    def evaluate(model, dataloader, dataset_size, criterion, device):
        model.eval()
        model.to(device)
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()*inputs.size(0)
                running_corrects += (preds==labels).sum().item()
                
        return running_loss/dataset_size, running_corrects/dataset_size
        
    def testing(model, dataloader, dataset_size, criterion, device):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()*inputs.size(0)
                running_corrects += (preds==labels).sum().item()
                      
        return running_loss/dataset_size, running_corrects/dataset_size
        
    @islab.register
    #def train(target_epoch):
    def start_to_train(history, dataset_size, target_epoch, dataloader, model, criterion, optimizer, scheduler):
        train_epoch = islab.get_train_epoch()       # 一定要寫到，否則代數會每次都重新計算
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"use GPU:{torch.cuda.is_available()} in train")
        print(f"use device:{device}")
        
        model = model.to(device)
        
        for i in range(train_epoch, target_epoch+1):
            print(f'EPOCH {i}/{target_epoch}')
            
            train_loss, train_acc = training(model, dataloader['train'], dataset_size['train'], criterion, optimizer, scheduler, device)
            print(f'train loss:{train_loss} acc:{train_acc}')
            
            val_loss, val_acc = evaluate(model, dataloader['val'], dataset_size['val'], criterion, device)
            print(f'val loss:{val_loss} acc:{val_acc}')
            
            with open('./output_AFF/log.txt', 'a') as f:
                f.write(f'train loss: {train_loss} acc: {train_acc} val loss: {val_loss} acc: {val_acc}')
                f.write('\n')
                
            test_loss, test_acc = testing(model, dataloader['test'], dataset_size['test'], criterion, device)
            print(f'test loss:{test_loss} acc:{test_acc}')

            with open('./output_AFF/log_test.txt', 'a') as f:
                f.write(f'test loss: {test_loss} acc: {test_acc}')
                f.write('\n')
                
            path = './output_AFF/model/' + str(i+22)
            if not os.path.isdir(path):
                os.makedirs(path)
                
            torch.save(model, './output_AFF/model/{}/model.pth'.format(i+22))
            
                
            # 一定要寫到，否則增加過後的代數不會被記錄起來
            islab.add_train_epoch(i)
            
    # loading datasets
    path_train = './HWDB10-11/train'
    path_val = './HWDB10-11/val'
    path_test = './HWDB10-11/test'

    train = Path(path_train)
    val = Path(path_val)
    test = Path(path_test)
    print(train)
    print(val)
    print(test)
    
    vali_Transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))])
        
    train_Transform = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.RandomRotation(degrees=25, fill=255),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                          transforms.GaussianBlur(kernel_size=3),    
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
                                        ])

        
    train_data = datasets.ImageFolder(train, transform = train_Transform)
    val_data = datasets.ImageFolder(val, transform = vali_Transform)
    test_data = datasets.ImageFolder(test, transform = vali_Transform)
    
    dataset_size = {'train':len(train_data), 'val':len(val_data), 'test':len(test_data)}
    print(dataset_size)
    
    train_data = DataLoader(train_data, batch_size = 250, shuffle = True)
    val_data = DataLoader(val_data, batch_size = 250, shuffle = False)
    test_data = DataLoader(test_data, batch_size = 250, shuffle = False)
    
    images, labels = next(iter(test_data))
    print(images.shape)
    print(labels.shape)
    
    print('---------data loading completed------------')
    train_epoch = islab.get_train_epoch() 
    
    
    if train_epoch == 0:
        # loading model
        model = DenseNet(48, (6, 12, 36, 24), 96)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 4052)
    else:
        model = torch.load('./output_AFF/model/'+ str(train_epoch-1+22) +'/model.pth')
    
    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.075, momentum=0.9, weight_decay=0.0001, nesterov=True)
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    
    ###############################################
    dataloader = {'train':train_data, 'val':val_data, 'test':test_data}
    history = dict([[i, {'loss':[], 'acc':[]}] for i in dataloader])
    since = time.time()

    target_epoch = islab.get_max_epoch()
    print('start training--------------------------------')
    #start_to_train(history, dataset_size, target_epoch, dataloader, model, criterion, criterion_cent, optimizer, optimizer_centloss, exp_lr_scheduler)
    start_to_train(history, dataset_size, target_epoch, dataloader, model, criterion, optimizer, scheduler)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    
if (__name__ == "__main__"):
    print(islab.__call_api("GET"))
    main()