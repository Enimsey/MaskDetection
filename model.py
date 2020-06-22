# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:28:51 2020

@author: 33668
"""
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor,Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize
import os
import torch.nn as nn
from torch import device, cuda, float32, long, no_grad, save, load
import torch.nn.functional as F
import torch.optim as optim


def load_dataset(batch_size, train, path):
    data_path = os.getcwd()+path+'\\'
    transform = {
        'train': Compose(
            [Resize([64, 64]),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])]),
        'test': Compose(
            [Resize([64, 64]),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        }
    print(data_path)
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform['train' if train else 'test']
    )
    
    return dataset

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

class Flatten(nn.Module):
        def forward(self, x):
            N = x.shape[0] # read in N, C, H, W\n",
            return x.view(N, -1)  # \"flatten\" the C * H * W values into a single vector per image\n",

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with no_grad():
        for x, y in loader:
            x = x.to(device=dvc, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=dvc, dtype=long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train_model(model, train_set, nb_epochs = 1, epoch_length = 200):
    model.to(device=dvc)
    for ep in range(nb_epochs):
        for t,(x,y) in enumerate(train):
            model.train()  # put model to training mode\n",
            x = x.to(device=dvc, dtype=dtype)  # move to device, e.g. GPU\n",
            y = y.to(device=dvc, dtype=long)
            scores = model(x)
            
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % epoch_length == 0:
                print("epoch: " + str(ep))
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                print("accuracy on validation set: ")
                check_accuracy(validation, model)
                print()
if __name__ == "__main__":    
    #=============================================================================
    NUM_TRAIN = 400
    batch_size, train = 50, True
    dataset_train = load_dataset(batch_size, train, '\\final_dataset\\test')
    
    train = DataLoader(dataset_train, batch_size=50, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=True, timeout=0,
                worker_init_fn=None)
    
    dataset_val = load_dataset(batch_size, train, '\\final_dataset\\val')
    
    validation = DataLoader(dataset_val, batch_size=50, shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=True, timeout=0,
                worker_init_fn=None)
    
    batch_size, do_train = 100, False
    dataset_test = load_dataset(batch_size, do_train, '\\final_dataset\\test')
    
    test = DataLoader(dataset_test, batch_size=50, shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None)
    # #=============================================================================
    dvc = device('cuda')
    if cuda.is_available():
        dvc = device('cuda')
    
    dtype = float32
    channel_0 = 32
    channel_1 = 16
    channel_2 = 8
    channel_3 = 4
    model = nn.Sequential(
            nn.Conv2d(3, channel_0,(7, 7),padding = 3),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(channel_0, channel_1,(5, 5),padding = 2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Conv2d(channel_1, channel_2,(3, 3),padding = 1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(channel_2, channel_3,(3, 3),padding = 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            Flatten(),
            nn.Linear(channel_3 * 64 * 64//16, 2)
        )       
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, nesterov=True)
    # Initialize layers' weights
    for sequential in model.children():
        for layer in sequential.children():
            if isinstance(layer, (Linear, Conv2d)):
                init.kaiming_normal_(layer.weight)
    model.apply(init_weights)
    
    train_model(model, train, 30)
    print("Testing accuracy on test set")
    acc = check_accuracy(validation, model)
        
    save(model.state_dict(), "model_bis"+str(acc)+".pth")
        
    #     model_bis = nn.Sequential(
    #             nn.Conv2d(3, channel_1,(3, 3),padding = 1),
    #             nn.ReLU(),
    #             nn.Dropout2d(p=0.25),
    #             nn.Conv2d(channel_1, channel_2,(3, 3),padding = 1),
    #             nn.ReLU(),
    #             nn.MaxPool2d((2,2)), 
    #             nn.Conv2d(channel_2, channel_3,(3, 3),padding = 1),
    #             nn.ReLU(),
    #             nn.Dropout2d(p=0.25),
    #             Flatten(),
    #             nn.Linear(channel_3 * 32 * 32//4, 2)
    #         )
        
    #     model_bis.load_state_dict(load("model"+str(acc)+".pth"), strict=False)
    
