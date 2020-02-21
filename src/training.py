#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : february 2020
Script to train and save a convolutional neural network
===========
"""

############| IMPORTS |#############
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, utils

from lib.dataset import FaceEmotionsDataset
from lib.transform import Rescale, RandomCrop, ToTensor, Normalize
from lib.network import Net
####################################

#############| NOTES |##############
"""
Parameter tuning:
    * batch size: a small batch size seems better.
    * epoch number: ...
    * rescale size: ...
    * random cropping size: ...
    * learning rate: ...
    * validation split: ...
"""
####################################

####################################
############| PROGRAM |#############
####################################

if __name__ == "__main__":

    ### Constants ###
    BATCH_SIZE = 4
    EPOCH_NBR = 30
    RESCALE_SIZE = 68
    RDM_CROP_SIZE = 64 
    LEARNING_RATE = 0.002
    VALIDATION_SPLIT = 0.2

    ### Load the dataset ###
    print('> DATASET LOADING')
    data_transform = transforms.Compose([
        Rescale(RESCALE_SIZE),
        RandomCrop(RDM_CROP_SIZE),
        ToTensor()
    ])

    emotions = [
        'neutral',
        'happiness',
        'surprise',
        'sadness',
        'anger',
        'disgust',
        'fear',
        'contempt'
    ]

    dataset = FaceEmotionsDataset(
        csv_file='./src/csv/balanced.csv',
        root_dir='./src/img/',
        classes=emotions,
        transform=data_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    print('>>> %d pictures loaded'%(len(dataloader)*BATCH_SIZE))
    ### Split the dataset ###
    print('> DATASET SPLITTING')
    shuffle_dataset = True
    random_seed = 0

    # Creating data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT*dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

    print('>>> %d pictures for training\n>>> %d pictures for testing'%(len(train_loader)*BATCH_SIZE, len(test_loader)*BATCH_SIZE))

    ### Define the network ###
    net = Net()
    net = net.double()
    ### Define a loss function and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    ### Train the network ###
    print('> STARTED TRAINING | %d epochs'%(EPOCH_NBR))
    for epoch in range(EPOCH_NBR):
        running_loss=0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, emotion_ids = data['image'], data['emotion']
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(images.double())
            loss = criterion(outputs, emotion_ids)
            loss.backward()
            optimizer.step()

            # print statistics
            done = int(30*i/len(train_loader))
            left = 30-done
            progress_bar = '| '+'#'*done+' '*left+' |'

            running_loss += loss.item()
            
            if i % len(train_loader) == len(train_loader)-1:    # print every 200 mini-batches
                print('>>> progress '+progress_bar+' %d/%d mean loss: %.3f' %
                    (epoch + 1, EPOCH_NBR, running_loss / len(train_loader)), end='\r')
                running_loss = 0.0
            else:
                print('>>> progress '+progress_bar+' %d/%d mean loss: %.3f' %
                    (epoch + 1, EPOCH_NBR, running_loss / (i+1)), end='\r')

    print('\n>>> finished training')
    
    ### Test the network ###
    print('> STARTED TESTING')

    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))
    with torch.no_grad():
        for data in test_loader:
            images, emotion_ids = data['image'], data['emotion']
            outputs = net(images.double())
            _, predicted = torch.max(outputs, 1)
            c = (predicted == emotion_ids).squeeze()
            try:
                for i in range(len(emotion_ids)):
                    emotion_id = emotion_ids[i]
                    class_correct[emotion_id] += c[i].item()
                    class_total[emotion_id] += 1
            except:
                emotion_id = emotion_ids[0]
                class_correct[emotion_id] += c.item()
                class_total[emotion_id] += 1

    correct = sum(class_correct)
    total = sum(class_total)
    print('>>> overall accuracy: %d%%' % (100*correct/total))
    for i in range(8):
        print('>>> accuracy for %5s : %2d%%' % (
            emotions[i], 100 * class_correct[i] / class_total[i]))
    
    full_test = True if 'y' == input('Do you want to do a test over the whole database? (y/n): ').lower() else False

    if full_test:
        print('> STARTED FULL TESTING')
        dataset = FaceEmotionsDataset(
            csv_file='./src/csv/cleaned_data.csv',
            root_dir='./src/img/',
            classes=emotions,
            transform=data_transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        print('>>> %d pictures loaded'%(len(dataloader)*BATCH_SIZE))

        class_correct = list(0. for i in range(8))
        class_total = list(0. for i in range(8))
        with torch.no_grad():
            for k, data in enumerate(dataloader, 0):
                images, emotion_ids = data['image'], data['emotion']
                outputs = net(images.double())
                _, predicted = torch.max(outputs, 1)
                c = (predicted == emotion_ids).squeeze()
                try:
                    for i in range(len(emotion_ids)):
                        emotion_id = emotion_ids[i]
                        class_correct[emotion_id] += c[i].item()
                        class_total[emotion_id] += 1
                except:
                    emotion_id = emotion_ids[0]
                    class_correct[emotion_id] += c.item()
                    class_total[emotion_id] += 1
                
                done = int(30*k/len(dataloader))
                left = 30-done
                progress_bar = '| '+'#'*done+' '*left+' |'
                
                print('>>> progress '+progress_bar+' %d/%d' %
                    ((k+1)*BATCH_SIZE, len(dataloader)*BATCH_SIZE), end='\r')
                
        correct = sum(class_correct)
        total = sum(class_total)
        print('\n>>> overall accuracy: %d%%' % (100*correct/total))
        for i in range(8):
            print('>>> accuracy for %5s : %2d%%' % (
                emotions[i], 100 * class_correct[i] / class_total[i]))

    print('>>> finished testing')
    
    save = True if 'y' == input('Do you want to save this network? (y/n): ').lower() else False

    if save:
        PATH = './src/models/'+input('Please name your network save: ')+'.pth'
        torch.save(net.state_dict(), PATH)
        print('>>> network saved')
    
    print('> EXITING')