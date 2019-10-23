#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import pandas as pd


# In[3]:


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    print("Data Loaded Successfully!")
    return dataloaders, validloaders, testloader, image_datasets


# In[5]:


def build_model(architecture, learning_rate, hidden_units, epochs,device,class_to_idx):
    # TODO: Build and train your network
    print("Building Model...")
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_units = 1024
    else:
         print("Please choose either vgg16 or densenet121 only")
    
    output_units = len(class_to_idx)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, output_units),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device);
    print("Model Built Successfully!")
    return model, criterion, optimizer


# In[6]:


def train_model(epochs, dataloaders, validloaders, model, criterion, optimizer, device):
    print("Training model...")
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloaders):.3f}")
                running_loss = 0
                model.train()
    print("Finished Training")

# In[7]:


def calculate_acc(model, testloader, device):
    # TODO: Do validation on the test set
    print("Calculating Accuracy...")
    valid_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)


            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(testloader):.3f}")


# In[9]:


def save_checkpoint(architecture,model,image_datasets,optimizer,epochs,checkpoint_dir):
    # TODO: Save the checkpoint 
    print("Saving Checkpoint...")
    checkpoint = {'architecture': architecture,
                 'classifier': model.classifier,
                 'state_dict': model.state_dict(),
                 'class_to_idx': image_datasets.class_to_idx,
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epochs': epochs}
    
    torch.save(checkpoint, checkpoint_dir + 'checkpoint.pth')
    print("Checkpoint Saved")


# In[10]:


def load_checkpoint(checkpoint_dir,device):
    print("Loading Checkpoint")
    checkpoint = torch.load(checkpoint_dir)
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif checkpoint['architecture'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_units = 1024
    else:
         print("Please choose either vgg16 or densenet121 only")
    
    for param in model.parameters():
        param.requires_grad = False
    

        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        #epoch = checkpoint['epochs']
        #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)
    print("Checkpoint Loaded")
    return model, model.class_to_idx##############


# In[11]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    print("Processing Image")
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size)
    
    width, height = image.size
    portion = 224
    left = (width - portion)/2
    top = (height - portion)/2
    right = (width + portion)/2
    bottom = (height + portion)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image = np_image/ 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    transpose_image = np_image.transpose((2,0,1))
    
    return transpose_image


# In[15]:


#idx_to_class = {v: k for k, v in model.class_to_idx.items()}##############
def predict(image_path, model, topk,device,idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    
    image =  torch.from_numpy(process_image(image)).to(device)#.type(torch.FloatTensor)
    image = image.unsqueeze_(0)
    image = image.float()
    
    #model.to(device)

    log_ps = model.forward(image)
    ps = torch.exp(log_ps)
    top_p, top_indeces = ps.topk(topk,dim=1)
    top_label = [idx_to_class[index] for index in top_indeces[0].tolist()]
    return top_p, top_label


# In[ ]: